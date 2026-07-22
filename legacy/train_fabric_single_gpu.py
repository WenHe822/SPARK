import os
import gc
import datetime
import math
import logging
import torch
import hydra
import wandb
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import sys

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss, ssim, tv_3d_loss
from r2_gaussian.gaussian import render, query
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset
from eval import evaluate_model

# ----------------------------
# 正则化函数
# ----------------------------
def density_regularization(density, threshold=0.1, lambda_density=5):
    """
    密度正则: 对低于 threshold 的密度施加惩罚
    """
    penalty = torch.relu(threshold - density)
    loss_density_reg = lambda_density * torch.mean(penalty)
    return loss_density_reg

def xyz_boundary_regularization(xyz, min_coord=-1.0, max_coord=1.0, lambda_xyz_boundary=1.0):
    """
    惩罚超出 [min_coord, max_coord] 边界的 xyz 坐标。

    Args:
        xyz (torch.Tensor): 预测的世界坐标，形状 (B, N_points, 3) 或类似。
        min_coord (float): 坐标允许的最小值。
        max_coord (float): 坐标允许的最大值。
        lambda_xyz_boundary (float): 正则化强度权重。

    Returns:
        torch.Tensor: 正则化损失值。
    """
    # 计算超出下边界的量 (对于 x < min_coord, 计算 min_coord - x)
    lower_penalty = torch.relu(min_coord - xyz)
    # 计算超出上边界的量 (对于 x > max_coord, 计算 x - max_coord)
    upper_penalty = torch.relu(xyz - max_coord)

    # 对所有点和所有维度计算平均惩罚
    boundary_violation = (lower_penalty + upper_penalty).mean()

    return lambda_xyz_boundary * boundary_violation


# ----------------------------
# 日志设置
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.setLevel(logging.ERROR)
    logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    logger.info(f"日志将保存到: {os.path.join(log_dir, 'training.log')}")

def is_wandb_enabled():
    """
    辅助函数：判断 wandb 是否可用或是否被显式禁用
    """
    return os.getenv("WANDB_DISABLED", "false").lower() != "true"

def init_training(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    precision = "16-mixed" if cfg.general.mixed_precision else None
    fabric = Fabric(accelerator="cuda", devices=1, strategy="auto", precision=precision)
    fabric.launch()
    
    device = safe_state(cfg)
    
    run_name = getattr(cfg.logging, "wandb_run_name", "3")
    log_dir = os.path.join(os.getcwd(), "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)
    
    disable_wandb = getattr(cfg.logging, "disable_wandb", False)
    if disable_wandb or os.getenv("DISABLE_WANDB", "false").lower() == "true":
        logger.info("WandB已禁用，仅使用本地日志")
        os.environ["WANDB_DISABLED"] = "true"
        return fabric, device, log_dir
    
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    project_name = getattr(cfg.logging, "wandb_project", "SingleProjRecon")
    wandb_offline = getattr(cfg.logging, "wandb_offline", False)
    
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        logger.info("WandB将在离线模式下运行")
    
    os.environ["WANDB_START_METHOD"] = "thread"
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            resume="allow",
            reinit=True,
            config=wandb_config,
            settings=wandb.Settings(console="off")
        )
        logger.info(f"WandB项目: {project_name}, 运行名称: {run_name}")
    except Exception as e:
        logger.error(f"WandB初始化失败: {e}")
        logger.info("将继续训练，但不会记录到WandB")
    
    return fabric, device, log_dir

# ----------------------------
# 数据加载与 collate
# ----------------------------
def ct_collate_fn(batch, cfg):
    input_images_count = cfg.data.input_images
    
    batch_input_images = []
    batch_camera_angles = []
    batch_view_to_world = []
    scanner_cfg_list = []
    bbox_list = []
    batch_source_cv2wT_quat = []
    cameras_list = []

    for sample in batch:
        # 增加对 cameras 数量的检查
        if len(sample["cameras"]) < input_images_count:
            raise ValueError(f"Sample cameras数量不足，需要至少 {input_images_count} 个，实际得到 {len(sample['cameras'])}")
        cameras = sample["cameras"]
        cameras_list.append(cameras)

        sample_images = []
        sample_angles = []
        sample_v2w = []
        for cam in cameras[:input_images_count]:
            sample_images.append(cam.original_image)
            sample_angles.append(cam.angle)
            sample_v2w.append(cam.view_world_transform)
        
        sample_images = torch.stack(sample_images, dim=0)
        batch_input_images.append(sample_images)
        batch_camera_angles.append(torch.tensor(sample_angles, dtype=torch.float32))
        batch_view_to_world.append(torch.stack(sample_v2w, dim=0))

        scanner_cfg_list.append(sample["scanner_cfg"])
        bbox_list.append(sample["bbox"])
        batch_source_cv2wT_quat.append(sample["source_cv2wT_quat"][:input_images_count])

    batch_input_images = torch.stack(batch_input_images, dim=0)
    batch_camera_angles = torch.stack(batch_camera_angles, dim=0)
    batch_view_to_world = torch.stack(batch_view_to_world, dim=0)
    batch_bbox = torch.stack(bbox_list, dim=0)
    batch_source_cv2wT_quat = torch.stack(batch_source_cv2wT_quat, dim=0)
    
    batch_camera_params = {
        "angle": batch_camera_angles,
        "view_to_world": batch_view_to_world,
    }
    
    return {
        "input_images": batch_input_images,
        "camera_params": batch_camera_params,
        "scanner_cfg": scanner_cfg_list,
        "bbox": batch_bbox,
        "source_cv2wT_quat": batch_source_cv2wT_quat,
        "cameras": cameras_list
    }

def get_dataloaders(cfg: DictConfig, fabric: Fabric):
    logger.info("初始化数据集...")
    train_dataset = get_dataset(data_path=cfg.data.data_path, type="train")
    test_dataset = get_dataset(data_path=cfg.data.data_path, type="test")
    logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    
    batch_size = getattr(cfg.data, "batch_size", 4)
    num_workers = getattr(cfg.data, "num_workers", 0)  # 默认改为0，避免多进程CUDA张量共享问题
    pin_memory = getattr(cfg.data, "pin_memory", False)  # 默认为False，避免内存峰值过高
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_mem < 24:  # 适当调整显存阈值
            batch_size = min(batch_size, 2)
            num_workers = 0
            logger.info(f"GPU内存较小，调整批次大小: {batch_size}, 工作进程: {num_workers}")
    
    import functools
    collate_fn_with_cfg = functools.partial(ct_collate_fn, cfg=cfg)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_with_cfg,
        persistent_workers=False,  # 避免持久化工作进程
        multiprocessing_context='spawn' if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn_with_cfg
    )
    
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)
    
    return train_loader, test_loader

# ----------------------------
# 模型与优化器
# ----------------------------
def init_model(cfg: DictConfig, fabric: Fabric, device):
    # 启用cudnn算法自动选择和内存优化
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = True
    
    # 使用更保守的内存使用方式，避免过度内存占用
    model = GaussianSplatPredictor(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params}")
    model = model.to(memory_format=torch.channels_last)
    
    # 优化模型内存使用
    for param in model.parameters():
        param.grad = None  # 初始化梯度为None而不是零张量，节约内存
    
    param_groups = [{'params': model.network.parameters(), 'lr': cfg.opt.base_lr}]
    weight_decay = getattr(cfg.opt, "weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=cfg.opt.base_lr, 
        eps=1e-8, 
        betas=cfg.opt.betas,
        weight_decay=weight_decay
    )
    
    # 检查是否开启混合精度
    if getattr(cfg.general, "mixed_precision", False):
        logger.info("已配置混合精度训练，可有效减少内存使用...")
    
    model, optimizer = fabric.setup(model, optimizer)
    
    total_steps = cfg.opt.iterations
    warmup_steps = cfg.opt.warmup_steps
    min_lr_factor = 0.001
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return 0.1 + 0.9 * (current_step / warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
     
    ema = None
    if getattr(cfg.opt.ema, "use", False):
        from ema_pytorch import EMA
        ema = EMA(
            model,
            beta=cfg.opt.ema.beta,
            update_every=cfg.opt.ema.update_every,
            update_after_step=cfg.opt.ema.update_after_step
        )
        ema = fabric.to_device(ema)
    
    # 清理初始化过程中的缓存
    torch.cuda.empty_cache()
    
    return model, optimizer, scheduler, ema

def load_checkpoint(model, optimizer, scheduler, ema, log_dir, device, cfg: DictConfig):
    first_iter = 0
    best_metric = 0.0
    ckpt_path = None
    if cfg.opt.pretrained_ckpt is not None:
        pretrained_path = cfg.opt.pretrained_ckpt
        if os.path.isfile(pretrained_path):
            ckpt_path = pretrained_path
        elif os.path.isdir(pretrained_path):
            for ckpt_name in ["model_interrupt.pth", "model_best.pth", "model_latest.pth"]:
                candidate_path = os.path.join(pretrained_path, ckpt_name)
                if os.path.isfile(candidate_path):
                    ckpt_path = candidate_path
                    break
    if ckpt_path is None:
        interrupt_ckpt = os.path.join(log_dir, "model_interrupt.pth")
        if os.path.isfile(interrupt_ckpt):
            ckpt_path = interrupt_ckpt
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        logger.info(f"加载检查点: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            logger.info("使用strict=False加载模型权重")
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.error(f"加载优化器状态失败: {e}")
        if "scheduler_state_dict" in checkpoint and hasattr(scheduler, 'load_state_dict'):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.error(f"加载调度器状态失败: {e}")
        if ema is not None and "ema_state_dict" in checkpoint:
            try:
                ema.load_state_dict(checkpoint["ema_state_dict"])
            except Exception as e:
                logger.error(f"加载EMA状态失败: {e}")
        first_iter = checkpoint["iteration"]
        best_metric = checkpoint.get("best_metric", 0.0)
        logger.info(f"从迭代 {first_iter} 继续训练，最佳指标: {best_metric:.4f}")
    else:
        logger.info("未找到检查点，从头开始训练")
    
    return first_iter, best_metric

# ----------------------------
# 日志记录函数
# ----------------------------
def log_metrics(loss_dict, iteration, current_lr):
    log_data = {
        "总损失": loss_dict["total_loss"].item(),
        "投影损失": loss_dict["reproj_loss"].item(),
        "SSIM损失": loss_dict.get("ssim_loss", 0.0) if isinstance(loss_dict.get("ssim_loss"), torch.Tensor) else 0.0,
        "TV损失": loss_dict.get("tv_loss", 0.0) if isinstance(loss_dict.get("tv_loss"), torch.Tensor) else 0.0,
        "学习率": current_lr,
        "分割区域损失": loss_dict.get("mask_reg_loss", 0.0) if isinstance(loss_dict.get("mask_reg_loss"), torch.Tensor) else 0.0,
    }
    
    # 添加这些行，将损失信息写入日志
    logger.info(f"迭代 {iteration} - 总损失: {log_data['总损失']:.4f}, 投影损失: {log_data['投影损失']:.4f}, SSIM损失: {log_data['SSIM损失']:.4f}, 分割区域损失：{log_data['分割区域损失']:.4f}学习率: {current_lr:.8f}")
    
    # 其他正则损失
    if "density_reg_loss" in loss_dict and isinstance(loss_dict["density_reg_loss"], torch.Tensor):
        log_data["密度正则化损失"] = loss_dict["density_reg_loss"].item()
        logger.info(f"迭代 {iteration} - 密度正则化损失: {log_data['密度正则化损失']:.4f}")
    
    if is_wandb_enabled():
        try:
            wandb.log(log_data, step=iteration)
        except Exception as e:
            logger.error(f"wandb记录失败: {e}")

def log_parameter_histograms(gaussian_splats, iteration):
    """
    记录高斯点参数的分布直方图到wandb
    
    该函数从网络输出的高斯参数中提取主要参数（密度，偏移，缩放，旋转，xyz坐标）的统计信息，
    计算它们的最大值、最小值、平均值，并创建分布直方图记录到wandb，方便可视化参数分布
    变化趋势，以及监控训练过程中的参数稳定性。
    
    Args:
        gaussian_splats (dict): 包含高斯参数的字典，来自网络输出，应包含以下键：
                               "density", "offset", "scaling", "rotation", "xyz"
        iteration (int): 当前迭代次数，用于wandb记录
    
    记录到wandb的指标包括：
        - 密度分布/偏移分布/缩放分布/旋转分布: 各参数值的直方图分布
        - 密度最大值/最小值/平均值: 密度参数的统计量
        - 偏移最大值/最小值/平均值: 偏移参数的统计量
        - 缩放最大值/最小值/平均值: 缩放参数的统计量
        - 旋转最大值/最小值/平均值: 旋转参数的统计量
        - x/y/z坐标分布: 三个坐标轴方向的分布直方图
        - x/y/z坐标最大值/最小值/平均值: 三个坐标轴的统计量
        - x/y/z坐标范围: 三个坐标轴的取值范围大小(最大值-最小值)
    """
    if not is_wandb_enabled():
        return
    
    histogram_dict = {}
    
    # 取第一个batch的数据作为样本
    sample_idx = 0  # 仅使用第一个样本进行统计
    try:
        # 密度直方图
        if "density" in gaussian_splats:
            density_tensor = gaussian_splats["density"][sample_idx].contiguous()
            # 确保张量是连续的，并且被flatten为一维
            density_values = density_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["密度分布"] = wandb.Histogram(density_values)
            histogram_dict["密度最大值"] = float(density_values.max())
            histogram_dict["密度最小值"] = float(density_values.min())
            histogram_dict["密度平均值"] = float(density_values.mean())
        
        # 偏移直方图
        if "offset" in gaussian_splats:
            offset_tensor = gaussian_splats["offset"][sample_idx].contiguous()
            offset_values = offset_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["偏移分布"] = wandb.Histogram(offset_values)
            histogram_dict["偏移最大值"] = float(offset_values.max())
            histogram_dict["偏移最小值"] = float(offset_values.min())
            histogram_dict["偏移平均值"] = float(offset_values.mean())
        
        # 缩放直方图
        if "scaling" in gaussian_splats:
            scaling_tensor = gaussian_splats["scaling"][sample_idx].contiguous()
            scaling_values = scaling_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["缩放分布"] = wandb.Histogram(scaling_values)
            histogram_dict["缩放最大值"] = float(scaling_values.max())
            histogram_dict["缩放最小值"] = float(scaling_values.min())
            histogram_dict["缩放平均值"] = float(scaling_values.mean())
        
        # 旋转直方图
        if "rotation" in gaussian_splats:
            rotation_tensor = gaussian_splats["rotation"][sample_idx].contiguous()
            rotation_values = rotation_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["旋转分布"] = wandb.Histogram(rotation_values)
            histogram_dict["旋转最大值"] = float(rotation_values.max())
            histogram_dict["旋转最小值"] = float(rotation_values.min())
            histogram_dict["旋转平均值"] = float(rotation_values.mean())
        
        # xyz坐标直方图
        if "xyz" in gaussian_splats:
            xyz_tensor = gaussian_splats["xyz"][sample_idx].contiguous()
            
            # 单独记录x、y、z三个方向的分布
            x_values = xyz_tensor[:, 0].detach().cpu().numpy()
            y_values = xyz_tensor[:, 1].detach().cpu().numpy()
            z_values = xyz_tensor[:, 2].detach().cpu().numpy()
            
            # 记录每个方向的直方图
            histogram_dict["x坐标分布"] = wandb.Histogram(x_values)
            histogram_dict["y坐标分布"] = wandb.Histogram(y_values)
            histogram_dict["z坐标分布"] = wandb.Histogram(z_values)
            
            # 记录每个方向的最大值、最小值和平均值
            histogram_dict["x坐标最大值"] = float(x_values.max())
            histogram_dict["x坐标最小值"] = float(x_values.min())
            histogram_dict["x坐标平均值"] = float(x_values.mean())
            
            histogram_dict["y坐标最大值"] = float(y_values.max())
            histogram_dict["y坐标最小值"] = float(y_values.min())
            histogram_dict["y坐标平均值"] = float(y_values.mean())
            
            histogram_dict["z坐标最大值"] = float(z_values.max())
            histogram_dict["z坐标最小值"] = float(z_values.min())
            histogram_dict["z坐标平均值"] = float(z_values.mean())
            
            # 记录坐标范围的大小
            histogram_dict["x坐标范围"] = float(x_values.max() - x_values.min())
            histogram_dict["y坐标范围"] = float(y_values.max() - y_values.min())
            histogram_dict["z坐标范围"] = float(z_values.max() - z_values.min())
        
        try:
            wandb.log(histogram_dict, step=iteration)
        except Exception as e:
            logger.error(f"wandb记录直方图失败: {e}")
    
    except Exception as e:
        logger.error(f"处理参数直方图时出错: {e}")

def log_visualizations(loss_dict, iteration, cfg):
    if not is_wandb_enabled():
        # 若 wandb 被禁用，直接跳过可视化
        return
    
    log_dict = {}
    if loss_dict.get("rendered_images") is not None:
        # fg_weight = getattr(cfg.opt, "foreground_weight", 5.0)
        # bg_weight = getattr(cfg.opt, "background_weight", 0.1)
        # 确保灰度图正确显示
        render_vis = loss_dict["rendered_images"][0].squeeze().detach().cpu().numpy()
        gt_vis = loss_dict["gt_images"][0].squeeze().detach().cpu().numpy()
        
        # 处理灰度图显示
        if render_vis.ndim == 2:
            log_dict["渲染视角0"] = wandb.Image(render_vis, mode="L")
        else:
            log_dict["渲染视角0"] = wandb.Image(render_vis)
            
        if gt_vis.ndim == 2:
            log_dict["真实视角0"] = wandb.Image(gt_vis, mode="L")
        else:
            log_dict["真实视角0"] = wandb.Image(gt_vis)
        
        # if loss_dict.get("mask_images") is not None:
        #     mask_vis = loss_dict["mask_images"][0].squeeze().detach().cpu().numpy()
        #     # 确保二类分割掩码正确显示
        #     log_dict["掩码视角0"] = wandb.Image(
        #         mask_vis, 
        #         mode="L",
        #         caption=f"前景权重={fg_weight}, 背景权重={bg_weight}"
        #     )
            
        #     # 创建权重可视化
        #     weight_mask_vis = np.where(mask_vis > 0.5, fg_weight, bg_weight)
        #     normalized_weight_mask = (weight_mask_vis - bg_weight) / (fg_weight - bg_weight)
        #     log_dict["权重掩码视角0"] = wandb.Image(
        #         normalized_weight_mask,
        #         mode="L",
        #         caption=f"前景={fg_weight}, 背景={bg_weight}"
        #     )
    
    if (loss_dict.get("vol_pred") is not None and 
        isinstance(loss_dict["vol_pred"], torch.Tensor) and
        loss_dict["vol_pred"].dim() >= 4):
        vol_pred = loss_dict["vol_pred"]
        mid_z = vol_pred.shape[-3] // 2
        mid_y = vol_pred.shape[-2] // 2
        mid_x = vol_pred.shape[-1] // 2

        axial_slice = vol_pred[..., mid_z, :, :].squeeze().detach().cpu().numpy()
        coronal_slice = vol_pred[..., :, mid_y, :].squeeze().detach().cpu().numpy()
        sagittal_slice = vol_pred[..., :, :, mid_x].squeeze().detach().cpu().numpy()

        log_dict["轴向切片"] = wandb.Image(axial_slice)
        log_dict["冠状切片"] = wandb.Image(coronal_slice)
        log_dict["矢状切片"] = wandb.Image(sagittal_slice)
        
        save_interval = getattr(cfg.logging, "volume_save_interval", 1000)
        if iteration % save_interval == 0 or iteration <= 10:
            # 将体数据另存
            volumes_dir = os.path.join(os.getcwd(), "logs", wandb.run.name, "volumes")
            os.makedirs(volumes_dir, exist_ok=True)
            vol_data = vol_pred.squeeze().detach().cpu().numpy()
            vol_path = os.path.join(volumes_dir, f"volume_iter_{iteration:06d}.npy")
            np.save(vol_path, vol_data)
    
    if log_dict:
        try:
            wandb.log(log_dict, step=iteration)
        except Exception as e:
            logger.error(f"wandb可视化记录失败: {e}")

def save_checkpoint(model, optimizer, scheduler, ema, iteration, metric, log_dir, filename):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": (ema.ema_model if ema is not None else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": metric,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
    os.makedirs(log_dir, exist_ok=True)
    ckpt_path = os.path.join(log_dir, filename)
    torch.save(checkpoint, ckpt_path)
    logger.info(f"模型已保存到: {ckpt_path}，迭代: {iteration}, 指标: {metric:.4f}")

# ----------------------------
# 训练循环
# ----------------------------
def train_one_epoch(fabric, train_loader, model, optimizer, scheduler, ema, cfg, device, iteration, log_dir):
    """
    单个 epoch 训练逻辑，这里根据 accumulation_steps 实现梯度累积，
    并仅在 step 时更新 iteration。
    """
    model.train()
    model.current_iter = iteration

    accumulation_steps = getattr(cfg.opt, "accumulation_steps", 1)
    mem_clean_interval = getattr(cfg.opt, "mem_clean_interval", 100)  # 降低清理间隔
    total_steps = cfg.opt.iterations
    
    # 仅当执行 optimizer.step() 时，才增加 effective_iter
    effective_iter = iteration
    pbar = tqdm(total=len(train_loader), desc="训练", dynamic_ncols=True, leave=False, position=0, file=sys.stdout)

    # 训练前清理内存
    torch.cuda.empty_cache()
    gc.collect()

    for batch_idx, data in enumerate(train_loader):
        if effective_iter > total_steps:
            break
        
        batch_size = len(data["cameras"])
        input_images = data["input_images"].to(device, non_blocking=True)
        source_cv2wT_quat = data["source_cv2wT_quat"].to(device, non_blocking=True)

        angles = data["camera_params"]["angle"].to(device, non_blocking=True)
        view_to_world = data["camera_params"]["view_to_world"].to(device, non_blocking=True)
        camera_params_list = []
        for i in range(batch_size):
            for j in range(cfg.data.input_images):
                cp = {"angle": float(angles[i,j].item()),
                      "view_to_world": view_to_world[i,j]}
                camera_params_list.append(cp)
        scanner_cfg_list = []
        for i in range(batch_size):
            for j in range(cfg.data.input_images):
                scanner_cfg_list.append(data["scanner_cfg"][i])

        gaussian_splats = model(input_images, source_cv2wT_quat, camera_params_list, scanner_cfg_list)
        
        # 存储当前batch的gaussian_splats用于直方图统计
        first_batch_gaussian_splats = gaussian_splats.copy()

        # 准备损失项累加
        batch_reproj_loss = 0.0
        batch_ssim_loss = 0.0
        batch_tv_loss = 0.0
        batch_density_reg_loss = 0.0
        batch_offset_reg_loss = 0.0
        batch_scaling_reg_loss = 0.0
        batch_rotation_reg_loss = 0.0

        first_sample_rendered = None
        first_sample_gt = None
        first_sample_vol_pred = None
        # first_sample_mask = None

        # 读取或默认获取
        fg_weight = getattr(cfg.opt, "foreground_weight", 5.0)
        bg_weight = getattr(cfg.opt, "background_weight", 0.1)
        density_threshold = getattr(cfg.opt, "density_threshold", 3)
        lambda_density = 0.1
        use_density_reg = False
        use_offset_reg  = False
        use_scaling_reg = False
        use_rotation_reg = False
        s_min = getattr(cfg.opt, "scaling_min", 0.05)
        s_max = getattr(cfg.opt, "scaling_max", 0.2)

        # 损失权重
        w_reproj = getattr(cfg.opt, "w_l12", 1.0)
        w_ssim   = getattr(cfg.opt, "w_ssim", 0.0)
        w_tv     = getattr(cfg.opt, "w_tv",   0.0)

        # 遍历 batch 内每个 sample
        for sample_idx in range(batch_size):
            if effective_iter > total_steps:
                break

            gaussian_splat_sample = {k: v[sample_idx].contiguous()
                                     for k, v in gaussian_splats.items()}
            cameras = data["cameras"][sample_idx]
            scanner_cfg = data["scanner_cfg"][sample_idx]
            bbox = data["bbox"][sample_idx]

            # 四种正则，仅在对应开关和权重非零时计算
            if use_density_reg and "density" in gaussian_splat_sample:
                density = gaussian_splat_sample["density"]
                sample_density_reg = density_regularization(
                    density, threshold=density_threshold, lambda_density=lambda_density
                )
                batch_density_reg_loss += sample_density_reg
            
            if use_offset_reg and "offset" in gaussian_splat_sample:
                offset_val = gaussian_splat_sample["offset"]
                sample_offset_reg = offset_regularization(
                    offset_val, lambda_offset=getattr(cfg.opt, "lambda_offset", 0.01)
                )
                batch_offset_reg_loss += sample_offset_reg
            
            if use_scaling_reg and "scaling" in gaussian_splat_sample:
                scaling_val = gaussian_splat_sample["scaling"]
                sample_scaling_reg = scaling_regularization(
                    scaling_val, s_min=s_min, s_max=s_max,
                    lambda_scale=getattr(cfg.opt, "lambda_scale", 0.01)
                )
                batch_scaling_reg_loss += sample_scaling_reg
            
            if use_rotation_reg and "rotation" in gaussian_splat_sample:
                rotation_val = gaussian_splat_sample["rotation"]
                sample_rotation_reg = rotation_regularization(
                    rotation_val, lambda_rot=getattr(cfg.opt, "lambda_rot", 0.01)
                )
                batch_rotation_reg_loss += sample_rotation_reg

            # 随机选择输入
            total_cameras = len(cameras)
            imgs_per_obj = getattr(cfg.opt, "imgs_per_obj", 6)
            if total_cameras > imgs_per_obj:
                indices = torch.randperm(total_cameras)[:imgs_per_obj]
                selected_cameras = [cameras[i] for i in indices]
            else:
                selected_cameras = cameras
            
            # 渲染选定相机
            rendered_images = []
            gt_images = []
            
            for cam in selected_cameras:
                render_out = render(cam, gaussian_splat_sample)
                rendered_images.append(render_out["render"].unsqueeze(0))
                gt_images.append(cam.original_image.unsqueeze(0))
                # 移除mask相关代码
                # if hasattr(cam, 'mask_image') and cam.mask_image is not None:
                #     mask_images.append(cam.mask_image.unsqueeze(0))
                # else:
                #     mask_images.append(torch.ones_like(cam.original_image).unsqueeze(0))

            rendered_images = torch.cat(rendered_images, dim=0).to(device)
            gt_images = torch.cat(gt_images, dim=0).to(device)
            # mask_images = torch.cat(mask_images, dim=0).to(device)

            # 移除前景/背景加权
            # weight_mask = torch.where(mask_images > 0, fg_weight, bg_weight)

            # 根据选择计算 L1 或 L2
            if cfg.opt.loss == "l2":
                pixel_loss = (rendered_images - gt_images) ** 2
                sample_reproj = pixel_loss.mean()
            else:
                pixel_loss = torch.abs(rendered_images - gt_images)
                sample_reproj = pixel_loss.mean()

            # 计算 SSIM
            sample_ssim_val = 0.0
            if w_ssim > 0.0:
                # 将 (1 - SSIM) 乘上掩码平均，这里简单直接
                sample_ssim_val = (1.0 - ssim(rendered_images, gt_images)) 

            # 计算 TV
            sample_tv_val = 0.0
            if w_tv > 0.0:
                nVoxel = scanner_cfg.get("nVoxel", [32, 32, 32])
                sVoxel = scanner_cfg.get("sVoxel", [32, 32, 32])
                tv_vol_nVoxel = torch.tensor(nVoxel, device=device)
                tv_vol_sVoxel = torch.tensor(sVoxel, device=device)
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + \
                                (bbox[1] - tv_vol_sVoxel - bbox[0]) * torch.rand(3, device=device)
                vol_pred = query(gaussian_splat_sample, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel)["vol"]
                sample_tv_val = tv_3d_loss(vol_pred, reduction="mean")
                if sample_idx == 0:
                    first_sample_vol_pred = vol_pred

            batch_reproj_loss += sample_reproj
            batch_ssim_loss += sample_ssim_val
            batch_tv_loss += sample_tv_val

            if sample_idx == 0:
                first_sample_rendered = rendered_images
                first_sample_gt = gt_images
                # 移除mask相关代码
                # first_sample_mask = mask_images
                
            # 每个样本处理后及时清理中间变量
            del rendered_images, gt_images
            # 移除mask相关代码
            # del rendered_images, gt_images, mask_images, weight_mask
            if sample_tv_val > 0.0:
                del vol_pred

        # 平均每个 sample 的损失
        reproj_loss = batch_reproj_loss / batch_size
        ssim_loss = batch_ssim_loss / batch_size if w_ssim > 0 else 0.0
        tv_loss = batch_tv_loss / batch_size if w_tv > 0 else 0.0
        density_reg_loss = batch_density_reg_loss / batch_size if use_density_reg else 0.0
        offset_reg_loss = batch_offset_reg_loss / batch_size if use_offset_reg else 0.0
        scaling_reg_loss = batch_scaling_reg_loss / batch_size if use_scaling_reg else 0.0
        rotation_reg_loss = batch_rotation_reg_loss / batch_size if use_rotation_reg else 0.0

        # 组装总损失
        total_loss = (w_reproj * reproj_loss +
                      w_ssim   * ssim_loss +
                      w_tv     * tv_loss +
                      density_reg_loss +
                      offset_reg_loss +
                      scaling_reg_loss +
                      rotation_reg_loss)

        # 检查 NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"迭代 {effective_iter} 产生 NaN 或 Inf 损失，跳过该 batch。")
            continue

        # 反向传播计算梯度
        try:
            # 尝试使用save_on_cpu=False避免保存中间计算图
            with torch.autograd.graph.save_on_cpu(False):
                fabric.backward(total_loss)
        except (AttributeError, ImportError):
            # 如果PyTorch版本不支持该API，则直接执行反向传播
            fabric.backward(total_loss)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update()
            effective_iter += 1

            # 日志记录
            current_lr = optimizer.param_groups[0]['lr']

            if effective_iter % getattr(cfg.logging, "loss_log", 100) == 0:
                loss_dict = {
                    "total_loss": total_loss,
                    "reproj_loss": reproj_loss,
                    "ssim_loss": ssim_loss,
                    "tv_loss": tv_loss,
                    "density_reg_loss": density_reg_loss,
                    "rendered_images": first_sample_rendered,
                    "gt_images": first_sample_gt,
                    "vol_pred": first_sample_vol_pred
                    # 移除mask相关代码
                    # "mask_images": first_sample_mask
                }
                log_metrics(loss_dict, effective_iter, current_lr)

            if effective_iter % getattr(cfg.logging, "render_log", 500) == 0:
                loss_dict = {
                    "total_loss": total_loss,
                    "reproj_loss": reproj_loss,
                    "ssim_loss": ssim_loss,
                    "tv_loss": tv_loss,
                    "density_reg_loss": density_reg_loss,
                    "rendered_images": first_sample_rendered,
                    "gt_images": first_sample_gt,
                    "vol_pred": first_sample_vol_pred
                    # 移除mask相关代码
                    # "mask_images": first_sample_mask
                }
                log_visualizations(loss_dict, effective_iter, cfg)
                
                # 在可视化后立即清理无需保留的变量
                del first_sample_rendered, first_sample_gt, first_sample_vol_pred
                # 移除mask相关代码
                # del first_sample_rendered, first_sample_gt, first_sample_vol_pred, first_sample_mask
                torch.cuda.empty_cache()

            # 记录参数直方图 - 配置记录频率
            if effective_iter % getattr(cfg.logging, "histogram_log", 10) == 0:
                log_parameter_histograms(first_batch_gaussian_splats, effective_iter)

            if effective_iter % getattr(cfg.logging, "ckpt_iterations", 1000) == 0 and effective_iter > 0:
                pbar_was_disabled = pbar.disable
                pbar.disable = True
                save_checkpoint(model, optimizer, scheduler, ema,
                                effective_iter, 0.0, log_dir, "model_latest.pth")
                pbar.disable = pbar_was_disabled
                torch.cuda.empty_cache()
                gc.collect()

            if effective_iter % mem_clean_interval == 0:
                # 更频繁清理内存
                torch.cuda.empty_cache()
                gc.collect()

        # 在每个批次后清除不需要的张量
        del input_images, source_cv2wT_quat, angles, view_to_world, gaussian_splats, first_batch_gaussian_splats
        
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "iter": effective_iter
        })
        pbar.update(1)

    pbar.close()

    # 如果在该 epoch 内最后一次累积未达到 accumulation_steps，也要手动更新
    if (len(train_loader) % accumulation_steps) != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update()
        effective_iter += 1

    # 最后彻底清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    return effective_iter

# ----------------------------
# 验证函数
# ----------------------------
def validate(model, optimizer, scheduler, ema, test_loader, cfg, device, iteration, log_dir):
    print("\n" + "="*50)
    logger.info(f"开始验证 - 迭代: {iteration}")
    print(f"开始验证 - 迭代: {iteration}")
    torch.cuda.empty_cache()
    
    eval_model = ema.ema_model if ema is not None else model
    eval_model.eval()
    
    with torch.no_grad():
        val_metric = evaluate_model(model=eval_model, dataloader=test_loader, cfg=cfg, device=device)
    
    log_dict = {f"验证_{k}": v for k, v in val_metric.items()}
    if is_wandb_enabled():
        try:
            wandb.log(log_dict, step=iteration)
        except Exception as e:
            logger.error(f"wandb记录验证结果失败: {e}")
    
    result_msg = f"验证结果 - 迭代 {iteration}: SSIM={val_metric.get('平均SSIM', 0.0):.4f}"
    logger.info(result_msg)
    print(result_msg)
    
    current_ssim = val_metric.get("平均SSIM", 0.0)
    save_checkpoint(model, optimizer, scheduler, ema, iteration, current_ssim, log_dir, "model_latest.pth")
    
    best_metric_path = os.path.join(log_dir, "best_metric.txt")
    best_metric = 0.0
    if os.path.exists(best_metric_path):
        with open(best_metric_path, 'r') as f:
            try:
                best_metric = float(f.read().strip())
            except:
                best_metric = 0.0
    
    if current_ssim > best_metric:
        best_msg = f"新最佳模型! 之前: {best_metric:.4f}, 当前: {current_ssim:.4f}"
        logger.info(best_msg)
        print(best_msg)
        save_checkpoint(model, optimizer, scheduler, ema, iteration, current_ssim, log_dir, "model_best.pth")
        with open(best_metric_path, 'w') as f:
            f.write(f"{current_ssim}")

    torch.cuda.empty_cache()
    gc.collect()
    
    print("验证完成")
    print("="*50 + "\n")
    
    return current_ssim

# ----------------------------
# 主函数
# ----------------------------
@hydra.main(version_base=None, config_path='../configs', config_name="default_config")
def main(cfg: DictConfig):
    start_time = datetime.datetime.now()
    logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置CUDA内存相关环境变量和配置
    # 注意：不同PyTorch版本支持的配置项可能不同
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 启用cudnn性能优化
    torch.backends.cudnn.benchmark = True
    
    fabric, device, log_dir = init_training(cfg)
    
    train_loader, test_loader = get_dataloaders(cfg, fabric)
    
    model, optimizer, scheduler, ema = init_model(cfg, fabric, device)
    
    first_iter, best_metric = load_checkpoint(model, optimizer, scheduler, ema, log_dir, device, cfg)
    
    iteration = first_iter + 1
    logger.info(f"从迭代 {iteration} 开始训练，目标迭代: {cfg.opt.iterations}")
    
    try:
        # 在训练开始前清理CUDA缓存并收集未引用对象
        torch.cuda.empty_cache()
        gc.collect()
        
        while iteration <= cfg.opt.iterations:
            # 训练前验证当前内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(f"GPU内存状态 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
            
            # 重要：使用torch.cuda.amp.autocast进行混合精度训练以减少内存使用
            if getattr(cfg.general, "mixed_precision", False):
                logger.info("使用混合精度训练...")
                
            iteration = train_one_epoch(fabric, train_loader, model, optimizer,
                                        scheduler, ema, cfg, device, iteration, log_dir)
            
            # 验证逻辑
            if iteration % cfg.logging.val_log == 0 or iteration > cfg.opt.iterations:
                # 验证前清理内存
                torch.cuda.empty_cache()
                gc.collect()
                best_metric = validate(model, optimizer, scheduler, ema,
                                       test_loader, cfg, device, iteration, log_dir)
    
    except KeyboardInterrupt:
        logger.info("检测到键盘中断，正在保存模型...")
        save_checkpoint(model, optimizer, scheduler, ema, iteration, best_metric, log_dir, "model_interrupt.pth")
    
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
    
    finally:
        # 确保在退出前释放所有资源
        del model, optimizer, scheduler
        if ema is not None:
            del ema
        
        end_time = datetime.datetime.now()
        training_duration = (end_time - start_time).total_seconds() / 3600.0
        logger.info(f"训练结束，总时长: {training_duration:.2f}小时")
        logger.info(f"完成迭代次数: {iteration-1}/{cfg.opt.iterations}")
        logger.info(f"最佳验证指标: {best_metric:.4f}")
        if wandb.run is not None:
            wandb.run.finish()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # 设置多进程启动方法为spawn，避免fork造成的CUDA上下文问题
    multiprocessing.set_start_method('spawn', force=True)
    main()
