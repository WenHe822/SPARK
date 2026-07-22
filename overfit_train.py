import os
import gc
import datetime
import torch
import hydra
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import sys
import logging
import argparse
import nibabel as nib  # 用于保存nii格式
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入init_training但进行自定义修改
from legacy.train_fabric import log_metrics, log_visualizations, setup_logging
from legacy.train_fabric import log_parameter_histograms, save_checkpoint, is_wandb_enabled
import torch.nn.functional as F


from datasets.dataset_readers_ct import readBlenderInfo
from datasets.dataset_ct import DatasetCT
from utils.camera_utils import cameraList_from_camInfos
from torch.utils.data import Dataset, DataLoader
from lightning.fabric import Fabric
from scene.gaussian_predictor_multichannel import GaussianSplatPredictor
import math
from torch.cuda.amp import autocast, GradScaler
# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
def setup_logging(log_dir, rank=0): # 添加 rank 参数以匹配原始 setup_logging
    if rank == 0: # 只在主进程设置日志
        os.makedirs(log_dir, exist_ok=True)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        console_handler.setLevel(logging.INFO) # 将控制台级别改为 INFO
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        logger.info(f"日志将保存到: {os.path.join(log_dir, 'training.log')}")
    else:
        # 对于非主进程，禁用日志记录或设置更高级别
        logger.setLevel(logging.CRITICAL)

def is_wandb_enabled():
    """
    辅助函数：判断 wandb 是否可用或是否被显式禁用
    """
    # 检查环境变量 WANDB_DISABLED
    wandb_disabled_env = os.getenv("WANDB_DISABLED", "false").lower() == "true"

    # 检查 wandb.run 是否存在并且 active
    wandb_run_active = wandb.run is not None and wandb.run.settings.mode != "disabled"

    # 如果环境变量设置为禁用，或者 wandb run 未激活，则认为 wandb 被禁用
    return not wandb_disabled_env and wandb_run_active


# ----------------------------
# 检查点相关函数
# ----------------------------
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

def load_checkpoint(model, optimizer, scheduler, ema, log_dir, device, cfg: DictConfig):
    first_iter = 0
    best_metric = 0.0
    ckpt_path = None
    # 优先加载预训练检查点
    if cfg.opt.pretrained_ckpt is not None:
        pretrained_path = cfg.opt.pretrained_ckpt
        if os.path.isfile(pretrained_path):
            ckpt_path = pretrained_path
            logger.info(f"发现指定的预训练检查点: {ckpt_path}")
        elif os.path.isdir(pretrained_path):
             # 如果是目录，尝试加载常见的检查点文件
            for ckpt_name in ["model_interrupt.pth", "model_best.pth", "model_latest.pth"]:
                candidate_path = os.path.join(pretrained_path, ckpt_name)
                if os.path.isfile(candidate_path):
                    ckpt_path = candidate_path
                    logger.info(f"在预训练目录中发现检查点: {ckpt_path}")
                    break
            if ckpt_path is None:
                 logger.warning(f"指定的预训练路径 {pretrained_path} 是一个目录，但未找到有效的检查点文件。")
        else:
             logger.warning(f"指定的预训练检查点路径无效: {pretrained_path}")


    # 如果没有指定预训练检查点或未找到，尝试加载中断的检查点
    if ckpt_path is None:
        interrupt_ckpt = os.path.join(log_dir, "overfit", "model_overfit_interrupt.pth") # 检查 overfit 子目录
        if os.path.isfile(interrupt_ckpt):
            ckpt_path = interrupt_ckpt
            logger.info(f"发现中断的检查点: {ckpt_path}")
        else:
            # 尝试加载最新的检查点
            latest_ckpt = os.path.join(log_dir, "overfit", "model_overfit_latest.pth") # 检查 overfit 子目录
            if os.path.isfile(latest_ckpt):
                ckpt_path = latest_ckpt
                logger.info(f"发现最新的检查点: {ckpt_path}")


    if ckpt_path is not None and os.path.isfile(ckpt_path):
        logger.info(f"加载检查点: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        try:
            # 尝试加载模型状态字典
            model_state_dict = checkpoint.get("model_state_dict")
            if model_state_dict:
                 # 处理可能的 fabric 包装
                if hasattr(model, '_forward_module'): # 如果模型被 fabric 包装
                    model._forward_module.load_state_dict(model_state_dict, strict=False)
                else:
                    model.load_state_dict(model_state_dict, strict=False)
                logger.info("模型权重已加载 (strict=False)")
            else:
                logger.warning("检查点中未找到 'model_state_dict'")


            # 加载优化器状态字典
            optimizer_state_dict = checkpoint.get("optimizer_state_dict")
            if optimizer_state_dict:
                try:
                    optimizer.load_state_dict(optimizer_state_dict)
                    logger.info("优化器状态已加载")
                except Exception as e:
                    logger.error(f"加载优化器状态失败: {e}. 可能需要调整学习率或重新初始化优化器。")
            else:
                 logger.warning("检查点中未找到 'optimizer_state_dict'")

            # 加载调度器状态字典 (仅当使用非固定学习率时)
            use_fixed_lr = getattr(cfg.opt, "use_fixed_lr", False)
            scheduler_state_dict = checkpoint.get("scheduler_state_dict")
            if scheduler_state_dict and scheduler is not None and not use_fixed_lr:
                try:
                    scheduler.load_state_dict(scheduler_state_dict)
                    logger.info("学习率调度器状态已加载")
                except Exception as e:
                    logger.error(f"加载调度器状态失败: {e}")
            elif use_fixed_lr:
                logger.info("使用固定学习率，跳过加载调度器状态。")
                # 确保优化器学习率被重置为固定值
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.opt.base_lr
                logger.info(f"已将优化器学习率强制重置为固定值: {cfg.opt.base_lr}")
            else:
                 logger.warning("检查点中未找到 'scheduler_state_dict' 或未使用调度器")


            # 加载EMA状态字典
            ema_state_dict = checkpoint.get("ema_state_dict")
            if ema is not None and ema_state_dict:
                try:
                    ema.load_state_dict(ema_state_dict)
                    logger.info("EMA状态已加载")
                except Exception as e:
                    logger.error(f"加载EMA状态失败: {e}")
            elif ema is not None:
                logger.warning("检查点中未找到 'ema_state_dict'")


            # 加载迭代次数和最佳指标
            first_iter = checkpoint.get("iteration", 0)
            best_metric = checkpoint.get("best_metric", 0.0)
            logger.info(f"从迭代 {first_iter} 继续训练，最佳指标: {best_metric:.4f}")


        except Exception as e:
            logger.error(f"加载检查点时发生错误: {e}", exc_info=True)
            logger.warning("将从头开始训练。")
            first_iter = 0
            best_metric = 0.0
    else:
        logger.info("未找到有效检查点，从头开始训练")

    return first_iter, best_metric


# ----------------------------
# 日志记录函数
# ----------------------------
def log_metrics(loss_dict, iteration, current_lr):
    log_data = {
        "总损失": loss_dict["total_loss"].item(),
        "投影损失": loss_dict["reproj_loss"].item(),
        "SSIM损失": loss_dict.get("ssim_loss", 0.0) if isinstance(loss_dict.get("ssim_loss"), (torch.Tensor, float)) else 0.0,
        "TV损失": loss_dict.get("tv_loss", 0.0) if isinstance(loss_dict.get("tv_loss"), (torch.Tensor, float)) else 0.0,
        "XYZ边界损失": loss_dict.get("xyz_boundary_loss", 0.0) if isinstance(loss_dict.get("xyz_boundary_loss"), (torch.Tensor, float)) else 0.0,
        "Mask正则损失": loss_dict.get("mask_reg_loss", 0.0) if isinstance(loss_dict.get("mask_reg_loss"), (torch.Tensor, float)) else 0.0,
        "学习率": current_lr,
    }

    # 转换为浮点数以避免 Tensor 类型问题
    log_data["SSIM损失"] = float(log_data["SSIM损失"]) if isinstance(log_data["SSIM损失"], torch.Tensor) else log_data["SSIM损失"]
    log_data["TV损失"] = float(log_data["TV损失"]) if isinstance(log_data["TV损失"], torch.Tensor) else log_data["TV损失"]
    log_data["XYZ边界损失"] = float(log_data["XYZ边界损失"]) if isinstance(log_data["XYZ边界损失"], torch.Tensor) else log_data["XYZ边界损失"]
    log_data["Mask正则损失"] = float(log_data["Mask正则损失"]) if isinstance(log_data["Mask正则损失"], torch.Tensor) else log_data["Mask正则损失"]

    # 添加这些行，将损失信息写入日志
    log_msg = (f"迭代 {iteration} - 总损失: {log_data['总损失']:.4f}, "
               f"投影损失: {log_data['投影损失']:.4f}, SSIM损失: {log_data['SSIM损失']:.4f}, "
               f"TV损失: {log_data['TV损失']:.4f}, XYZ边界损失: {log_data['XYZ边界损失']:.4f}, "
               f"Mask正则损失: {log_data['Mask正则损失']:.6f}, 学习率: {current_lr:.8f}")
    logger.info(log_msg)


    if is_wandb_enabled():
        try:
            wandb.log(log_data, step=iteration)
        except Exception as e:
            logger.error(f"wandb记录失败: {e}")


def log_parameter_histograms(gaussian_splats, iteration):
    """
    记录高斯点参数的分布直方图到wandb
    (移植自 legacy/train_fabric_single_gpu.py)
    """
    if not is_wandb_enabled():
        return

    histogram_dict = {}

    # 取第一个batch的数据作为样本
    sample_idx = 0  # 仅使用第一个样本进行统计
    try:
        # 密度直方图
        if "density" in gaussian_splats and gaussian_splats["density"] is not None:
            density_tensor = gaussian_splats["density"][sample_idx].contiguous()
            # 确保张量是连续的，并且被flatten为一维
            density_values = density_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["密度分布"] = wandb.Histogram(density_values)
            histogram_dict["密度最大值"] = float(density_values.max())
            histogram_dict["密度最小值"] = float(density_values.min())
            histogram_dict["密度平均值"] = float(density_values.mean())

        # 偏移直方图
        if "offset" in gaussian_splats and gaussian_splats["offset"] is not None:
            offset_tensor = gaussian_splats["offset"][sample_idx].contiguous()
            offset_values = offset_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["偏移分布"] = wandb.Histogram(offset_values)
            histogram_dict["偏移最大值"] = float(offset_values.max())
            histogram_dict["偏移最小值"] = float(offset_values.min())
            histogram_dict["偏移平均值"] = float(offset_values.mean())

        # 缩放直方图
        if "scaling" in gaussian_splats and gaussian_splats["scaling"] is not None:
            scaling_tensor = gaussian_splats["scaling"][sample_idx].contiguous()
            scaling_values = scaling_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["缩放分布"] = wandb.Histogram(scaling_values)
            histogram_dict["缩放最大值"] = float(scaling_values.max())
            histogram_dict["缩放最小值"] = float(scaling_values.min())
            histogram_dict["缩放平均值"] = float(scaling_values.mean())

        # 旋转直方图
        if "rotation" in gaussian_splats and gaussian_splats["rotation"] is not None:
            rotation_tensor = gaussian_splats["rotation"][sample_idx].contiguous()
            rotation_values = rotation_tensor.detach().cpu().reshape(-1).numpy()
            histogram_dict["旋转分布"] = wandb.Histogram(rotation_values)
            histogram_dict["旋转最大值"] = float(rotation_values.max())
            histogram_dict["旋转最小值"] = float(rotation_values.min())
            histogram_dict["旋转平均值"] = float(rotation_values.mean())

        # xyz坐标直方图
        if "xyz" in gaussian_splats and gaussian_splats["xyz"] is not None:
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
    """
    记录可视化结果到wandb
    (移植并适配自 legacy/train_fabric_single_gpu.py)
    """
    if not is_wandb_enabled():
        return

    log_dict = {}
    num_gaussians = getattr(cfg.model, "num_gaussians", "N/A") # 获取高斯数量

    if loss_dict.get("rendered_images") is not None and loss_dict.get("gt_images") is not None:
        rendered_images = loss_dict["rendered_images"]
        gt_images = loss_dict["gt_images"]

        # 只记录第一个样本的前几个视角
        num_views_to_log = min(rendered_images.shape[0], gt_images.shape[0], 5) # 最多记录5个

        for i in range(num_views_to_log):
            render_vis = rendered_images[i].squeeze().detach().cpu().numpy()
            gt_vis = gt_images[i].squeeze().detach().cpu().numpy()

            caption = f"Iter: {iteration}, k={num_gaussians}"

            # 处理灰度图显示
            if render_vis.ndim == 2 or (render_vis.ndim == 3 and render_vis.shape[0] == 1):
                 render_vis = render_vis.squeeze() # 移除通道维度
                 log_dict[f"视角{i}/渲染"] = wandb.Image(render_vis, caption=f"渲染 - {caption}", mode="L")
            elif render_vis.ndim == 3 and render_vis.shape[0] == 3: # C, H, W -> H, W, C
                 log_dict[f"视角{i}/渲染"] = wandb.Image(render_vis.transpose(1, 2, 0), caption=f"渲染 - {caption}")

            if gt_vis.ndim == 2 or (gt_vis.ndim == 3 and gt_vis.shape[0] == 1):
                 gt_vis = gt_vis.squeeze()
                 log_dict[f"视角{i}/真实"] = wandb.Image(gt_vis, caption=f"真实 - {caption}", mode="L")
            elif gt_vis.ndim == 3 and gt_vis.shape[0] == 3:
                 log_dict[f"视角{i}/真实"] = wandb.Image(gt_vis.transpose(1, 2, 0), caption=f"真实 - {caption}")


    # 体积切片可视化 (如果存在)
    if (loss_dict.get("vol_pred") is not None and
        isinstance(loss_dict["vol_pred"], torch.Tensor) and
        loss_dict["vol_pred"].dim() >= 3): # 至少是 3D 张量

        vol_pred = loss_dict["vol_pred"]
        # 确保 vol_pred 是 3D (D, H, W)
        if vol_pred.dim() > 3:
            vol_pred = vol_pred.squeeze() # 尝试移除 batch 或 channel 维度
        if vol_pred.dim() != 3:
             logger.warning(f"无法可视化体积切片，期望3D张量，得到形状: {loss_dict['vol_pred'].shape}")
             vol_pred = None # 重置以跳过可视化

        if vol_pred is not None:
            mid_z = vol_pred.shape[0] // 2
            mid_y = vol_pred.shape[1] // 2
            mid_x = vol_pred.shape[2] // 2

            axial_slice = vol_pred[mid_z, :, :].squeeze().detach().cpu().numpy()
            coronal_slice = vol_pred[:, mid_y, :].squeeze().detach().cpu().numpy()
            sagittal_slice = vol_pred[:, :, mid_x].squeeze().detach().cpu().numpy()

            caption = f"Iter: {iteration}, k={num_gaussians}"
            log_dict["切片/轴向"] = wandb.Image(axial_slice, caption=f"轴向切片 - {caption}")
            log_dict["切片/冠状"] = wandb.Image(coronal_slice, caption=f"冠状切片 - {caption}")
            log_dict["切片/矢状"] = wandb.Image(sagittal_slice, caption=f"矢状切片 - {caption}")


    if log_dict:
        try:
            wandb.log(log_dict, step=iteration)
        except Exception as e:
            logger.error(f"wandb可视化记录失败: {e}")
# 自定义的初始化训练函数，支持二级目录结构和自定义输出目录
def init_training_overfit(cfg: DictConfig, experiment_name: str, output_dir: str = None):
    """自定义初始化训练函数，支持二级目录结构和自定义输出目录
    
    Args:
        cfg: 配置对象
        experiment_name: 实验名称，用于创建二级目录
        output_dir: 自定义输出目录路径，默认为当前工作目录
        
    Returns:
        fabric, device, log_dir: 与原始init_training相同的返回值
    """
    torch.set_float32_matmul_precision('high')
    precision = 'bf16-mixed'  # 使用32位浮点精度
    fabric = Fabric(accelerator="cuda", devices=1, strategy="auto", precision=precision)
    fabric.launch()
    
    # 导入safe_state函数
    from legacy.train_fabric import safe_state
    device = safe_state(cfg)
    
    # 使用自定义输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    
    # 使用二级目录结构
    run_name = getattr(cfg.logging, "wandb_run_name", "overfit_experiment")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{timestamp}"  # 添加时间戳避免覆盖
    
    # 更新配置中的运行名称
    cfg.logging.wandb_run_name = run_name
    
    base_dir = os.path.join(output_dir, "experiments", experiment_name)
    os.makedirs(base_dir, exist_ok=True)
    log_dir = os.path.join(base_dir, "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(log_dir,0)
    
    # 设置 wandb 目录
    os.environ["WANDB_DIR"] = os.path.join(output_dir, "wandb")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    
    # 处理wandb配置，与原始函数相同
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
        # 设置超时和重试参数
        wandb.init(
            project=project_name,
            name=run_name,
            resume="allow",
            reinit=True,
            config=wandb_config,
            settings=wandb.Settings(
                console="off",
                timeout=2,  # 连接超时10秒
                retry=dict(
                    retries=1,  # 仅重试1次
                    backoff=2,  # 退避系数
                    status_forcelist=(500, 502, 503, 504)  # 仅在这些状态码时重试
                )
            )
        )
        WANDB_ENABLED = True
        logger.info(f"WandB项目: {project_name}, 运行名称: {run_name}")
    except Exception as e:
        logger.error(f"WandB初始化失败: {e}")
        logger.info("将继续训练，但不会记录到WandB")
        WANDB_ENABLED = False
        os.environ["WANDB_DISABLED"] = "true"
    
    return fabric, device, log_dir

def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (3 x 3)
    Returns:
        q: quaternion of shape (4)
    """
    tr = 1 + M[ 0, 0] + M[ 1, 1] + M[ 2, 2]

    if tr > 0:
        r = torch.sqrt(tr) / 2.0
        x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
        y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
        z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
    elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
        S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
        r = (M[ 2, 1] - M[ 1, 2]) / S
        x = 0.25 * S
        y = (M[ 0, 1] + M[ 1, 0]) / S 
        z = (M[ 0, 2] + M[ 2, 0]) / S 
    elif M[ 1, 1] > M[ 2, 2]: 
        S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
        r = (M[ 0, 2] - M[ 2, 0]) / S
        x = (M[ 0, 1] + M[ 1, 0]) / S
        y = 0.25 * S
        z = (M[ 1, 2] + M[ 2, 1]) / S
    else:
        S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
        r = (M[ 1, 0] - M[ 0, 1]) / S
        x = (M[ 0, 2] + M[ 2, 0]) / S
        y = (M[ 1, 2] + M[ 2, 1]) / S
        z = 0.25 * S

    return torch.stack([r, x, y, z], dim=-1)
class SingleSampleDataset(Dataset):
    """
    单样本数据集：只包含指定proj_dir中的所有投影
    用于过拟合训练，固定一个视角作为输入，其他视角作为目标
    """
    def __init__(self, proj_dir, device=None):
        """
        Args:
            proj_dir (str): 投影目录路径
            device (torch.device, optional): 如果提供，则提前缓存 vol_mask 到这个设备
        """
        self.proj_dir = proj_dir
        self.device = device
        # 用于缓存 GPU 版的 vol_mask
        self._vol_mask_gpu = None
        
        # 读取场景信息
        self.scene_info = readBlenderInfo(proj_dir)
        self.cameras = cameraList_from_camInfos(self.scene_info.cameras)
        self.vol_gt = self.scene_info.vol
        self.scanner_cfg = self.scene_info.scanner_cfg
        self.scene_scale = self.scene_info.scene_scale
        
        # 读取体积掩码
        self.vol_mask = self.scene_info.vol_mask
        if self.vol_mask is not None and self.device is not None:
            if not isinstance(self.vol_mask, torch.Tensor):
                self.vol_mask = torch.tensor(self.vol_mask, dtype=torch.float32)
            else:
                self.vol_mask = self.vol_mask.float()
            # 缓存到 GPU
            self._vol_mask_gpu = self.vol_mask.to(self.device, non_blocking=True)
        else:
            logger.warning("未找到体积掩码，将使用全1掩码")
            # 创建与体积相同大小的全1掩码
            if hasattr(self.scene_info, 'vol') and self.scene_info.vol is not None:
                self.vol_mask = torch.ones_like(self.scene_info.vol, dtype=torch.float32)
            else:
                # 使用scanner_cfg中的体素分辨率创建掩码
                nVoxel = self.scanner_cfg.get("nVoxel", [256, 256, 256])
                self.vol_mask = torch.ones(nVoxel, dtype=torch.float32)
        
        # 确保至少有一个相机
        assert len(self.cameras) > 0, f"在 {proj_dir} 中找不到任何相机/投影"
        logger.info(f"加载了 {len(self.cameras)} 个投影用于过拟合训练")
        
        # 计算边界框
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"]) - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"]) + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )
        
        # 计算相机位姿四元数表示
        world_view_transforms = []
        view_to_world_transforms = []
        for cam in self.cameras:
            world_view_transforms.append(cam.world_view_transform)
            view_to_world_transforms.append(cam.view_world_transform)
        
        self.world_view_transforms = torch.stack(world_view_transforms)
        self.view_to_world_transforms = torch.stack(view_to_world_transforms)
        
        # 计算四元数
        self.source_cv2wT_quat = self._get_source_cw2wT(self.view_to_world_transforms)
        
        # 固定第一个视角作为输入视角
        self.input_camera_idx = 81
        logger.info(f"固定使用第 {self.input_camera_idx} 个视角作为输入视角")
        
    def _get_source_cw2wT(self, source_cameras_view_to_world):
        """从视图到世界变换矩阵计算四元数表示"""
        qs = []
        for c_idx in range(source_cameras_view_to_world.shape[0]):
            qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
        return torch.stack(qs, dim=0)
        
    def __len__(self):
        # 始终返回固定数量，便于多次迭代
        return 100
    
    def __getitem__(self, idx):
        # 使用固定的输入视角
        input_image_idx = self.input_camera_idx

        # 返回已经缓存到 GPU 的 vol_mask（若 device 未指定，仍然返回 CPU 版）
        vol_mask = self._vol_mask_gpu if self._vol_mask_gpu is not None else self.vol_mask
        return {
            "cameras": self.cameras,
            "scanner_cfg": self.scanner_cfg,
            "vol": self.vol_gt,
            "scene_scale": self.scene_scale,
            "bbox": self.bbox,
            "source_cv2wT_quat": self.source_cv2wT_quat,
            "input_image_idx": input_image_idx,
            "vol_mask": vol_mask
        }

def overfit_collate_fn(batch, input_images_count):
    """
    为过拟合训练设计的collate函数
    使用固定的输入视角，随机选择其他视角作为目标
    """
    batch_input_images = []
    batch_camera_angles = []
    batch_view_to_world = []
    scanner_cfg_list = []
    bbox_list = []
    batch_source_cv2wT_quat = []
    cameras_list = []
    vol_mask_list = []  # 新增：收集体积掩码

    for sample in batch:
        cameras = sample["cameras"]
        cameras_list.append(cameras)
        
        # 使用固定的输入视角
        input_image_idx = sample["input_image_idx"]
        
        # 获取所有可用的目标视角索引（排除输入视角）
        target_indices = list(range(len(cameras)))
        target_indices.remove(input_image_idx)
        
        # 随机选择目标视角
        if len(target_indices) > input_images_count - 1:
            target_indices = torch.tensor(target_indices)[torch.randperm(len(target_indices))[:input_images_count - 1]].tolist()
        
        # 组合输入视角和目标视角
        selected_indices = [input_image_idx] + target_indices
        
        sample_images = []
        sample_angles = []
        sample_v2w = []
        for idx in selected_indices:
            cam = cameras[idx]
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
        
        # 添加体积掩码到列表
        if "vol_mask" in sample and sample["vol_mask"] is not None:
            vol_mask_list.append(sample["vol_mask"])

    batch_input_images = torch.stack(batch_input_images, dim=0)
    batch_camera_angles = torch.stack(batch_camera_angles, dim=0)
    batch_view_to_world = torch.stack(batch_view_to_world, dim=0)
    batch_bbox = torch.stack(bbox_list, dim=0)
    batch_source_cv2wT_quat = torch.stack(batch_source_cv2wT_quat, dim=0)
    
    # 处理体积掩码
    if vol_mask_list:
        batch_vol_mask = torch.stack(vol_mask_list, dim=0)
    else:
        # 如果没有掩码，创建默认的全1掩码
        batch_vol_mask = None
        logger.warning("批次中没有有效的体积掩码")
    
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
        "cameras": cameras_list,
        "vol_mask":batch_vol_mask

    }

def save_voxelized_volume(gaussian_splats, bbox, nVoxel, save_path, device):
    """
    将高斯参数体素化并保存为nii.gz格式
    
    参数:
        gaussian_splats: 高斯参数字典
        bbox: 边界框 (2, 3)
        nVoxel: 体素分辨率 [x, y, z]
        save_path: 保存路径
        device: 计算设备
    """
    logger.info(f"体素化并保存体积到: {save_path}")
    
    # 确保所有张量在同一设备上
    processed_splats = {}
    for k, v in gaussian_splats.items():
        # 检查并移除batch维度 (B, N, D) -> (N, D) 
        if v.dim() > 2 and v.shape[0] == 1:
            processed_splats[k] = v.squeeze(0).to(device)
        else:
            processed_splats[k] = v.to(device)
    
    # 计算体积中心
    center = (bbox[0] + bbox[1]) / 2
    
    # 计算体积大小
    sVoxel = bbox[1] - bbox[0]
    
    # 使用query函数进行体素化
    with torch.no_grad():
        vol_output = query(
            gaussian_splats=processed_splats,
            center=center,
            nVoxel=torch.tensor(nVoxel, device=device),
            sVoxel=sVoxel,
            scaling_modifier=1.0
        )
    
    vol_pred = vol_output["vol"]
    
    # 转换为numpy数组
    vol_np = vol_pred.detach().cpu().numpy()
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建NIfTI对象并保存
    affine = np.eye(4)  # 默认仿射变换矩阵
    nii_img = nib.Nifti1Image(vol_np, affine)
    nib.save(nii_img, save_path)
    
    logger.info(f"体积已保存为: {save_path}，形状: {vol_np.shape}")
    
    return vol_pred
def mask_xyz_regularization_loss(gaussian_splats, vol_mask, lambda_mask=1.0, sample_num=10000):
    """
    随机抽样部分高斯点，并用最近邻采样取得 mask 值，显著加速。
    Args:
        gaussian_splats['xyz']: Tensor (N,3)，(-1,1)³ 内
        vol_mask:           Tensor (D,H,W)，已在 GPU 上
        sample_num:         最多采样点数
    Returns:
        loss: 标量 Tensor
    """
    xyz = gaussian_splats['xyz']          # (N,3)
    N = xyz.shape[0]
    # 随机子采样
    if N > sample_num:
        idx = torch.randperm(N, device=xyz.device)[:sample_num]
        xyz = xyz[idx]                     # (M,3)，M ≤ sample_num

    # 最近邻采样：将 (-1,1) 映射到体素坐标 [0, D-1]/[0,H-1]/[0,W-1]
    D, H, W = vol_mask.shape
    # 先归一化到 [0, D-1]
    coords = (xyz + 1.0) * 0.5
    coords = coords * torch.tensor([D-1, H-1, W-1], device=xyz.device)
    idxs = coords.round().long()          # (M,3)
    # clamp 防止越界
    idxs[:,0].clamp_(0, D-1)
    idxs[:,1].clamp_(0, H-1)
    idxs[:,2].clamp_(0, W-1)

    # 取值，最近邻
    mask_vals = vol_mask[idxs[:,0], idxs[:,1], idxs[:,2]]  # (M,)

    # loss = mean(1 - mask)
    return lambda_mask * (1.0 - mask_vals).mean()

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

def count_points_in_boundary(xyz, min_coord=-1.0, max_coord=1.0):
    """
    统计xyz坐标位于指定边界内的点的数量和比例
    
    Args:
        xyz (torch.Tensor): 形状为(N, 3)或(B, N, 3)的坐标张量
        min_coord (float): 边界最小值
        max_coord (float): 边界最大值
        
    Returns:
        dict: 包含点数统计的字典
    """
    # 确保输入是二维张量 (N, 3)
    original_shape = xyz.shape
    if len(original_shape) > 2:
        xyz = xyz.reshape(-1, original_shape[-1])
    
    total_points = xyz.shape[0]
    
    # 检查每个维度是否在范围内
    in_x_range = (xyz[:, 0] >= min_coord) & (xyz[:, 0] <= max_coord)
    in_y_range = (xyz[:, 1] >= min_coord) & (xyz[:, 1] <= max_coord)
    in_z_range = (xyz[:, 2] >= min_coord) & (xyz[:, 2] <= max_coord)
    
    # 统计每个维度的点数和比例
    points_in_x_range = torch.sum(in_x_range).item()
    points_in_y_range = torch.sum(in_y_range).item()
    points_in_z_range = torch.sum(in_z_range).item()
    
    # 同时在所有维度范围内的点
    in_all_range = in_x_range & in_y_range & in_z_range
    points_in_range = torch.sum(in_all_range).item()
    
    # 计算各维度的最小值、最大值和平均值
    x_min, x_max = xyz[:, 0].min().item(), xyz[:, 0].max().item()
    y_min, y_max = xyz[:, 1].min().item(), xyz[:, 1].max().item()
    z_min, z_max = xyz[:, 2].min().item(), xyz[:, 2].max().item()
    
    x_mean, y_mean, z_mean = xyz[:, 0].mean().item(), xyz[:, 1].mean().item(), xyz[:, 2].mean().item()
    
    return {
        "total_points": total_points,
        "points_in_range": points_in_range,
        "percentage": 100.0 * points_in_range / total_points,
        "x_stats": {
            "in_range": points_in_x_range,
            "percentage": 100.0 * points_in_x_range / total_points,
            "min": x_min,
            "max": x_max,
            "mean": x_mean
        },
        "y_stats": {
            "in_range": points_in_y_range,
            "percentage": 100.0 * points_in_y_range / total_points,
            "min": y_min,
            "max": y_max,
            "mean": y_mean
        },
        "z_stats": {
            "in_range": points_in_z_range,
            "percentage": 100.0 * points_in_z_range / total_points,
            "min": z_min,
            "max": z_max,
            "mean": z_mean
        }
    }

def local_log_visualizations(loss_dict, iteration, cfg, log_dir):
    """本地保存可视化结果，不依赖WandB"""
    from torchvision.utils import save_image
    import matplotlib.pyplot as plt
    
    # 获取每个像素的高斯数量
    num_gaussians = getattr(cfg.model, "num_gaussians", 4)
    
    # 创建保存目录
    vis_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 保存渲染图像和真实图像的对比
    if "rendered_images" in loss_dict and "gt_images" in loss_dict:
        rendered = loss_dict["rendered_images"]
        gt = loss_dict["gt_images"]
        
        if rendered is not None and gt is not None:
            # 确保渲染图像和GT图像具有相同数量
            min_imgs = min(rendered.size(0), gt.size(0))
            
            # 创建网格图像
            grid_size = min(min_imgs, 5)  # 最多显示5个图像
            
            # 保存每对图像的对比
            for i in range(grid_size):
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # 在左侧显示渲染图像
                render_img = rendered[i].detach().cpu().permute(1, 2, 0).numpy()
                if render_img.shape[2] == 1:  # 灰度图像
                    render_img = render_img.squeeze(2)
                    axes[0].imshow(render_img, cmap='gray')
                else:  # RGB图像
                    axes[0].imshow(render_img)
                axes[0].set_title(f"渲染图像 (k={num_gaussians})")
                axes[0].axis('off')
                
                # 在右侧显示GT图像
                gt_img = gt[i].detach().cpu().permute(1, 2, 0).numpy()
                if gt_img.shape[2] == 1:  # 灰度图像
                    gt_img = gt_img.squeeze(2)
                    axes[1].imshow(gt_img, cmap='gray')
                else:  # RGB图像
                    axes[1].imshow(gt_img)
                axes[1].set_title("真实图像")
                axes[1].axis('off')
                
                # 保存图像
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"compare_iter_{iteration}_view_{i}_k{num_gaussians}.png"))
                plt.close(fig)
            
            logger.info(f"已保存渲染对比图像到: {vis_dir}")
    
    # 保存体积数据（如果存在）
    if "vol_pred" in loss_dict and loss_dict["vol_pred"] is not None:
        vol_dir = os.path.join(log_dir, "volumes")
        os.makedirs(vol_dir, exist_ok=True)
        
        vol_pred = loss_dict["vol_pred"].detach().cpu().numpy()
        
        # 保存中心切片
        if len(vol_pred.shape) == 3:
            # 获取中心切片
            slice_z = vol_pred[vol_pred.shape[0]//2, :, :]
            slice_y = vol_pred[:, vol_pred.shape[1]//2, :]
            slice_x = vol_pred[:, :, vol_pred.shape[2]//2]
            
            # 保存切片
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(slice_z, cmap='gray')
            plt.title('Z轴中心切片')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(slice_y, cmap='gray')
            plt.title('Y轴中心切片')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(slice_x, cmap='gray')
            plt.title('X轴中心切片')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vol_dir, f"volume_slices_iter_{iteration}_k{num_gaussians}.png"))
            plt.close()
            
            logger.info(f"已保存体积切片图像到: {vol_dir}")

def train_one_epoch_overfit(fabric, train_loader, model, optimizer, scheduler, ema, cfg, device, iteration, log_dir):
    """适用于过拟合训练的单轮训练函数"""



    model.train()
    model.current_iter = iteration

    # 关闭梯度累积（过拟合训练通常不需要）
    accumulation_steps = 1
    mem_clean_interval = getattr(cfg.opt, "mem_clean_interval", 1000)
    
    # 从配置中获取每个样本要使用的图像数量
    imgs_per_obj = getattr(cfg.opt, "imgs_per_obj", 4)
    
    # 获取xyz边界正则化参数
    lambda_xyz_boundary = getattr(cfg.opt, "lambda_xyz_boundary", 100)
    min_coord = getattr(cfg.opt, "min_coord", -1.0)
    max_coord = getattr(cfg.opt, "max_coord", 1.0)

    lambda_mask = getattr(cfg.opt, "lambda_mask", 0.01)
    
    # 检查是否使用固定学习率
    use_fixed_lr = getattr(cfg.opt, "use_fixed_lr", True)
    
    # 获取每个像素的高斯数量
    num_gaussians = getattr(cfg.model, "num_gaussians", 6)
    logger.info(f"训练使用每个像素 {num_gaussians} 个高斯球")
    
    # 创建保存体素化结果的目录
    voxel_dir = os.path.join(log_dir, "overfit", "volumes")
    os.makedirs(voxel_dir, exist_ok=True)
    
    # 仅当执行 optimizer.step() 时，才增加 effective_iter
    effective_iter = iteration
    pbar = tqdm(total=len(train_loader), desc="过拟合训练", dynamic_ncols=True, leave=False)

    # 训练前清理内存
    torch.cuda.empty_cache()
    gc.collect()

    # 用于保存最后一个批次的数据，以便在训练结束时体素化
    last_batch_data = None

    for batch_idx, data in enumerate(train_loader):
        if effective_iter > cfg.opt.iterations:
            break
  
        # 保存最后一个批次的数据
        last_batch_data = data
        
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

        gaussian_splats = model(input_images, source_cv2wT_quat, camera_params_list, scanner_cfg_list)#gaussian_splats['xyz].shape:[1, 262144, 3]
        
        # 存储当前batch的gaussian_splats用于直方图统计
        first_batch_gaussian_splats = gaussian_splats.copy()

        # 准备损失项累加
        batch_reproj_loss = 0.0
        batch_ssim_loss = 0.0
        batch_tv_loss = 0.0
        batch_xyz_boundary_loss = 0.0
        batch_mask_reg_loss = 0.0

        first_sample_rendered = None
        first_sample_gt = None
        first_sample_vol_pred = None


        w_reproj = getattr(cfg.opt, "w_l12", 1.0)
        w_ssim   = getattr(cfg.opt, "w_ssim", 0.0)
        w_tv     = getattr(cfg.opt, "w_tv",   0.0)

        # 遍历 batch 内每个 sample
        for sample_idx in range(batch_size):
            gaussian_splat_sample = {k: v[sample_idx].contiguous()
                                     for k, v in gaussian_splats.items()}
            cameras = data["cameras"][sample_idx]
            scanner_cfg = data["scanner_cfg"][sample_idx]
            bbox = data["bbox"][sample_idx]
            
            # 计算xyz边界正则化损失
            if lambda_xyz_boundary > 0 and 'xyz' in gaussian_splat_sample:
                xyz_reg_loss = xyz_boundary_regularization(
                    gaussian_splat_sample['xyz'], 
                    min_coord=min_coord,
                    max_coord=max_coord,
                    lambda_xyz_boundary=lambda_xyz_boundary
                )
                batch_xyz_boundary_loss += xyz_reg_loss
            # 计算掩码正则化损失
            if lambda_mask > 0 and effective_iter%20==0:
                mask_xyz_loss = mask_xyz_regularization_loss(
                    gaussian_splat_sample,
                    data["vol_mask"].squeeze(),  # 保证 (D,H,W)
                    lambda_mask,
                    sample_num=10000
                )
                batch_mask_reg_loss += mask_xyz_loss
                if sample_idx == 0:
                    logger.info(f"迭代 {effective_iter}：统一 mask_xyz 正则化损失 = {mask_xyz_loss.item():.6f}")


            total_cameras = len(cameras)
            if total_cameras > imgs_per_obj:
                # 随机选择索引，确保每次都选不同的投影
                indices = torch.randperm(total_cameras)[:imgs_per_obj]
                selected_cameras = [cameras[i] for i in indices]
            else:
                # 如果投影数量不足，使用全部投影
                selected_cameras = cameras

            # 渲染选定的相机视角并计算重投影损失
            rendered_images = []
            gt_images = []
            
            # 在渲染前添加坐标范围统计（只对第一个相机进行统计）
            if 'xyz' in gaussian_splat_sample:
                stats = count_points_in_boundary(gaussian_splat_sample['xyz'])
                
                # 每10次迭代或第一次迭代时打印详细统计信息
                if effective_iter % 10 == 0 or effective_iter == 0:
                    logger.info(f"迭代 {effective_iter}，样本 {sample_idx} 高斯点坐标统计:")
                    logger.info(f"总点数: {stats['total_points']}, 在[-1,1]³内: {stats['points_in_range']} ({stats['percentage']:.2f}%)")
                    logger.info(f"X轴: 在[-1,1]内: {stats['x_stats']['in_range']} ({stats['x_stats']['percentage']:.2f}%), "
                              f"范围: [{stats['x_stats']['min']:.3f}, {stats['x_stats']['max']:.3f}], 均值: {stats['x_stats']['mean']:.3f}")
                    logger.info(f"Y轴: 在[-1,1]内: {stats['y_stats']['in_range']} ({stats['y_stats']['percentage']:.2f}%), "
                              f"范围: [{stats['y_stats']['min']:.3f}, {stats['y_stats']['max']:.3f}], 均值: {stats['y_stats']['mean']:.3f}")
                    logger.info(f"Z轴: 在[-1,1]内: {stats['z_stats']['in_range']} ({stats['z_stats']['percentage']:.2f}%), "
                              f"范围: [{stats['z_stats']['min']:.3f}, {stats['z_stats']['max']:.3f}], 均值: {stats['z_stats']['mean']:.3f}")
                else:
                    # 其他迭代次数只打印简短统计
                    logger.info(f"迭代 {effective_iter}：点数 {stats['total_points']}，在[-1,1]³内: {stats['percentage']:.2f}%")
            
            for cam_idx, cam in enumerate(selected_cameras):
                render_out = render(cam, gaussian_splat_sample)
                rendered_images.append(render_out["render"].unsqueeze(0))
                gt_images.append(cam.original_image.unsqueeze(0))

            rendered_images = torch.cat(rendered_images, dim=0).to(device)
            gt_images = torch.cat(gt_images, dim=0).to(device)

            # 移除前景/背景加权，直接计算损失
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
                from legacy.train_fabric import ssim
                sample_ssim_val = (1.0 - ssim(rendered_images, gt_images))

            # 计算 TV
            sample_tv_val = 0.0
            if w_tv > 0.0:
                from legacy.train_fabric import tv_3d_loss, query
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
                # 仅保存第一个视角的渲染结果用于可视化
                first_sample_rendered = rendered_images[:5]  # 最多保存5个视角
                first_sample_gt = gt_images[:5]
                
                # 每1000次迭代保存一次体素化结果
                if effective_iter % 1000 == 0 or effective_iter == 0:
                    # 使用scanner_cfg中的分辨率进行体素化
                    nVoxel = scanner_cfg.get("nVoxel", [64, 64, 64])
                    # 在文件名中包含高斯数量信息
                    save_path = os.path.join(voxel_dir, f"volume_iter_{effective_iter}_k{num_gaussians}.nii.gz")
                    save_voxelized_volume(gaussian_splat_sample, bbox, nVoxel, save_path, device)
                
            # 每个样本处理后及时清理中间变量
            del rendered_images, gt_images
            if sample_tv_val > 0.0:
                del vol_pred

        # 平均每个 sample 的损失
        reproj_loss = batch_reproj_loss / batch_size
        ssim_loss = batch_ssim_loss / batch_size if w_ssim > 0 else 0.0
        tv_loss = batch_tv_loss / batch_size if w_tv > 0 else 0.0
        xyz_boundary_loss = batch_xyz_boundary_loss / batch_size if lambda_xyz_boundary > 0 else 0.0
        mask_reg_loss = batch_mask_reg_loss / batch_size if lambda_mask > 0 else 0.0
        # 组装总损失
        total_loss = (w_reproj * reproj_loss +
                      w_ssim   * ssim_loss +
                      w_tv     * tv_loss +
                      xyz_boundary_loss + 
                      mask_reg_loss)   

        # 检查 NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"迭代 {effective_iter} 产生 NaN 或 Inf 损失，跳过该 batch。")
            continue

        # 检查梯度是否包含NaN
        has_nan = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan = True
                break

        if not has_nan:
            optimizer.zero_grad()
            fabric.backward(total_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            logger.warning(f"迭代 {effective_iter} 检测到NaN梯度，跳过参数更新")
        
        # 根据配置决定是否调用scheduler.step()
        if not use_fixed_lr and scheduler is not None:
            scheduler.step()
            if effective_iter % 10 == 0:  # 每10次迭代记录一次学习率变化
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"学习率已更新，当前学习率: {current_lr:.6f}")
        else:
            if effective_iter % 100 == 0:  # 每100次迭代确认一次固定学习率
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"使用固定学习率，当前学习率: {current_lr:.6f}")
        
        if ema is not None:
            ema.update()
        
        effective_iter += 1

        # 日志记录
        current_lr = optimizer.param_groups[0]['lr']

        if effective_iter % cfg.logging.loss_log == 0:
            loss_dict = {
                "total_loss": total_loss,
                "reproj_loss": reproj_loss,
                "ssim_loss": ssim_loss,
                "tv_loss": tv_loss,
                "xyz_boundary_loss": xyz_boundary_loss,
                "mask_reg_loss": mask_reg_loss,
                "rendered_images": first_sample_rendered,
                "gt_images": first_sample_gt,
                "vol_pred": first_sample_vol_pred
                # 移除 mask_images
            }
            log_metrics(loss_dict, effective_iter, current_lr)

        if effective_iter % cfg.logging.render_log == 0:
            loss_dict = {
                "total_loss": total_loss,
                "reproj_loss": reproj_loss,
                "ssim_loss": ssim_loss,
                "tv_loss": tv_loss,
                "xyz_boundary_loss": xyz_boundary_loss,
                "rendered_images": first_sample_rendered,
                "gt_images": first_sample_gt,
                "vol_pred": first_sample_vol_pred
            }
            # 先尝试使用wandb记录可视化
            log_visualizations(loss_dict, effective_iter, cfg)
            
            # 同时保存到本地
            local_log_visualizations(loss_dict, effective_iter, cfg, log_dir)
            
        # 记录参数直方图
        if effective_iter % cfg.logging.histogram_log == 0:
            log_parameter_histograms(first_batch_gaussian_splats, effective_iter)

        if effective_iter % cfg.logging.ckpt_iterations == 0 and effective_iter > 0:
            # 确保保存到overfit目录
            overfit_dir = os.path.join(log_dir, "overfit")
            os.makedirs(overfit_dir, exist_ok=True)
            save_checkpoint(model, optimizer, scheduler, ema,
                            effective_iter, 0.0, overfit_dir, "model_overfit_latest.pth")
            torch.cuda.empty_cache()
            gc.collect()

        # if effective_iter % mem_clean_interval == 0:
        #     # 更频繁清理内存
        #     torch.cuda.empty_cache()
        #     gc.collect()
            
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "lr": f"{current_lr:.2e}",
            "iter": effective_iter
        })
        pbar.update(1)

    pbar.close()
    
    # 如果训练已完成，保存最后一次体素化结果
    if effective_iter >= cfg.opt.iterations and last_batch_data is not None:
        # 获取最新模型输出
        model.eval()
        with torch.no_grad():
            input_images = last_batch_data["input_images"].to(device)
            source_cv2wT_quat = last_batch_data["source_cv2wT_quat"].to(device)
            
            angles = last_batch_data["camera_params"]["angle"].to(device)
            view_to_world = last_batch_data["camera_params"]["view_to_world"].to(device)
            camera_params_list = []
            for i in range(batch_size):
                for j in range(cfg.data.input_images):
                    cp = {"angle": float(angles[i,j].item()),
                          "view_to_world": view_to_world[i,j]}
                    camera_params_list.append(cp)
            scanner_cfg_list = []
            for i in range(batch_size):
                for j in range(cfg.data.input_images):
                    scanner_cfg_list.append(last_batch_data["scanner_cfg"][i])
                    
            final_gaussian_splats = model(input_images, source_cv2wT_quat, camera_params_list, scanner_cfg_list)
            
            # 对第一个样本进行体素化
            sample_idx = 0
            gaussian_splat_sample = {k: v[sample_idx].contiguous() for k, v in final_gaussian_splats.items()}
            bbox = last_batch_data["bbox"][sample_idx]
            scanner_cfg = last_batch_data["scanner_cfg"][sample_idx]
            
            # 使用scanner_cfg中的分辨率进行体素化
            nVoxel = scanner_cfg.get("nVoxel", [64, 64, 64])
            # 在文件名中包含高斯数量信息
            num_gaussians = getattr(cfg.model, "num_gaussians", 1)
            save_path = os.path.join(voxel_dir, f"volume_final_k{num_gaussians}.nii.gz")
            save_voxelized_volume(gaussian_splat_sample, bbox, nVoxel, save_path, device)
            
            # 保存一个高分辨率版本
            high_res_nVoxel = [128, 128, 128]
            save_path_high_res = os.path.join(voxel_dir, f"volume_final_high_res_k{num_gaussians}.nii.gz")
            save_voxelized_volume(gaussian_splat_sample, bbox, high_res_nVoxel, save_path_high_res, device)
    
    # 最后彻底清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    return effective_iter
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

# 为过拟合训练定义固定学习率版本的init_model
def init_model_fixed_lr(cfg: DictConfig, fabric: Fabric, device):
    """创建使用固定学习率的模型和优化器
    
    这个函数基于原始的init_model，但使用常数学习率调度器而不是余弦衰减
    """
    # 使用原始函数创建模型和优化器
    model, optimizer, _, ema = init_model(cfg, fabric, device)
    
    # 覆盖优化器中的学习率为固定值
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.opt.base_lr
    
    # 创建一个空的调度器，step()方法不会改变学习率
    class ConstantLRScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self._last_lr = [cfg.opt.base_lr] * len(optimizer.param_groups)
            
        def step(self):
            return
            
        def state_dict(self):
            return {'last_lr': self._last_lr}
            
        def load_state_dict(self, state_dict):
            self._last_lr = state_dict.get('last_lr', self._last_lr)
    
    scheduler = ConstantLRScheduler(optimizer)
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"为过拟合训练创建固定学习率调度器: 基础学习率 {cfg.opt.base_lr}，当前学习率 {current_lr}")
    
    return model, optimizer, scheduler, ema

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='过拟合训练模型到单个CT投影')
    parser.add_argument('--proj_dir', type=str, default=r'/Disk_10TB/zhouhaowei/data/train/LIDC-IDRI-0001.20000101.3000566.1_cone', help='投影目录路径')
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml', help='基础配置文件路径')
    parser.add_argument('--input_images', type=int, default=1, help='使用多少张图像作为输入')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--iterations', type=int, default=100000000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=0.00001, help='学习率')
    parser.add_argument('--imgs_per_obj', type=int, default=4, help='每次选择多少个投影计算损失')
    parser.add_argument('--ckpt_interval', type=int, default=10000, help='保存检查点间隔')
    parser.add_argument('--loss_log_interval', type=int, default=10, help='损失记录间隔')
    parser.add_argument('--render_log_interval', type=int, default=10, help='渲染结果记录间隔')
    parser.add_argument('--voxel_interval', type=int, default=1000, help='体素化保存间隔')
    parser.add_argument('--use_fixed_lr', action='store_true', help='是否使用固定学习率（无衰减）')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='预训练检查点路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录路径，默认为当前工作目录')
    parser.add_argument('--num_gaussians', type=int, default=4, help='每个像素的Gaussian数量')
    args = parser.parse_args()

    # 确保投影目录存在
    proj_dir = args.proj_dir
    if not os.path.exists(proj_dir):
        raise ValueError(f"指定的过拟合数据路径不存在: {proj_dir}")
    
    # 从配置文件加载基础配置
    with open(args.config_path, 'r') as f:
        cfg_dict = OmegaConf.load(f)
    
    # 修改配置以适应过拟合训练
    cfg = OmegaConf.create(cfg_dict)
    
    # 添加cam_embd配置，避免ConfigAttributeError
    if not hasattr(cfg, 'cam_embd'):
        cfg.cam_embd = OmegaConf.create({
            "embedding": "pose",
            "encode_embedding": None,
            "dimension": 32,
            "method": "film"
        })
    
    # 设置数据相关配置
    if not hasattr(cfg, 'data'):
        cfg.data = OmegaConf.create({})
    cfg.data.input_images = args.input_images
    cfg.data.batch_size = args.batch_size
    
    # 设置模型相关配置
    if not hasattr(cfg, 'model'):
        cfg.model = OmegaConf.create({})
    # 设置每个像素的高斯数量
    cfg.model.num_gaussians = args.num_gaussians
    logger.info(f"每个像素使用 {args.num_gaussians} 个高斯")
    
    # 设置优化器相关配置
    if not hasattr(cfg, 'opt'):
        cfg.opt = OmegaConf.create({})
    cfg.opt.iterations = args.iterations
    cfg.opt.base_lr = args.lr
    cfg.opt.imgs_per_obj = args.imgs_per_obj
    # 设置是否使用固定学习率
    cfg.opt.use_fixed_lr = args.use_fixed_lr
    
    # 设置日志相关配置
    if not hasattr(cfg, 'logging'):
        cfg.logging = OmegaConf.create({})
    cfg.logging.ckpt_iterations = args.ckpt_interval
    cfg.logging.loss_log = args.loss_log_interval
    cfg.logging.render_log = args.render_log_interval
    cfg.logging.histogram_log = 50
    cfg.logging.wandb_run_name = f"overfit_{os.path.basename(proj_dir)}"
    
    # 将命令行参数添加到配置中
    if args.pretrained_ckpt is not None:
        cfg.opt.pretrained_ckpt = args.pretrained_ckpt
    
    logger.info(f"使用投影目录进行过拟合训练: {proj_dir}")
    
    # 使用自定义初始化训练函数，创建二级目录结构
    experiment_name = os.path.basename(proj_dir)
    fabric, device, log_dir = init_training_overfit(cfg, experiment_name, args.output_dir)
    
    # 创建overfit目录
    overfit_dir = os.path.join(log_dir, "overfit")
    os.makedirs(overfit_dir, exist_ok=True)
    
    # 创建单样本数据集，并提前缓存 vol_mask 到 GPU
    dataset = SingleSampleDataset(proj_dir, device=device)
    
    # 创建数据加载器
    import functools
    collate_fn_with_cfg = functools.partial(overfit_collate_fn, input_images_count=cfg.data.input_images)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,  # 过拟合训练不需要多进程
        collate_fn=collate_fn_with_cfg
    )
    dataloader = fabric.setup_dataloaders(dataloader)
    
    # 初始化模型和优化器 - 基于命令行参数选择是否使用固定学习率
    if args.use_fixed_lr:
        logger.info("使用固定学习率（无学习率衰减）进行过拟合训练")
        model, optimizer, scheduler, ema = init_model_fixed_lr(cfg, fabric, device)
    else:
        model, optimizer, scheduler, ema = init_model(cfg, fabric, device)
    
    # 加载检查点（如果存在）
    first_iter = 0
    best_metric = 0.0
    if cfg.opt.pretrained_ckpt is not None:
        pretrained_path = cfg.opt.pretrained_ckpt
        if os.path.isfile(pretrained_path):
            logger.info(f"加载检查点: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # 当使用固定学习率时，不加载scheduler状态
                if "scheduler_state_dict" in checkpoint and scheduler is not None and not args.use_fixed_lr:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logger.info("已加载学习率调度器状态")
                elif args.use_fixed_lr:
                    logger.info("使用固定学习率，跳过加载学习率调度器状态")
                    # 重要：强制重置优化器中的学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = cfg.opt.base_lr
                    logger.info(f"已重置优化器学习率为固定值: {cfg.opt.base_lr}")
                if ema is not None and "ema_state_dict" in checkpoint:
                    ema.load_state_dict(checkpoint["ema_state_dict"])
                first_iter = checkpoint["iteration"]
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")
    
    # 开始过拟合训练
    logger.info(f"开始过拟合训练，起始迭代: {first_iter}")
    
    try:
        iteration = first_iter
        while iteration <= cfg.opt.iterations:
            iteration = train_one_epoch_overfit(
                fabric, dataloader, model, optimizer, scheduler, ema, 
                cfg, device, iteration, log_dir
            )
            
            # 过拟合训练通常不需要验证，但可以保存检查点
            if iteration % cfg.logging.ckpt_iterations == 0:
                # 确保保存到overfit目录
                save_checkpoint(
                    model, optimizer, scheduler, ema,
                    iteration, 0.0, overfit_dir, f"model_overfit_{iteration}.pth"
                )
                
    except KeyboardInterrupt:
        logger.info("检测到键盘中断，正在保存模型...")
        # 确保保存到overfit目录
        save_checkpoint(
            model, optimizer, scheduler, ema, 
            iteration, 0.0, overfit_dir, "model_overfit_interrupt.pth"
        )
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        
    finally:
        # 保存最终模型到overfit目录
        # save_checkpoint(
        #     model, optimizer, scheduler, ema,
        #     iteration, 0.0, overfit_dir, "model_overfit_final.pth"
        # )
        
        # 清理资源
        del model, optimizer, scheduler
        if ema is not None:
            del ema
        if wandb.run is not None:
            wandb.run.finish()
        torch.cuda.empty_cache()
        gc.collect()
        
        # logger.info(f"过拟合训练完成，总迭代次数: {iteration}，结果保存在: {overfit_dir}")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 设置可见GPU为3号
if __name__ == "__main__":
    # 确保正确导入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from r2_gaussian.gaussian import render, query
    main()
