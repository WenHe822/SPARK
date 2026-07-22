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
import math
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# - 移除从 train_network 导入的函数

from datasets.dataset_readers_ct import readBlenderInfo
from datasets.dataset_ct import DatasetCT
from utils.camera_utils import cameraList_from_camInfos
# + 导入基础 loss 和 general utils
from utils.general_utils import safe_state
from utils.loss_utils import ssim, tv_3d_loss # 确保这些在 utils 中
from torch.utils.data import Dataset, DataLoader
from lightning.fabric import Fabric
from scene.gaussian_predictor_multichannel import GaussianSplatPredictor
from r2_gaussian.gaussian import render, query # 确保这个导入是正确的


# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# +++ 从 legacy/train_fabric_single_gpu.py 移植过来的函数 +++

# ----------------------------
# 日志设置
# ----------------------------
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


# --- 其他辅助函数保持不变 ---
# def init_training_overfit(...)
# def matrix_to_quaternion(...)
# class SingleSampleDataset(...)
# def overfit_collate_fn(...)
# def save_voxelized_volume(...)
# def mask_xyz_regularization_loss(...)
# def xyz_boundary_regularization(...)
# def count_points_in_boundary(...)
# def local_log_visualizations(...)
# def train_one_epoch_overfit(...)
# def init_model(...)
# def init_model_fixed_lr(...)
# def main(...)

# +++ 结束移植的函数 +++

# ----------------------------
# 自定义初始化训练函数
# ----------------------------
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
    precision = 'bf16-mixed'  # 使用 bfloat16 混合精度
    fabric = Fabric(accelerator="cuda", devices=1, strategy="auto", precision=precision)
    fabric.launch()

    # 使用移植过来的 safe_state
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

    # 设置日志 (使用移植过来的 setup_logging)
    # 传递 fabric.global_rank 确保只有主进程写日志
    setup_logging(log_dir, fabric.global_rank)

    # 设置 wandb 目录
    wandb_dir = os.path.join(output_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    # 处理wandb配置
    disable_wandb = getattr(cfg.logging, "disable_wandb", False) or os.getenv("DISABLE_WANDB", "false").lower() == "true"

    if disable_wandb:
        logger.info("WandB已禁用，仅使用本地日志")
        os.environ["WANDB_DISABLED"] = "true"
    elif fabric.global_rank == 0: # 只在主进程初始化 wandb
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
                dir=wandb_dir, # 指定 wandb 目录
                settings=wandb.Settings(
                    console="off",
                     _disable_stats=True, # 减少系统监控开销
                     _disable_meta=True,
                    start_method="thread",
                    # timeout=10, # 连接超时10秒
                    # retry=dict(
                    #     retries=1,  # 仅重试1次
                    #     backoff=2,  # 退避系数
                    #     status_forcelist=(500, 502, 503, 504) # 仅在这些状态码时重试
                    # )
                )
            )
            logger.info(f"WandB项目: {project_name}, 运行名称: {run_name}")
        except Exception as e:
            logger.error(f"WandB初始化失败: {e}")
            logger.info("将继续训练，但不会记录到WandB")
            os.environ["WANDB_DISABLED"] = "true" # 标记为禁用
    else:
         # 非主进程也标记为禁用，避免意外调用
         os.environ["WANDB_DISABLED"] = "true"


    return fabric, device, log_dir

# ... (剩余代码保持不变) ...

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

    # 增加一个 epsilon 防止 sqrt(0) 或除以零
    epsilon = 1e-8

    if tr > epsilon:
        r = torch.sqrt(tr) / 2.0
        x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
        y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
        z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
    elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
        S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2] + epsilon) * 2 # S=4*qx
        r = (M[ 2, 1] - M[ 1, 2]) / S
        x = 0.25 * S
        y = (M[ 0, 1] + M[ 1, 0]) / S
        z = (M[ 0, 2] + M[ 2, 0]) / S
    elif M[ 1, 1] > M[ 2, 2]:
        S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2] + epsilon) * 2 # S=4*qy
        r = (M[ 0, 2] - M[ 2, 0]) / S
        x = (M[ 0, 1] + M[ 1, 0]) / S
        y = 0.25 * S
        z = (M[ 1, 2] + M[ 2, 1]) / S
    else:
        S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1] + epsilon) * 2 # S=4*qz
        r = (M[ 1, 0] - M[ 0, 1]) / S
        x = (M[ 0, 2] + M[ 2, 0]) / S
        y = (M[ 1, 2] + M[ 2, 1]) / S
        z = 0.25 * S

    # 返回 w, x, y, z 顺序
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
        if self.vol_mask is not None:
            if not isinstance(self.vol_mask, torch.Tensor):
                self.vol_mask = torch.tensor(self.vol_mask, dtype=torch.float32)
            else:
                self.vol_mask = self.vol_mask.float()
            # 如果提供了设备，缓存到 GPU
            if self.device is not None:
                try:
                    self._vol_mask_gpu = self.vol_mask.to(self.device, non_blocking=True)
                    logger.info(f"体积掩码已缓存到设备: {self.device}")
                except Exception as e:
                    logger.error(f"无法将体积掩码缓存到 GPU: {e}. 将在 CPU 上处理。")
                    self._vol_mask_gpu = None # 回退到 CPU
                    self.vol_mask = self.vol_mask.cpu() # 确保在 CPU 上
        else:
            logger.warning("未找到体积掩码，将使用全1掩码")
            # 创建与体积相同大小的全1掩码
            if hasattr(self.scene_info, 'vol') and self.scene_info.vol is not None:
                 # 确保vol是Tensor
                 if not isinstance(self.scene_info.vol, torch.Tensor):
                      vol_tensor = torch.tensor(self.scene_info.vol)
                 else:
                      vol_tensor = self.scene_info.vol
                 self.vol_mask = torch.ones_like(vol_tensor, dtype=torch.float32)
            else:
                # 使用scanner_cfg中的体素分辨率创建掩码
                nVoxel = self.scanner_cfg.get("nVoxel", [256, 256, 256])
                self.vol_mask = torch.ones(nVoxel, dtype=torch.float32)
            # 如果需要，也尝试缓存全1掩码
            if self.device is not None:
                 try:
                    self._vol_mask_gpu = self.vol_mask.to(self.device, non_blocking=True)
                    logger.info(f"全1体积掩码已缓存到设备: {self.device}")
                 except Exception as e:
                    logger.error(f"无法将全1体积掩码缓存到 GPU: {e}. 将在 CPU 上处理。")
                    self._vol_mask_gpu = None
                    self.vol_mask = self.vol_mask.cpu()


        # 确保至少有一个相机
        assert len(self.cameras) > 0, f"在 {proj_dir} 中找不到任何相机/投影"
        logger.info(f"加载了 {len(self.cameras)} 个投影用于过拟合训练")

        # 计算边界框
        offOrigin = torch.tensor(self.scanner_cfg["offOrigin"], dtype=torch.float32)
        sVoxel = torch.tensor(self.scanner_cfg["sVoxel"], dtype=torch.float32)
        self.bbox = torch.stack(
            [
                offOrigin - sVoxel / 2,
                offOrigin + sVoxel / 2,
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
             # 提取旋转矩阵 R (view-to-world 的左上 3x3)
             # 注意：view-to-world 的 R 是从相机坐标系到世界坐标系的旋转
             # 我们需要世界到相机的旋转 R^T 来计算四元数
             # 或者直接用 view-to-world 的 R 计算，得到 q_v2w
             # 通常相机姿态用 q_w2c (世界到相机)
             # R_w2c = R_v2w^T
             # R_v2w = source_cameras_view_to_world[c_idx, :3, :3]
             # q = matrix_to_quaternion(R_v2w) # 这得到的是 q_v2w
             # 如果需要 q_w2c, 应该用 R_w2c = R_v2w.T
             # R_w2c = source_cameras_view_to_world[c_idx, :3, :3].T
             # qs.append(matrix_to_quaternion(R_w2c))

             # 当前代码使用 view_to_world 直接计算，得到 q_v2w
             qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3]))

        return torch.stack(qs, dim=0)

    def __len__(self):
        # 始终返回固定数量，便于多次迭代
        return 100 # 可以设为 1，因为每个 epoch 都是一样的数据

    def __getitem__(self, idx):
        # 使用固定的输入视角
        input_image_idx = self.input_camera_idx

        # 返回已经缓存到 GPU 的 vol_mask（若 device 未指定 或 缓存失败，仍然返回 CPU 版）
        vol_mask_to_return = self._vol_mask_gpu if self._vol_mask_gpu is not None else self.vol_mask

        return {
            "cameras": self.cameras,
            "scanner_cfg": self.scanner_cfg,
            "vol": self.vol_gt, # 注意：vol_gt 可能很大，每次都加载可能影响性能
            "scene_scale": self.scene_scale,
            "bbox": self.bbox,
            "source_cv2wT_quat": self.source_cv2wT_quat,
            "input_image_idx": input_image_idx,
            "vol_mask": vol_mask_to_return
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
    vol_mask_list = []  # 收集体积掩码

    # 因为是过拟合，batch 中所有样本都是一样的
    sample = batch[0]
    num_samples_in_batch = len(batch) # 通常是 1，但处理以防万一

    for _ in range(num_samples_in_batch):
        cameras = sample["cameras"]
        cameras_list.append(cameras) # 每个 batch item 都有完整的相机列表

        # 使用固定的输入视角
        input_image_idx = sample["input_image_idx"]

        # 获取所有可用的目标视角索引（排除输入视角）
        all_indices = list(range(len(cameras)))
        target_indices = [i for i in all_indices if i != input_image_idx]

        # 随机选择目标视角
        num_targets_to_select = input_images_count - 1
        if len(target_indices) > num_targets_to_select:
            # 随机打乱并选择前 num_targets_to_select 个
            selected_target_indices = torch.tensor(target_indices)[torch.randperm(len(target_indices))[:num_targets_to_select]].tolist()
        else:
            # 如果目标视角不够，就全选
            selected_target_indices = target_indices

        # 组合输入视角和目标视角
        selected_indices = [input_image_idx] + selected_target_indices

        sample_images = []
        sample_angles = []
        sample_v2w = []
        sample_quat = [] # 同时收集对应选择相机的四元数
        for idx in selected_indices:
            cam = cameras[idx]
            sample_images.append(cam.original_image)
            sample_angles.append(cam.angle)
            sample_v2w.append(cam.view_world_transform)
            sample_quat.append(sample["source_cv2wT_quat"][idx]) # 获取对应的四元数

        sample_images = torch.stack(sample_images, dim=0)
        batch_input_images.append(sample_images)
        batch_camera_angles.append(torch.tensor(sample_angles, dtype=torch.float32))
        batch_view_to_world.append(torch.stack(sample_v2w, dim=0))
        batch_source_cv2wT_quat.append(torch.stack(sample_quat, dim=0)) # 使用选择的四元数

        # scanner_cfg 和 bbox 对于所有样本是相同的
        scanner_cfg_list.append(sample["scanner_cfg"])
        bbox_list.append(sample["bbox"])

        # 添加体积掩码到列表
        if "vol_mask" in sample and sample["vol_mask"] is not None:
            vol_mask_list.append(sample["vol_mask"])


    batch_input_images = torch.stack(batch_input_images, dim=0)
    batch_camera_angles = torch.stack(batch_camera_angles, dim=0)
    batch_view_to_world = torch.stack(batch_view_to_world, dim=0)
    batch_bbox = torch.stack(bbox_list, dim=0)
    batch_source_cv2wT_quat = torch.stack(batch_source_cv2wT_quat, dim=0)

    # 处理体积掩码
    batch_vol_mask = None
    if vol_mask_list:
         # 检查所有掩码是否在同一设备上
         devices = {mask.device for mask in vol_mask_list}
         if len(devices) > 1:
             logger.warning(f"批次中的体积掩码位于不同设备: {devices}. 将尝试移动到第一个掩码的设备。")
             target_device = vol_mask_list[0].device
             vol_mask_list = [mask.to(target_device) for mask in vol_mask_list]

         try:
             batch_vol_mask = torch.stack(vol_mask_list, dim=0)
         except Exception as e:
             logger.error(f"堆叠体积掩码失败: {e}. 将返回 None。")
             batch_vol_mask = None # 出错则返回 None
    else:
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
        "source_cv2wT_quat": batch_source_cv2wT_quat, # 返回选择的四元数
        "cameras": cameras_list, # 返回完整的相机列表
        "vol_mask":batch_vol_mask
    }


# ... (save_voxelized_volume, mask_xyz_regularization_loss, xyz_boundary_regularization, count_points_in_boundary, local_log_visualizations)


# ----------------------------
# 正则化函数 (移植或确认存在)
# ----------------------------
def xyz_boundary_regularization(xyz, min_coord=-1.0, max_coord=1.0, lambda_xyz_boundary=1.0):
    """
    惩罚超出 [min_coord, max_coord] 边界的 xyz 坐标。
    (与 legacy/train_fabric_single_gpu.py 中的版本一致)
    """
    # 计算超出下边界的量 (对于 x < min_coord, 计算 min_coord - x)
    lower_penalty = torch.relu(min_coord - xyz)
    # 计算超出上边界的量 (对于 x > max_coord, 计算 x - max_coord)
    upper_penalty = torch.relu(xyz - max_coord)

    # 对所有点和所有维度计算平均惩罚
    boundary_violation = (lower_penalty + upper_penalty).mean()

    return lambda_xyz_boundary * boundary_violation

def density_regularization(density, threshold=0.1, lambda_density=5):
    """
    密度正则: 对低于 threshold 的密度施加惩罚
    (移植自 legacy/train_fabric_single_gpu.py)
    """
    penalty = torch.relu(threshold - density)
    loss_density_reg = lambda_density * torch.mean(penalty)
    return loss_density_reg


def save_voxelized_volume(gaussian_splats, bbox, nVoxel, save_path, device, scaling_modifier=1.0):
    """
    将高斯参数体素化并保存为nii.gz格式

    参数:
        gaussian_splats: 高斯参数字典 (应为单个样本的参数, 如 (N, D))
        bbox: 边界框 (2, 3), tensor on device
        nVoxel: 体素分辨率 [x, y, z], list or tuple
        save_path: 保存路径 (str)
        device: 计算设备
        scaling_modifier: 缩放因子
    """
    logger.info(f"体素化并保存体积到: {save_path}")

    # 确保所有张量在同一设备上
    processed_splats = {}
    for k, v in gaussian_splats.items():
        # 确保输入是 (N, D) 形状
        if v.dim() > 2:
            if v.shape[0] == 1: # 如果有 batch 维度且为 1，则移除
                 processed_splats[k] = v.squeeze(0).to(device)
            else:
                 logger.error(f"save_voxelized_volume 期望单个样本的 splats，但键 '{k}' 的形状为 {v.shape}")
                 return None # 无法处理多样本
        elif v.dim() == 2: # 正确的形状 (N, D)
            processed_splats[k] = v.to(device)
        else:
             logger.error(f"save_voxelized_volume 键 '{k}' 的形状无效: {v.shape}")
             return None

    # 确保 bbox 在正确设备上
    bbox = bbox.to(device)

    # 计算体积中心和大小
    center = (bbox[0] + bbox[1]) / 2.0 # (3,)
    sVoxel = bbox[1] - bbox[0]       # (3,)
    nVoxel_tensor = torch.tensor(nVoxel, device=device, dtype=torch.long) # 确保是 LongTensor

    # 使用 query 函数进行体素化
    with torch.no_grad():
        try:
            vol_output = query(
                gaussian_splats=processed_splats,
                center=center,
                nVoxel=nVoxel_tensor,
                sVoxel=sVoxel,
                scaling_modifier=scaling_modifier
            )
            # query 返回的是字典，vol 在 'vol' 键中
            vol_pred = vol_output.get("vol")
            if vol_pred is None:
                 logger.error("体素化查询未返回 'vol' 键。")
                 return None
        except Exception as e:
             logger.error(f"执行体素化查询时出错: {e}", exc_info=True)
             return None # 出错则不继续

    # 转换为numpy数组 (确保先移动到 CPU)
    vol_np = vol_pred.detach().cpu().numpy()

    # 创建目录（如果不存在）
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except Exception as e:
        logger.error(f"创建目录 {os.path.dirname(save_path)} 失败: {e}")
        return None

    # 创建NIfTI对象并保存
    try:
        affine = np.eye(4)  # 默认仿射变换矩阵
        # 考虑体素大小和原点来设置更准确的仿射矩阵
        voxel_sizes = (sVoxel / nVoxel_tensor).cpu().numpy()
        origin_corner = bbox[0].cpu().numpy() # 通常用体素中心或角点

        affine[0, 0] = voxel_sizes[0]
        affine[1, 1] = voxel_sizes[1]
        affine[2, 2] = voxel_sizes[2]
        affine[:3, 3] = origin_corner # 设置原点

        # NIfTI 数据通常是 (x, y, z) 顺序，但 torch 是 (d, h, w) -> (z, y, x)
        # 需要调整轴序或确认 nibabel 如何处理
        # 假设 vol_np 是 (z, y, x)，需要转置为 (x, y, z)
        # vol_np_nifti = np.transpose(vol_np, (2, 1, 0)) # z,y,x -> x,y,z

        # 直接保存，让使用者根据需要调整
        vol_np_nifti = vol_np

        nii_img = nib.Nifti1Image(vol_np_nifti, affine)
        nib.save(nii_img, save_path)
        logger.info(f"体积已保存为: {save_path}，形状: {vol_np_nifti.shape}")
    except Exception as e:
        logger.error(f"保存 NIfTI 文件失败: {e}", exc_info=True)
        return None

    return vol_pred # 返回 GPU 上的预测体积


def mask_xyz_regularization_loss(gaussian_splats, vol_mask, lambda_mask=1.0, sample_num=10000):
    """
    随机抽样部分高斯点，并用最近邻采样取得 mask 值，显著加速。
    Args:
        gaussian_splats['xyz']: Tensor (N,3)，应在 (-1,1)³ 内
        vol_mask:           Tensor (D,H,W)，已在 GPU 上
        lambda_mask:        损失权重
        sample_num:         最多采样点数
    Returns:
        loss: 标量 Tensor
    """
    if 'xyz' not in gaussian_splats or gaussian_splats['xyz'] is None:
         logger.warning("mask_xyz_regularization_loss: 未找到 'xyz' 数据。")
         return torch.tensor(0.0, device=vol_mask.device) # 返回零损失

    xyz = gaussian_splats['xyz']          # (N,3)
    if xyz.numel() == 0:
        return torch.tensor(0.0, device=vol_mask.device) # 如果没有点，损失为0

    N = xyz.shape[0]
    target_device = vol_mask.device # 确保所有计算在掩码所在设备进行
    xyz = xyz.to(target_device)

    # 随机子采样
    if N > sample_num:
        idx = torch.randperm(N, device=target_device)[:sample_num]
        xyz_sampled = xyz[idx]                     # (M,3)，M ≤ sample_num
    else:
        xyz_sampled = xyz                          # 使用所有点

    # 检查 vol_mask 是否有效
    if vol_mask is None or vol_mask.numel() == 0:
        logger.warning("mask_xyz_regularization_loss: vol_mask 无效。")
        return torch.tensor(0.0, device=target_device)

    # 最近邻采样：将 (-1,1) 映射到体素坐标 [0, D-1]/[0,H-1]/[0,W-1]
    D, H, W = vol_mask.shape
    shape_tensor = torch.tensor([D-1, H-1, W-1], device=target_device, dtype=torch.float32)

    # 归一化到 [0, 1]
    coords_normalized = (xyz_sampled + 1.0) * 0.5
    # 缩放到体素索引范围 [0, Shape-1]
    coords_voxel = coords_normalized * shape_tensor
    # 四舍五入到最近的整数索引
    idxs = coords_voxel.round().long()          # (M,3)

    # clamp 防止越界
    idxs[:,0].clamp_(0, D-1)
    idxs[:,1].clamp_(0, H-1)
    idxs[:,2].clamp_(0, W-1)

    # 使用整数索引从 vol_mask 中取值 (最近邻)
    try:
        # vol_mask 需要是 long 类型才能用于索引？ 不，索引是long，mask是float
        # 确保 vol_mask 是 float 类型
        if vol_mask.dtype != torch.float32:
             vol_mask = vol_mask.float()
        mask_vals = vol_mask[idxs[:,0], idxs[:,1], idxs[:,2]]  # (M,)
    except IndexError as e:
        logger.error(f"mask_xyz_regularization_loss: 索引越界错误 - {e}")
        logger.error(f"Mask shape: {vol_mask.shape}, Idx min: {idxs.min(0).values}, Idx max: {idxs.max(0).values}")
        return torch.tensor(0.0, device=target_device) # 返回零损失
    except Exception as e:
        logger.error(f"mask_xyz_regularization_loss: 获取掩码值时出错 - {e}")
        return torch.tensor(0.0, device=target_device) # 返回零损失


    # loss = mean(relu(1 - mask)) # 只惩罚 mask < 1 的区域
    # loss = (1.0 - mask_vals).mean() # 原始实现：惩罚所有 mask != 1 的区域

    # 修改为只惩罚预测在 mask 为 0 的区域的点： loss = mean(1 - mask_value) if mask_value < 0.5 else 0
    # 或者更简单：loss = mean(relu(-mask_vals)) ??? 不对
    # 应该是惩罚 mask=0 区域内的点。
    # 我们希望 xyz 落在 mask=1 的区域。如果落在 mask=0 的区域，则产生损失。
    # mask_vals 是对应 xyz 位置的 mask 值。如果 mask_vals=0，损失应为 1；如果 mask_vals=1，损失应为 0。
    # 所以损失是 1 - mask_vals
    loss = (1.0 - mask_vals).mean()

    return lambda_mask * loss

# ... (count_points_in_boundary, local_log_visualizations)

def count_points_in_boundary(xyz, min_coord=-1.0, max_coord=1.0):
    """
    统计xyz坐标位于指定边界内的点的数量和比例
    (版本一致)
    """
    # 确保输入是二维张量 (N, 3)
    original_shape = xyz.shape
    if len(original_shape) > 2:
        if original_shape[0] == 1: # 处理可能的 batch 维度
            xyz = xyz.squeeze(0)
        else:
            xyz = xyz.reshape(-1, original_shape[-1])

    if xyz.numel() == 0: # 处理空张量
        stats = {
            "total_points": 0, "points_in_range": 0, "percentage": 0.0,
            "x_stats": {"in_range": 0, "percentage": 0.0, "min": float('nan'), "max": float('nan'), "mean": float('nan')},
            "y_stats": {"in_range": 0, "percentage": 0.0, "min": float('nan'), "max": float('nan'), "mean": float('nan')},
            "z_stats": {"in_range": 0, "percentage": 0.0, "min": float('nan'), "max": float('nan'), "mean": float('nan')}
        }
        return stats

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
    # 使用 .item() 转换为 Python float
    x_min, x_max = xyz[:, 0].min().item(), xyz[:, 0].max().item()
    y_min, y_max = xyz[:, 1].min().item(), xyz[:, 1].max().item()
    z_min, z_max = xyz[:, 2].min().item(), xyz[:, 2].max().item()

    x_mean, y_mean, z_mean = xyz[:, 0].mean().item(), xyz[:, 1].mean().item(), xyz[:, 2].mean().item()

    safe_division = lambda num, den: (num / den) if den > 0 else 0.0

    return {
        "total_points": total_points,
        "points_in_range": points_in_range,
        "percentage": 100.0 * safe_division(points_in_range, total_points),
        "x_stats": {
            "in_range": points_in_x_range,
            "percentage": 100.0 * safe_division(points_in_x_range, total_points),
            "min": x_min,
            "max": x_max,
            "mean": x_mean
        },
        "y_stats": {
            "in_range": points_in_y_range,
            "percentage": 100.0 * safe_division(points_in_y_range, total_points),
            "min": y_min,
            "max": y_max,
            "mean": y_mean
        },
        "z_stats": {
            "in_range": points_in_z_range,
            "percentage": 100.0 * safe_division(points_in_z_range, total_points),
            "min": z_min,
            "max": z_max,
            "mean": z_mean
        }
    }


def local_log_visualizations(loss_dict, iteration, cfg, log_dir):
    """本地保存可视化结果，不依赖WandB"""
    try:
        import matplotlib.pyplot as plt
        from torchvision.utils import save_image # 可以用来保存张量网格
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        logger.warning("Matplotlib 未安装，无法保存本地可视化图像。")
        MATPLOTLIB_AVAILABLE = False
        return # 如果没有 matplotlib，直接返回

    # 获取每个像素的高斯数量
    num_gaussians = getattr(cfg.model, "num_gaussians", "N/A")

    # 创建保存目录
    vis_dir = os.path.join(log_dir, "visualizations", f"iter_{iteration}")
    os.makedirs(vis_dir, exist_ok=True)

    # 保存渲染图像和真实图像的对比
    if "rendered_images" in loss_dict and "gt_images" in loss_dict:
        rendered = loss_dict["rendered_images"]
        gt = loss_dict["gt_images"]

        if rendered is not None and gt is not None and rendered.shape[0] > 0 and gt.shape[0] > 0:
            # 确保渲染图像和GT图像具有相同数量
            num_views = min(rendered.size(0), gt.size(0))

            for i in range(num_views): # 保存每个视角
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Iter: {iteration}, View: {i}, k={num_gaussians}", fontsize=12)

                # --- 渲染图像 ---
                render_img = rendered[i].detach().cpu() # (C, H, W)
                # 处理单通道和三通道
                if render_img.shape[0] == 1: # 灰度图像 (1, H, W)
                    render_img_display = render_img.squeeze(0).numpy() # (H, W)
                    cmap = 'gray'
                elif render_img.shape[0] == 3: # RGB图像 (3, H, W)
                    render_img_display = render_img.permute(1, 2, 0).numpy() # (H, W, 3)
                    cmap = None
                else:
                     logger.warning(f"不支持的渲染图像通道数: {render_img.shape[0]}")
                     continue # 跳过这个视角

                axes[0].imshow(render_img_display, cmap=cmap)
                axes[0].set_title("渲染图像")
                axes[0].axis('off')

                # --- 真实图像 ---
                gt_img = gt[i].detach().cpu() # (C, H, W)
                if gt_img.shape[0] == 1: # 灰度图像 (1, H, W)
                    gt_img_display = gt_img.squeeze(0).numpy() # (H, W)
                    cmap = 'gray'
                elif gt_img.shape[0] == 3: # RGB图像 (3, H, W)
                    gt_img_display = gt_img.permute(1, 2, 0).numpy() # (H, W, 3)
                    cmap = None
                else:
                     logger.warning(f"不支持的真实图像通道数: {gt_img.shape[0]}")
                     continue # 跳过这个视角

                axes[1].imshow(gt_img_display, cmap=cmap)
                axes[1].set_title("真实图像")
                axes[1].axis('off')

                # 保存图像
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应标题
                save_path = os.path.join(vis_dir, f"compare_view_{i}.png")
                try:
                    plt.savefig(save_path)
                except Exception as e:
                    logger.error(f"保存图像 {save_path} 失败: {e}")
                plt.close(fig) # 关闭图形，释放内存

            logger.info(f"已保存渲染对比图像到: {vis_dir}")

    # 保存体积数据切片（如果存在且为3D）
    if "vol_pred" in loss_dict and loss_dict["vol_pred"] is not None:
        vol_pred_tensor = loss_dict["vol_pred"]
        # 确保是 3D 张量 (D, H, W)
        if vol_pred_tensor.dim() > 3:
            vol_pred_tensor = vol_pred_tensor.squeeze() # 尝试移除 batch/channel
        if vol_pred_tensor.dim() == 3:
             vol_pred = vol_pred_tensor.detach().cpu().numpy()

             # 保存中心切片
             slice_z = vol_pred[vol_pred.shape[0]//2, :, :]
             slice_y = vol_pred[:, vol_pred.shape[1]//2, :]
             slice_x = vol_pred[:, :, vol_pred.shape[2]//2]

             fig, axes = plt.subplots(1, 3, figsize=(15, 5))
             fig.suptitle(f"Iter: {iteration}, Volume Slices, k={num_gaussians}", fontsize=12)

             axes[0].imshow(slice_z, cmap='gray')
             axes[0].set_title('Z轴中心切片')
             axes[0].axis('off')

             axes[1].imshow(slice_y, cmap='gray')
             axes[1].set_title('Y轴中心切片')
             axes[1].axis('off')

             axes[2].imshow(slice_x, cmap='gray')
             axes[2].set_title('X轴中心切片')
             axes[2].axis('off')

             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
             save_path = os.path.join(vis_dir, f"volume_slices.png")
             try:
                 plt.savefig(save_path)
                 logger.info(f"已保存体积切片图像到: {save_path}")
             except Exception as e:
                 logger.error(f"保存体积切片图像失败: {e}")
             plt.close(fig)

# ----------------------------
# 训练循环 (主体逻辑不变)
# ----------------------------
def train_one_epoch_overfit(fabric, train_loader, model, optimizer, scheduler, ema, cfg, device, iteration, log_dir):
    """适用于过拟合训练的单轮训练函数"""
    model.train()
    if hasattr(model, '_forward_module'): # 如果使用了 fabric.setup
        model._forward_module.current_iter = iteration
    else:
        model.current_iter = iteration


    # 关闭梯度累积（过拟合训练通常不需要）
    accumulation_steps = 1 # 固定为1
    mem_clean_interval = getattr(cfg.opt, "mem_clean_interval", 1000) # 内存清理间隔

    # 从配置中获取每个样本要使用的图像数量
    imgs_per_obj = getattr(cfg.opt, "imgs_per_obj", 4) # 目标视角数量

    # 获取xyz边界正则化参数
    lambda_xyz_boundary = getattr(cfg.opt, "lambda_xyz_boundary", 0.01)
    min_coord = getattr(cfg.opt, "min_coord", -1.0)
    max_coord = getattr(cfg.opt, "max_coord", 1.0)

    # 获取掩码正则化参数
    lambda_mask = getattr(cfg.opt, "lambda_mask", 0.01)
    mask_reg_interval = getattr(cfg.opt, "mask_reg_interval", 1) # 控制计算频率
    mask_sample_num = getattr(cfg.opt, "mask_sample_num", 10000) # 采样点数

    # 检查是否使用固定学习率
    use_fixed_lr = getattr(cfg.opt, "use_fixed_lr", False)

    # 获取每个像素的高斯数量
    num_gaussians = getattr(cfg.model, "num_gaussians", 6)
    # logger.info(f"训练使用每个像素 {num_gaussians} 个高斯球") # 不必每次都打印

    # 创建保存体素化结果的目录
    voxel_dir = os.path.join(log_dir, "overfit", "volumes")
    os.makedirs(voxel_dir, exist_ok=True)

    effective_iter = iteration # 当前有效迭代次数
    pbar = tqdm(total=cfg.opt.iterations - iteration + 1, # 显示剩余迭代次数
                initial=iteration, # 从当前迭代开始
                desc="过拟合训练", dynamic_ncols=True, leave=True, # leave=True 保持进度条
                unit="iter")


    # 训练前清理内存
    torch.cuda.empty_cache()
    gc.collect()

    # 用于保存最后一个批次的数据，以便在训练结束时体素化
    last_batch_data = None
    last_gaussian_splats = None # 保存最后的高斯参数

    # 主训练循环，直接迭代需要的次数
    while effective_iter <= cfg.opt.iterations:

        # --- 数据加载 ---
        # 过拟合数据加载器可以只返回一次数据
        try:
             # 获取一个批次的数据
             data = next(iter(train_loader))
             last_batch_data = data # 保存数据以备后用
        except StopIteration:
             logger.error("数据加载器为空，无法继续训练。")
             break # 退出循环
        except Exception as e:
             logger.error(f"数据加载时出错: {e}", exc_info=True)
             break

        # --- 模型前向传播 ---
        # 准备输入数据
        batch_size = len(data["cameras"]) # 通常是1
        input_images = data["input_images"].to(device, non_blocking=True)
        source_cv2wT_quat = data["source_cv2wT_quat"].to(device, non_blocking=True)

        # camera_params_list 需要为模型输入构建 (B*N_in, ...) 的结构?
        # 或者模型内部处理 (B, N_in, ...) ?
        # 根据当前模型实现，似乎需要平铺列表
        angles = data["camera_params"]["angle"].to(device, non_blocking=True) # (B, N_in)
        view_to_world = data["camera_params"]["view_to_world"].to(device, non_blocking=True) # (B, N_in, 4, 4)
        N_in = angles.shape[1] # 输入图像数量

        camera_params_list = []
        for i in range(batch_size):
            for j in range(N_in):
                cp = {"angle": float(angles[i,j].item()), # 确保是 float
                      "view_to_world": view_to_world[i,j]}
                camera_params_list.append(cp)

        scanner_cfg_list = []
        for i in range(batch_size):
             # scanner_cfg 对输入视角可能都一样
             # 假设 scanner_cfg_list 长度为 B
             scanner_cfg = data["scanner_cfg"][i]
             for _ in range(N_in):
                 scanner_cfg_list.append(scanner_cfg)


        # 执行模型前向传播
        try:
            gaussian_splats = model(input_images, source_cv2wT_quat, camera_params_list, scanner_cfg_list)
            # gaussian_splats['xyz'] shape: [B, N_points, 3]
            # gaussian_splats['features'] shape: [B, N_points, F]
            # ... 其他参数
            last_gaussian_splats = {k: v.detach() for k, v in gaussian_splats.items()} # 保存最后状态
        except Exception as e:
            logger.error(f"模型前向传播失败 at iter {effective_iter}: {e}", exc_info=True)
            # 可以选择跳过此迭代或中止
            effective_iter += 1
            pbar.update(1)
            continue # 跳到下一次迭代


        # --- 损失计算 ---
        batch_reproj_loss = 0.0
        batch_ssim_loss = 0.0
        batch_tv_loss = 0.0
        batch_xyz_boundary_loss = 0.0
        batch_mask_reg_loss = 0.0

        # 用于可视化的数据
        first_sample_rendered = None
        first_sample_gt = None
        first_sample_vol_pred = None # 用于 TV 损失计算和可视化

        # 损失权重
        w_reproj = getattr(cfg.opt, "w_l12", 1.0)
        w_ssim   = getattr(cfg.opt, "w_ssim", 0.0)
        w_tv     = getattr(cfg.opt, "w_tv",   0.0)


        # 遍历 batch 内每个 sample (通常只有一个)
        for sample_idx in range(batch_size):
            # 提取当前样本的高斯参数
            try:
                 # 确保所有值都是 contiguous 的
                 gaussian_splat_sample = {k: v[sample_idx].contiguous()
                                         for k, v in gaussian_splats.items() if v is not None}
            except IndexError:
                 logger.error(f"无法获取 sample_idx={sample_idx} 的高斯参数，可能是 batch size 不匹配。")
                 continue # 跳过这个样本

            cameras = data["cameras"][sample_idx] # 获取完整的相机列表
            scanner_cfg = data["scanner_cfg"][sample_idx]
            bbox = data["bbox"][sample_idx].to(device) # 确保 bbox 在 GPU 上
            vol_mask = data.get("vol_mask") # 获取体积掩码
            if vol_mask is not None:
                 # 确保掩码在正确的设备上并且是单个样本的
                 if vol_mask.dim() == 4 and vol_mask.shape[0] == batch_size:
                      vol_mask = vol_mask[sample_idx].to(device)
                 elif vol_mask.dim() == 3:
                      vol_mask = vol_mask.to(device)
                 else:
                      logger.warning(f"体积掩码形状不符合预期: {vol_mask.shape}. 将禁用掩码正则化。")
                      vol_mask = None


            # --- 计算正则化损失 ---
            # 1. XYZ 边界正则化
            if lambda_xyz_boundary > 0 and 'xyz' in gaussian_splat_sample:
                xyz_reg_loss = xyz_boundary_regularization(
                    gaussian_splat_sample['xyz'],
                    min_coord=min_coord,
                    max_coord=max_coord,
                    lambda_xyz_boundary=lambda_xyz_boundary
                )
                batch_xyz_boundary_loss += xyz_reg_loss

            # 2. Mask 正则化 (根据间隔计算)
            if lambda_mask > 0 and vol_mask is not None and effective_iter % mask_reg_interval == 0:
                mask_xyz_loss = mask_xyz_regularization_loss(
                    gaussian_splat_sample,
                    vol_mask, # 应该是 (D,H,W) on device
                    lambda_mask,
                    sample_num=mask_sample_num
                )
                batch_mask_reg_loss += mask_xyz_loss
                # if sample_idx == 0: # 仅打印第一个样本的
                #     logger.info(f"Iter {effective_iter}: Mask XYZ Loss = {mask_xyz_loss.item():.6f}")


            # --- 计算渲染损失 ---
            # 随机选择目标视角进行渲染和比较
            total_cameras = len(cameras)
            input_image_idx = data["input_image_idx"] # 固定输入视角索引
            target_indices = [i for i in range(total_cameras) if i != input_image_idx]

            if not target_indices:
                 logger.warning("没有可用的目标视角来计算渲染损失。")
                 continue # 如果没有目标视角，无法计算渲染损失

            # 选择 imgs_per_obj 个目标视角
            if len(target_indices) > imgs_per_obj:
                selected_target_indices = torch.tensor(target_indices)[torch.randperm(len(target_indices))[:imgs_per_obj]].tolist()
            else:
                selected_target_indices = target_indices

            # 获取选中的相机对象
            selected_cameras = [cameras[i] for i in selected_target_indices]


            rendered_images = []
            gt_images = []

            # 渲染选定的目标视角
            for cam_idx, cam in enumerate(selected_cameras):
                try:
                    render_out = render(cam, gaussian_splat_sample) # cam 需要包含相机参数
                    rendered_images.append(render_out["render"].unsqueeze(0)) # (1, C, H, W)
                    gt_images.append(cam.original_image.unsqueeze(0).to(device)) # (1, C, H, W)
                except Exception as e:
                     logger.error(f"渲染相机 {cam.colmap_id} (idx {target_indices[cam_idx]}) 失败 at iter {effective_iter}: {e}", exc_info=True)
                     # 可以选择填充一个零张量或跳过这个相机
                     continue # 跳过这个失败的相机

            if not rendered_images: # 如果所有渲染都失败了
                 logger.warning("所有选定视角的渲染均失败，无法计算渲染损失。")
                 continue

            rendered_images = torch.cat(rendered_images, dim=0) # (N_targets, C, H, W)
            gt_images = torch.cat(gt_images, dim=0)         # (N_targets, C, H, W)

            # 计算 L1/L2 损失
            if cfg.opt.loss == "l2":
                pixel_loss = F.mse_loss(rendered_images, gt_images, reduction='mean')
            else: # 默认 L1
                pixel_loss = F.l1_loss(rendered_images, gt_images, reduction='mean')
            sample_reproj = pixel_loss # 已经是均值
            batch_reproj_loss += sample_reproj

            # 计算 SSIM 损失
            sample_ssim_val = 0.0
            if w_ssim > 0.0:
                try:
                     # ssim 函数期望 (B, C, H, W)
                     sample_ssim_val = (1.0 - ssim(rendered_images, gt_images))
                     batch_ssim_loss += sample_ssim_val
                except Exception as e:
                     logger.error(f"计算 SSIM 失败: {e}")

            # 计算 TV 损失 (如果需要)
            sample_tv_val = 0.0
            if w_tv > 0.0:
                 try:
                    # 使用 query 获取体积表示
                    # 定义 TV 计算所需的分辨率和范围 (可以从 scanner_cfg 获取或固定)
                    nVoxel_tv = scanner_cfg.get("nVoxel", [32, 32, 32]) # 使用配置或默认值
                    sVoxel_tv = bbox[1] - bbox[0] # 使用边界框大小
                    center_tv = (bbox[0] + bbox[1]) / 2.0 # 使用边界框中心

                    # nVoxel_tv = torch.tensor(nVoxel_tv, device=device, dtype=torch.long)
                    # tv_vol_output = query(gaussian_splat_sample, center_tv, nVoxel_tv, sVoxel_tv)

                    # 直接调用 save_voxelized_volume 获取预测体积更方便
                    # 体素化并计算 TV loss
                    vol_pred_tv = save_voxelized_volume(
                         gaussian_splat_sample, bbox, nVoxel_tv,
                         save_path=None, # 不保存文件
                         device=device, scaling_modifier=1.0)

                    if vol_pred_tv is not None:
                         sample_tv_val = tv_3d_loss(vol_pred_tv.unsqueeze(0).unsqueeze(0), reduction="mean") # tv_3d_loss 可能需要 5D 输入
                         batch_tv_loss += sample_tv_val
                         if sample_idx == 0: # 保存第一个样本的 TV 体积用于可视化
                              first_sample_vol_pred = vol_pred_tv.detach()
                    else:
                         logger.warning("TV 损失计算失败，因为体素化失败。")

                 except Exception as e:
                     logger.error(f"计算 TV 损失失败: {e}", exc_info=True)


            # 保存第一个样本的渲染结果用于可视化
            if sample_idx == 0 and rendered_images.numel() > 0:
                first_sample_rendered = rendered_images.detach()[:5]  # 最多保存5个视角
                first_sample_gt = gt_images.detach()[:5]

            # 清理渲染相关的中间变量
            del rendered_images, gt_images, selected_cameras
            if w_tv > 0.0 and 'vol_pred_tv' in locals():
                del vol_pred_tv


        # --- 汇总损失 ---
        # 平均每个 sample 的损失 (如果 batch_size > 1)
        reproj_loss = batch_reproj_loss / batch_size
        ssim_loss = batch_ssim_loss / batch_size if w_ssim > 0 else torch.tensor(0.0, device=device)
        tv_loss = batch_tv_loss / batch_size if w_tv > 0 else torch.tensor(0.0, device=device)
        xyz_boundary_loss = batch_xyz_boundary_loss / batch_size if lambda_xyz_boundary > 0 else torch.tensor(0.0, device=device)
        mask_reg_loss = batch_mask_reg_loss / batch_size if lambda_mask > 0 else torch.tensor(0.0, device=device)

        # 组装总损失
        total_loss = (w_reproj * reproj_loss +
                      w_ssim   * ssim_loss +
                      w_tv     * tv_loss +
                      xyz_boundary_loss +
                      mask_reg_loss)


        # --- 反向传播与优化 ---
        # 检查 NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"迭代 {effective_iter} 产生 NaN/Inf 损失 ({total_loss.item()})，跳过参数更新。")
            # 在这里可以考虑减小学习率或采取其他措施
        else:
            # 梯度清零
            optimizer.zero_grad(set_to_none=True) # 使用 set_to_none=True 更节省内存
            # 反向传播 (由 fabric 处理混合精度)
            try:
                fabric.backward(total_loss)
            except Exception as e:
                logger.error(f"反向传播失败 at iter {effective_iter}: {e}", exc_info=True)
                # 根据错误类型决定是否继续
                effective_iter += 1
                pbar.update(1)
                continue # 跳到下一次迭代

            # 梯度裁剪 (可选但推荐)
            fabric.clip_gradients(model, optimizer, max_norm=1.0)

            # 优化器步骤
            try:
                optimizer.step()
            except Exception as e:
                logger.error(f"优化器步骤失败 at iter {effective_iter}: {e}", exc_info=True)
                # 可能需要检查梯度或优化器状态

            # 更新学习率调度器 (如果使用)
            if not use_fixed_lr and scheduler is not None:
                scheduler.step()

            # 更新 EMA 模型 (如果使用)
            if ema is not None:
                ema.update()

        # --- 日志记录与保存 ---
        current_lr = optimizer.param_groups[0]['lr']

        # 1. 记录损失指标 (使用移植的 log_metrics)
        if effective_iter % cfg.logging.loss_log == 0:
            loss_dict_log = {
                "total_loss": total_loss,
                "reproj_loss": reproj_loss,
                "ssim_loss": ssim_loss,
                "tv_loss": tv_loss,
                "xyz_boundary_loss": xyz_boundary_loss,
                "mask_reg_loss": mask_reg_loss,
                # 可视化用的数据不需要传给 log_metrics
            }
            log_metrics(loss_dict_log, effective_iter, current_lr)


        # 2. 记录可视化结果 (WandB + 本地)
        if effective_iter % cfg.logging.render_log == 0:
            vis_dict_log = {
                 # 传递损失值可能有用
                "total_loss": total_loss,
                "reproj_loss": reproj_loss,
                 # 传递用于可视化的图像和体积
                "rendered_images": first_sample_rendered,
                "gt_images": first_sample_gt,
                "vol_pred": first_sample_vol_pred # TV计算用的体积或单独查询的体积
            }
            # 使用移植的 log_visualizations (WandB)
            log_visualizations(vis_dict_log, effective_iter, cfg)
            # 使用本地保存函数
            local_log_visualizations(vis_dict_log, effective_iter, cfg, log_dir)

        # 3. 记录参数直方图 (使用移植的 log_parameter_histograms)
        if effective_iter % cfg.logging.histogram_log == 0 and gaussian_splats is not None:
             # 传递原始的、包含 batch 维度的 gaussian_splats
             log_parameter_histograms(gaussian_splats, effective_iter)

        # 4. 保存检查点 (使用移植的 save_checkpoint)
        if effective_iter % cfg.logging.ckpt_iterations == 0 and effective_iter > 0:
            overfit_ckpt_dir = os.path.join(log_dir, "overfit") # 保存到 overfit 子目录
            os.makedirs(overfit_ckpt_dir, exist_ok=True)
            save_checkpoint(model, optimizer, scheduler, ema,
                            effective_iter, 0.0, # 过拟合通常没有验证指标
                            overfit_ckpt_dir, f"model_overfit_{effective_iter}.pth")
            # 保存最新检查点
            save_checkpoint(model, optimizer, scheduler, ema,
                             effective_iter, 0.0,
                             overfit_ckpt_dir, "model_overfit_latest.pth")


        # 5. 定期体素化并保存 (使用移植的 save_voxelized_volume)
        voxel_interval = getattr(cfg.logging, "voxel_interval", 1000)
        if effective_iter % voxel_interval == 0 and effective_iter > 0 and gaussian_splats is not None:
             if batch_size == 1: # 确保只处理单个样本
                 gaussian_splat_voxel = {k: v[0].contiguous() for k, v in gaussian_splats.items()}
                 bbox_voxel = data["bbox"][0].to(device)
                 scanner_cfg_voxel = data["scanner_cfg"][0]
                 nVoxel_save = scanner_cfg_voxel.get("nVoxel", [128, 128, 128]) # 使用配置或默认高分辨率
                 save_path_nii = os.path.join(voxel_dir, f"volume_iter_{effective_iter}_k{num_gaussians}.nii.gz")
                 save_voxelized_volume(gaussian_splat_voxel, bbox_voxel, nVoxel_save, save_path_nii, device)
             else:
                  logger.warning("Batch size > 1, 跳过定期体素化保存。")


        # --- 清理与迭代更新 ---
        # 内存清理
        # if effective_iter % mem_clean_interval == 0:
        #     torch.cuda.empty_cache()
        #     gc.collect()

        # 清理本次迭代的变量
        del data, input_images, source_cv2wT_quat, angles, view_to_world
        del gaussian_splats, loss_dict_log
        if 'vis_dict_log' in locals(): del vis_dict_log
        if first_sample_rendered is not None: del first_sample_rendered
        if first_sample_gt is not None: del first_sample_gt
        if first_sample_vol_pred is not None: del first_sample_vol_pred


        # 更新进度条
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "reproj": f"{reproj_loss.item():.4f}",
            "ssim": f"{ssim_loss.item():.4f}" if isinstance(ssim_loss, torch.Tensor) else f"{ssim_loss:.4f}",
            "tv": f"{tv_loss.item():.4f}" if isinstance(tv_loss, torch.Tensor) else f"{tv_loss:.4f}",
            "xyz": f"{xyz_boundary_loss.item():.4f}" if isinstance(xyz_boundary_loss, torch.Tensor) else f"{xyz_boundary_loss:.4f}",
            "mask": f"{mask_reg_loss.item():.6f}" if isinstance(mask_reg_loss, torch.Tensor) else f"{mask_reg_loss:.6f}",
            "lr": f"{current_lr:.2e}"
        })
        pbar.update(1)

        # 增加迭代计数器
        effective_iter += 1


    pbar.close() # 训练循环结束，关闭进度条

    # --- 训练结束后操作 ---
    # 保存最终的体素化结果
    if last_gaussian_splats is not None and last_batch_data is not None:
         if batch_size == 1: # 确保只处理单个样本
             final_splat_sample = {k: v[0].contiguous() for k, v in last_gaussian_splats.items()}
             final_bbox = last_batch_data["bbox"][0].to(device)
             final_scanner_cfg = last_batch_data["scanner_cfg"][0]

             # 保存一个标准分辨率版本
             nVoxel_final = final_scanner_cfg.get("nVoxel", [128, 128, 128])
             save_path_final = os.path.join(voxel_dir, f"volume_final_iter_{effective_iter-1}_k{num_gaussians}.nii.gz")
             save_voxelized_volume(final_splat_sample, final_bbox, nVoxel_final, save_path_final, device)

             # 保存一个更高分辨率版本 (可选)
             # high_res_nVoxel = [res*2 for res in nVoxel_final] # 例如，分辨率加倍
             # save_path_high_res = os.path.join(voxel_dir, f"volume_final_high_res_iter_{effective_iter-1}_k{num_gaussians}.nii.gz")
             # save_voxelized_volume(final_splat_sample, final_bbox, high_res_nVoxel, save_path_high_res, device)
         else:
              logger.warning("Batch size > 1, 跳过最终体素化保存。")


    # 最后彻底清理内存
    torch.cuda.empty_cache()
    gc.collect()

    return effective_iter # 返回完成的迭代次数 + 1


# ----------------------------
# 模型与优化器初始化 (主体逻辑不变)
# ----------------------------
def init_model(cfg: DictConfig, fabric: Fabric, device):
    # 启用cudnn算法自动选择和内存优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # 使用更保守的内存使用方式，避免过度内存占用
    model = GaussianSplatPredictor(cfg) # 假设这是正确的模型类
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params}")

    # 尝试 channels_last 内存格式 (可能提高卷积性能)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception as e:
        logger.warning(f"设置 channels_last 失败: {e}. 继续使用默认格式。")

    # 移动模型到设备 (Fabric setup 之前)
    model = model.to(device)


    # 设置优化器
    # 区分不同的参数组 (如果需要)
    # param_groups = [{'params': model.network.parameters(), 'lr': cfg.opt.base_lr}]
    # 目前假设所有参数使用相同配置
    weight_decay = getattr(cfg.opt, "weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # 只优化需要梯度的参数
        lr=cfg.opt.base_lr,
        eps=getattr(cfg.opt, "eps", 1e-8),
        betas=getattr(cfg.opt, "betas", (0.9, 0.999)),
        weight_decay=weight_decay
    )

    # 配置学习率调度器 (如果使用)
    use_fixed_lr = getattr(cfg.opt, "use_fixed_lr", False)
    scheduler = None
    if not use_fixed_lr:
        total_steps = cfg.opt.iterations
        warmup_steps = getattr(cfg.opt, "warmup_steps", 0)
        min_lr_factor = getattr(cfg.opt, "min_lr_factor", 0.01) # 使用 min_lr_factor

        def lr_lambda(current_step):
            # 线性预热
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # 余弦衰减到 min_lr
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # 确保 progress 在 [0, 1] 范围内
            progress = min(max(progress, 0.0), 1.0)
            # 计算衰减因子 (从 1 衰减到 min_lr_factor)
            decay_factor = min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return decay_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info("使用带预热和余弦衰减的学习率调度器")
    else:
         logger.info(f"使用固定学习率: {cfg.opt.base_lr}")
         # 固定学习率时 scheduler 为 None


    # 配置 EMA (如果使用)
    ema = None
    if getattr(cfg.opt.ema, "use", False):
        try:
            from ema_pytorch import EMA
            ema = EMA(
                model,
                beta=getattr(cfg.opt.ema, "beta", 0.995),
                update_every=getattr(cfg.opt.ema, "update_every", 10),
                update_after_step=getattr(cfg.opt.ema, "update_after_step", 100)
            )
            ema = fabric.to_device(ema) # 移动 EMA 相关状态到设备
            logger.info("已启用 EMA")
        except ImportError:
            logger.warning("无法导入 ema-pytorch。EMA 将被禁用。")
        except Exception as e:
            logger.error(f"初始化 EMA 失败: {e}. EMA 将被禁用。")


    # 使用 Fabric 设置模型和优化器 (处理分布式和混合精度)
    # Fabric 会包装模型和优化器
    logger.info(f"使用 Fabric ({fabric.strategy.strategy_name}) 设置模型和优化器...")
    model, optimizer = fabric.setup(model, optimizer)


    # Fabric setup 后 scheduler 需要用 fabric 包装的 optimizer 重新创建或更新？
    # LambdaLR 通常不需要重新创建，因为它只依赖优化器实例的引用。
    # 但如果 fabric 替换了优化器实例，可能需要重新创建。检查 fabric 文档。
    # Lightning Fabric 通常会正确处理优化器状态，LambdaLR 基于 step 更新，应该没问题。


    # 清理初始化过程中的缓存
    torch.cuda.empty_cache()

    return model, optimizer, scheduler, ema


# 为过拟合训练定义固定学习率版本的init_model
def init_model_fixed_lr(cfg: DictConfig, fabric: Fabric, device):
    """创建使用固定学习率的模型和优化器
    (基本同上，只是强制 scheduler 为 None)
    """
    # 启用cudnn算法自动选择和内存优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = GaussianSplatPredictor(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params}")

    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception as e:
        logger.warning(f"设置 channels_last 失败: {e}. 继续使用默认格式。")

    model = model.to(device)

    weight_decay = getattr(cfg.opt, "weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.opt.base_lr, # 直接使用固定学习率
        eps=getattr(cfg.opt, "eps", 1e-8),
        betas=getattr(cfg.opt, "betas", (0.9, 0.999)),
        weight_decay=weight_decay
    )

    # 固定学习率，scheduler 设置为 None
    scheduler = None
    logger.info(f"为过拟合训练创建优化器，固定学习率: {cfg.opt.base_lr}")


    # 配置 EMA (如果使用)
    ema = None
    if getattr(cfg.opt.ema, "use", False):
        try:
            from ema_pytorch import EMA
            ema = EMA(
                model,
                beta=getattr(cfg.opt.ema, "beta", 0.995),
                update_every=getattr(cfg.opt.ema, "update_every", 10),
                update_after_step=getattr(cfg.opt.ema, "update_after_step", 100)
            )
            ema = fabric.to_device(ema)
            logger.info("已启用 EMA")
        except ImportError:
            logger.warning("无法导入 ema-pytorch。EMA 将被禁用。")
        except Exception as e:
            logger.error(f"初始化 EMA 失败: {e}. EMA 将被禁用。")


    # 使用 Fabric 设置
    logger.info(f"使用 Fabric ({fabric.strategy.strategy_name}) 设置模型和优化器...")
    model, optimizer = fabric.setup(model, optimizer)

    torch.cuda.empty_cache()

    return model, optimizer, scheduler, ema


# ----------------------------
# 主函数 (主体逻辑不变)
# ----------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='过拟合训练模型到单个CT投影')
    parser.add_argument('--proj_dir', type=str, default='/Disk_10TB/zhouhaowei/data/train/LIDC-IDRI-0001.20000101.3000566.1_cone', help='投影目录路径')
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml', help='基础配置文件路径')
    # 数据参数
    parser.add_argument('--input_images', type=int, default=1, help='使用多少张图像作为输入')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小 (过拟合通常为1)')
    # 模型参数
    parser.add_argument('--num_gaussians', type=int, default=4, help='每个像素的Gaussian数量')
    # 优化参数
    parser.add_argument('--iterations', type=int, default=100000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=1e-4, help='基础学习率') # 调整默认值
    parser.add_argument('--imgs_per_obj', type=int, default=4, help='每次选择多少个目标投影计算渲染损失')
    parser.add_argument('--use_fixed_lr', action='store_true', help='是否使用固定学习率（无衰减）')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW 权重衰减') # 默认为0
    parser.add_argument('--lambda_xyz_boundary', type=float, default=0.01, help='XYZ边界正则化权重')
    parser.add_argument('--lambda_mask', type=float, default=0.01, help='Mask正则化权重')
    parser.add_argument('--mask_reg_interval', type=int, default=10, help='Mask正则化计算频率 (每 N 次迭代)')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='预训练检查点路径 (模型或完整状态)')
    # 日志与保存参数
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录根路径，默认为 ./output')
    parser.add_argument('--ckpt_interval', type=int, default=1000, help='保存检查点间隔')
    parser.add_argument('--loss_log_interval', type=int, default=10, help='损失记录间隔')
    parser.add_argument('--render_log_interval', type=int, default=10, help='渲染结果记录间隔')
    parser.add_argument('--histogram_log_interval', type=int, default=200, help='参数直方图记录间隔')
    parser.add_argument('--voxel_interval', type=int, default=1000, help='体素化保存间隔')
    parser.add_argument('--disable_wandb', action='store_true', help='禁用 WandB 日志')


    args = parser.parse_args()

    # --- 配置加载与合并 ---
    # 确保投影目录存在
    proj_dir = args.proj_dir
    if not os.path.isdir(proj_dir): # 检查是否是目录
        logger.error(f"指定的过拟合数据路径不是一个有效目录: {proj_dir}")
        sys.exit(1) # 退出程序

    # 从配置文件加载基础配置
    try:
        with open(args.config_path, 'r') as f:
            cfg_dict = OmegaConf.load(f)
        cfg = OmegaConf.create(cfg_dict)
        logger.info(f"从 {args.config_path} 加载基础配置")
    except FileNotFoundError:
         logger.error(f"配置文件未找到: {args.config_path}")
         sys.exit(1)
    except Exception as e:
         logger.error(f"加载配置文件失败: {e}")
         sys.exit(1)


    # --- 合并命令行参数到配置对象 ---
    # data 部分
    OmegaConf.set_struct(cfg, False) # 允许添加新键
    if 'data' not in cfg: cfg.data = OmegaConf.create()
    cfg.data.input_images = args.input_images
    cfg.data.batch_size = args.batch_size # 过拟合通常为 1

    # model 部分
    if 'model' not in cfg: cfg.model = OmegaConf.create()
    cfg.model.num_gaussians = args.num_gaussians

    # opt 部分
    if 'opt' not in cfg: cfg.opt = OmegaConf.create()
    cfg.opt.iterations = args.iterations
    cfg.opt.base_lr = args.lr
    cfg.opt.imgs_per_obj = args.imgs_per_obj
    cfg.opt.use_fixed_lr = args.use_fixed_lr
    cfg.opt.weight_decay = args.weight_decay # 添加权重衰减
    cfg.opt.lambda_xyz_boundary = args.lambda_xyz_boundary # 添加正则化权重
    cfg.opt.lambda_mask = args.lambda_mask
    cfg.opt.mask_reg_interval = args.mask_reg_interval
    # 添加预训练检查点路径 (如果提供)
    if args.pretrained_ckpt:
        cfg.opt.pretrained_ckpt = args.pretrained_ckpt
    else:
        cfg.opt.pretrained_ckpt = None # 确保存在该字段

    # logging 部分
    if 'logging' not in cfg: cfg.logging = OmegaConf.create()
    cfg.logging.ckpt_iterations = args.ckpt_interval
    cfg.logging.loss_log = args.loss_log_interval
    cfg.logging.render_log = args.render_log_interval
    cfg.logging.histogram_log = args.histogram_log_interval # 使用命令行参数
    cfg.logging.voxel_interval = args.voxel_interval # 使用命令行参数
    cfg.logging.wandb_run_name = f"overfit_{os.path.basename(proj_dir)}_k{args.num_gaussians}" # 包含 k 值
    cfg.logging.disable_wandb = args.disable_wandb # 添加禁用标志

    # 添加 cam_embd 默认配置 (如果不存在)
    if 'cam_embd' not in cfg:
         cfg.cam_embd = OmegaConf.create({
             "embedding": "pose", # 或其他默认值
             "encode_embedding": None,
             "dimension": 32,
             "method": "film" # 或其他默认值
         })

    # general 部分 (如果需要混合精度等)
    if 'general' not in cfg: cfg.general = OmegaConf.create()
    # cfg.general.mixed_precision = True # 或者根据需要设置

    OmegaConf.set_struct(cfg, True) # 锁定配置结构

    # 打印最终配置
    logger.info("最终配置:")
    logger.info(OmegaConf.to_yaml(cfg))


    # --- 初始化训练环境 ---
    logger.info(f"使用投影目录进行过拟合训练: {proj_dir}")
    experiment_name = os.path.basename(proj_dir) # 使用 proj_dir 名称作为实验名

    # 确定输出目录
    output_root = args.output_dir if args.output_dir else "./output" # 默认 ./output

    # 使用移植的 init_training_overfit 初始化 Fabric, Device, Log dir
    fabric, device, log_dir = init_training_overfit(cfg, experiment_name, output_root)

    # 创建 overfit 子目录 (用于保存特定于过拟合的检查点和结果)
    overfit_dir = os.path.join(log_dir, "overfit")
    os.makedirs(overfit_dir, exist_ok=True)
    logger.info(f"日志和结果将保存在: {log_dir}")


    # --- 数据加载 ---
    # 创建单样本数据集，并提前缓存 vol_mask 到 GPU
    logger.info("初始化数据集...")
    try:
        dataset = SingleSampleDataset(proj_dir, device=device)
    except Exception as e:
         logger.error(f"创建数据集失败: {e}", exc_info=True)
         sys.exit(1)

    # 创建数据加载器
    import functools
    # 确保 collate 函数使用正确的 input_images_count
    collate_fn_with_cfg = functools.partial(overfit_collate_fn, input_images_count=cfg.data.input_images)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False, # 过拟合不需要 shuffle，每次数据一样
        num_workers=0,  # 过拟合训练不需要多进程
        pin_memory=False, # 单个样本通常不需要 pin_memory
        collate_fn=collate_fn_with_cfg
    )
    # 使用 Fabric 设置数据加载器 (处理分布式采样等，虽然这里是单卡)
    dataloader = fabric.setup_dataloaders(dataloader)
    logger.info("数据加载器已设置")


    # --- 模型、优化器、调度器、EMA 初始化 ---
    logger.info("初始化模型和优化器...")
    if args.use_fixed_lr:
        model, optimizer, scheduler, ema = init_model_fixed_lr(cfg, fabric, device)
    else:
        model, optimizer, scheduler, ema = init_model(cfg, fabric, device)


    # --- 加载检查点 ---
    logger.info("尝试加载检查点...")
    # 使用移植的 load_checkpoint 函数
    # 注意：load_checkpoint 需要 fabric setup 后的 model 和 optimizer
    first_iter, _ = load_checkpoint(model, optimizer, scheduler, ema, log_dir, device, cfg)
    # 在过拟合中，通常不关心 best_metric

    # --- 开始训练 ---
    start_iter = first_iter # 从加载的迭代或0开始
    logger.info(f"开始过拟合训练，起始迭代: {start_iter}, 总迭代: {cfg.opt.iterations}")
    start_time = datetime.datetime.now()


    try:
        # 调用训练循环
        # 注意：train_one_epoch_overfit 内部处理从 start_iter 到 cfg.opt.iterations 的循环
        final_iter = train_one_epoch_overfit(
            fabric, dataloader, model, optimizer, scheduler, ema,
            cfg, device, start_iter, log_dir
        )

    except KeyboardInterrupt:
        logger.info("检测到键盘中断，正在保存模型...")
        # 确保保存到 overfit 目录
        save_checkpoint(
            model, optimizer, scheduler, ema,
            final_iter, 0.0, overfit_dir, "model_overfit_interrupt.pth"
        )
        logger.info("已保存中断模型。")

    except Exception as e:
        logger.error(f"训练过程中发生严重错误: {str(e)}", exc_info=True)
        # 尝试保存最后状态
        try:
            save_checkpoint(
                model, optimizer, scheduler, ema,
                final_iter, 0.0, overfit_dir, "model_overfit_error.pth"
            )
            logger.info("已尝试保存错误状态模型。")
        except Exception as save_e:
            logger.error(f"保存错误状态模型失败: {save_e}")

    finally:
        # 训练结束（正常完成或中断/错误后）
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        logger.info(f"训练结束。总时长: {duration}")
        logger.info(f"完成迭代: {final_iter-1}/{cfg.opt.iterations}") # final_iter 是下一个迭代的编号

        # 保存最终模型 (如果训练完成或中断)
        if 'final_iter' in locals(): # 确保 final_iter 已定义
            save_checkpoint(
                 model, optimizer, scheduler, ema,
                 final_iter - 1, # 保存最后完成的迭代状态
                 0.0, overfit_dir, "model_overfit_final.pth"
            )
            logger.info(f"最终模型已保存到: {os.path.join(overfit_dir, 'model_overfit_final.pth')}")


        # 清理资源
        del model, optimizer, scheduler, dataloader, dataset
        if ema is not None:
            del ema
        if wandb.run is not None and is_wandb_enabled():
             wandb.finish()
             logger.info("WandB run 已结束。")

        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"过拟合训练完成，结果保存在: {log_dir}")
        logger.info("="*50)


# 确保在脚本主入口执行
if __name__ == "__main__":
    # 设置可见GPU (如果需要，但最好通过 CUDA_VISIBLE_DEVICES 环境变量设置)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # 确保 PYTHONPATH 包含项目根目录
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # if project_root not in sys.path:
    #     sys.path.insert(0, project_root)
    # logger.info(f"项目根目录已添加到 PYTHONPATH: {project_root}")

    main()