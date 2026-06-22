# -*- coding: utf-8 -*-
#CUDA_VISIBLE_DEVICES=0,2
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
from lightning.fabric import Fabric # 导入 Lightning Fabric
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import sys
import glob
from torch.cuda.amp import GradScaler, autocast
import pickle
import io

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss, ssim, tv_3d_loss
from r2_gaussian.gaussian import render, query # 保持原始的渲染和查询函数
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset
from eval import evaluate_model # 假设 evaluate_model 兼容 Fabric/DDP 或在 rank 0 运行

# ----------------------------
# 新增的正则化函数
# ----------------------------
def xyz_boundary_regularization(xyz, min_coord=-1.0, max_coord=1.0, lambda_xyz_boundary=1.0):
    """
    惩罚超出 [min_coord, max_coord] 边界的 xyz 坐标。

    Args:
        xyz (torch.Tensor): 预测的世界坐标，形状 (N_points, 3)。注意：传入前确保是单样本的坐标。
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
    # 注意：这里直接在传入的单一样本 xyz 上计算 mean
    boundary_violation = (lower_penalty + upper_penalty).mean()

    return lambda_xyz_boundary * boundary_violation

# ----------------------------
# 日志设置 (稍作修改以更好适应 DDP)
# ----------------------------
# 使用 Fabric rank 区分，避免日志冲突
logger = logging.getLogger(__name__) # 获取 logger 实例

def setup_logging(log_dir, rank):
    """配置日志记录，仅在 rank 0 上创建文件处理器"""
    # 移除所有现有处理器，防止重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_level = logging.INFO # 默认级别
    log_format = f"%(asctime)s - RANK {rank} - %(levelname)s - %(message)s" # 添加 Rank 信息
    formatter = logging.Formatter(log_format)

    # 所有 Rank 都添加控制台处理器，但只有 Rank 0 打印 INFO，其他打印 ERROR
    console_handler = logging.StreamHandler(sys.stdout) # 使用 stdout 避免 tqdm 冲突
    console_handler.setFormatter(formatter)
    if rank == 0:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.ERROR) # 其他 rank 只显示错误信息
    logger.addHandler(console_handler)

    # 仅 Rank 0 创建和写入日志文件
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info(f"日志将保存到: {log_file} (仅 Rank 0)")

    # 设置 logger 的总级别
    logger.setLevel(logging.INFO)

def is_wandb_enabled():
    """
    辅助函数：判断 wandb 是否可用或是否被显式禁用
    """
    # 检查环境变量 WANDB_DISABLED，如果设置为 "true"，则禁用 wandb
    return os.getenv("WANDB_DISABLED", "false").lower() != "true"

def init_training(cfg: DictConfig):
    """
    初始化 Fabric, 设备, WandB (仅 Rank 0), 和日志。
    """
    torch.set_float32_matmul_precision('high') # 设置矩阵乘法精度

    # --- Fabric 初始化 ---
    # 从配置读取设备数量和混合精度设置
    num_devices = getattr(cfg.general, "num_devices", 1) # 默认为1
    mixed_precision = getattr(cfg.general, "mixed_precision", False)
    precision = "16-mixed" if mixed_precision else "32-true" # DDP 推荐用 "32-true" 或 "bf16-mixed"
    strategy = "ddp" if num_devices > 1 else "auto" # 显式使用 DDP
    
    fabric = Fabric(accelerator="cuda", devices=num_devices, strategy=strategy, precision=precision)
    fabric.launch() # 启动 Fabric，会设置分布式环境
    print("fabric.launch() sucess")
    # 获取设备信息 (在 Fabric launch 后)
    device = fabric.device # 直接使用 fabric 的设备属性
    scaler = GradScaler(enabled=mixed_precision)
    # --- 日志和 WandB (仅 Rank 0) ---
    log_dir = None
    if fabric.is_global_zero: # 仅在全局 Rank 0 进程上执行
        run_name = getattr(cfg.logging, "wandb_run_name", "ddp_run") # 运行名称
        log_dir = os.path.join(os.getcwd(), "logs", run_name) # 日志目录
        setup_logging(log_dir, fabric.global_rank) # 设置日志，传入 rank

        disable_wandb = getattr(cfg.logging, "disable_wandb", False)
        if disable_wandb or not is_wandb_enabled():
            logger.info("WandB 已禁用或未配置，仅使用本地日志。")
            os.environ["WANDB_DISABLED"] = "true" # 确保禁用
        else:
            # --- WandB 初始化 (仅 Rank 0) ---
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
            project_name = getattr(cfg.logging, "wandb_project", "MultiGPUTraining")
            wandb_offline = getattr(cfg.logging, "wandb_offline", False)

            if wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
                logger.info("WandB将在离线模式下运行")

            os.environ["WANDB_START_METHOD"] = "thread" # 推荐使用线程模式

            try:
                # 检查是否需要恢复运行 (基于是否存在 wandb 目录中的 run_id)
                # 注意：这里的恢复逻辑可能需要根据 wandb 的具体实现调整
                wandb_dir = os.path.join(log_dir, "wandb")
                resume_id = None
                if os.path.exists(wandb_dir):
                     run_files = glob.glob(os.path.join(wandb_dir, "latest-run", "run-*.wandb"))
                     if run_files:
                         run_id = os.path.basename(run_files[0]).split("run-")[1].split(".wandb")[0]
                         resume_id = run_id
                         logger.info(f"尝试恢复 WandB 运行，ID: {resume_id}")

                wandb.init(
                    project=project_name,
                    name=run_name,
                    id=resume_id, # 使用找到的 ID 恢复
                    resume="allow", # 允许恢复
                    reinit=True, # 允许重新初始化 (以防万一)
                    config=wandb_config,
                    dir=log_dir, # 指定 wandb 目录
                    settings=wandb.Settings(console="off") # 关闭 wandb 的控制台镜像
                )
                logger.info(f"WandB 初始化成功。项目: {project_name}, 名称: {run_name}")
            except Exception as e:
                logger.error(f"WandB 初始化失败: {e}")
                logger.info("将继续训练，但不会记录到 WandB。")
                os.environ["WANDB_DISABLED"] = "true" # 出错时也禁用

    # 在所有进程中同步 WandB 禁用状态 (如果 rank 0 失败或禁用)
    is_disabled_tensor = torch.tensor(int(os.environ.get("WANDB_DISABLED", "false").lower() == "true"), device=fabric.device)
    fabric.broadcast(is_disabled_tensor, src=0)
    if is_disabled_tensor.item() == 1:
        os.environ["WANDB_DISABLED"] = "true"

    return fabric, device, log_dir, scaler # 返回 Fabric 实例, 设备, 和日志目录 (rank 0才有值)

# ----------------------------
# 数据加载与 collate (保持不变，Fabric 会处理分布式采样)
# ----------------------------
def ct_collate_fn(batch, cfg):
    """
    batch 中每个 sample 的结构：
        sample["cameras"]         # List[Camera]，每个 Camera 有 .original_image, .angle, .view_world_transform
        sample["scanner_cfg"]     # dict
        sample["vol"]             # Tensor (D, H, W)
        # sample["vol_mask"]        # Tensor (D, H, W)
        sample["scene_scale"]     # float 或 Tensor 标量
        sample["bbox"]            # Tensor (2, 3)
        sample["source_cv2wT_quat"]  # Tensor (N_total, 4)
    """

    bs = len(batch)
    N = cfg.data.input_images  # 你配置中用来输入的视图数

    # 1) 堆叠多视角输入图像 => (bs, N, C, H, W)
    input_images = torch.stack([
        torch.stack([cam.original_image for cam in sample["cameras"][:N]], dim=0)
        for sample in batch
    ], dim=0)

    # 2) 堆叠角度和 view_to_world => (bs, N) 和 (bs, N, 4, 4)
    angles = torch.stack([
        torch.tensor([cam.angle for cam in sample["cameras"][:N]], dtype=torch.float32)
        for sample in batch
    ], dim=0)
    view2world = torch.stack([
        torch.stack([cam.view_world_transform for cam in sample["cameras"][:N]], dim=0)
        for sample in batch
    ], dim=0)

    # 3) 源四元数 => (bs, N, 4)
    quats = torch.stack([
        sample["source_cv2wT_quat"][:N]
        for sample in batch
    ], dim=0)

    # 4) 体数据及 mask => (bs, D, H, W)
    vols      = torch.stack([sample["vol"]      for sample in batch], dim=0)
    # vols_mask = torch.stack([sample["vol_mask"] for sample in batch], dim=0)

    # 5) 其它标量或列表
    scene_scales  = torch.tensor([sample["scene_scale"] for sample in batch], dtype=torch.float32)
    bboxes        = torch.stack([sample["bbox"] for sample in batch], dim=0)
    scanner_cfgs  = [sample["scanner_cfg"] for sample in batch]
    cameras_lists = [sample["cameras"]    for sample in batch]  # 保留原结构

    return {
        "input_images":      input_images,
        "camera_params":     {"angle": angles, "view_to_world": view2world},
        "source_cv2wT_quat": quats,
        "scanner_cfg":       scanner_cfgs,
        "vol":               vols,
        # "vol_mask":          vols_mask,
        "scene_scale":       scene_scales,
        "bbox":              bboxes,
        "cameras":           cameras_lists
    }



def get_dataloaders(cfg: DictConfig, fabric: Fabric):
    """获取训练和测试数据加载器，Fabric 会自动处理分布式采样"""
    if fabric.is_global_zero:
        logger.info("初始化数据集...")
    train_dataset = get_dataset(data_path=cfg.data.data_path, type="train")
    test_dataset = get_dataset(data_path=cfg.data.data_path, type="test")
    if fabric.is_global_zero:
        logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # batch_size 是全局批次大小，Fabric 会自动分配到各个设备
    global_batch_size = getattr(cfg.data, "batch_size", 4)
    num_workers = getattr(cfg.data, "num_workers", 0) # DDP 下使用多 worker 需要小心
    pin_memory = getattr(cfg.data, "pin_memory", False)

    # 简单的内存检查 (仅 Rank 0 打印信息)
    if fabric.is_global_zero and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_mem < 24:
            logger.info(f"GPU 0 内存 ({gpu_mem:.1f}GB) 可能较小，请确保全局批次大小 ({global_batch_size}) 合适。")
        if num_workers > 0 and multiprocessing.get_start_method(allow_none=True) != 'spawn':
             logger.warning("多进程数据加载 (num_workers > 0) 在 DDP 中推荐使用 'spawn' 启动方法，请检查或设置 multiprocessing.set_start_method('spawn', force=True)")

    import functools
    collate_fn_with_cfg = functools.partial(ct_collate_fn, cfg=cfg)

    # 创建 DataLoader 时 batch_size 使用全局大小
    train_loader = DataLoader(
        train_dataset,
        batch_size=global_batch_size, # 使用全局批次大小
        shuffle=True, # DistributedSampler 会处理 shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_with_cfg,
        persistent_workers=num_workers > 0, # 如果使用 worker，可以持久化
        multiprocessing_context='spawn' if num_workers > 0 else None
    )

    # 验证集通常不需要 shuffle，batch_size 可以设为 1 或适应验证逻辑的批次
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # 验证时通常用单样本
        shuffle=False,
        num_workers=0, # 验证时一般不用多 worker
        pin_memory=False,
        collate_fn=collate_fn_with_cfg
    )

    # 使用 Fabric 设置 DataLoader，会自动添加 DistributedSampler
    # setup_dataloaders 会自动计算每个进程的 local_batch_size
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=True)
    test_loader = fabric.setup_dataloaders(test_loader, use_distributed_sampler=False) # 验证集通常不需分布式采样

    return train_loader, test_loader

# ----------------------------
# 模型与优化器 (EMA 初始化移到 main 函数中，在 setup 后进行)
# ----------------------------
def init_model_and_opt(cfg: DictConfig, fabric: Fabric):
    """初始化模型和优化器，但不进行 Fabric setup"""
    if fabric.is_global_zero:
        logger.info("初始化模型和优化器...")
        torch.backends.cudnn.benchmark = True # 启用 cudnn 优化
        torch.backends.cudnn.enabled = True

    model = GaussianSplatPredictor(cfg)
    if fabric.is_global_zero:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数 (单个副本): {total_params}")

    # 内存格式和梯度初始化，在所有 rank 上执行
    model = model.to(memory_format=torch.channels_last)
    for param in model.parameters():
        param.grad = None # 节约内存

    # --- 优化器设置 ---
    param_groups = [{'params': model.network.parameters(), 'lr': cfg.opt.base_lr}]
    weight_decay = getattr(cfg.opt, "weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.opt.base_lr, # 初始学习率，调度器会覆盖
        eps=1e-8,
        betas=cfg.opt.betas,
        weight_decay=weight_decay
    )

    # --- 学习率调度器 ---
    total_steps = cfg.opt.iterations
    warmup_steps = cfg.opt.warmup_steps
    min_lr_factor = getattr(cfg.opt, "min_lr_factor", 0.01) # 从 cfg 读取
    # 添加固定学习率选项，默认为False（使用原有的余弦退火）
    use_fixed_lr = getattr(cfg.opt, "use_fixed_lr", True)

    def lr_lambda(current_step):
        if use_fixed_lr:
            # 使用固定学习率，始终返回1.0（保持初始学习率）
            return 1.0
        else:
            # 原有的余弦退火策略
            if current_step < warmup_steps:
                # 线性预热 (从 0.1 * base_lr 到 base_lr)
                return 0.1 + 0.9 * (current_step / max(1, warmup_steps))
            # Cosine decay
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            # 确保 progress 不超过 1
            progress = min(progress, 1.0)
            # Cosine 退火到 min_lr_factor * base_lr
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 如果使用固定学习率，记录到日志
    if fabric.is_global_zero:
        if use_fixed_lr:
            logger.info(f"将使用固定学习率: {cfg.opt.base_lr}")
        else:
            logger.info(f"将使用余弦退火学习率调度，最小学习率因子: {min_lr_factor}")

    if fabric.is_global_zero and getattr(cfg.general, "mixed_precision", False):
        logger.info("已配置混合精度训练。")

    # 注意：此时不进行 fabric.setup，返回原始模型和优化器
    return model, optimizer, scheduler

def load_checkpoint(fabric: Fabric, model, # 不再需要传入 optimizer, scheduler
                    ema, log_dir, cfg: DictConfig):
    """
    在 Rank 0 上查找检查点，并广播起始迭代和最佳指标。
    返回检查点内容 (rank 0才有值) 和迭代信息。
    """
    first_iter = 0
    best_metric = 0.0
    checkpoint = None # 用于存储加载的检查点内容

    # 仅 Rank 0 查找和加载检查点文件
    if fabric.is_global_zero:
        ckpt_path = None
        # 优先加载指定的预训练模型
        if cfg.opt.pretrained_ckpt is not None:
            pretrained_path = cfg.opt.pretrained_ckpt
            if os.path.isfile(pretrained_path):
                ckpt_path = pretrained_path
            elif os.path.isdir(pretrained_path):
                # 在目录中查找特定名称的检查点
                for ckpt_name in ["model_interrupt.pth", "model_best.pth", "model_latest.pth"]:
                    candidate_path = os.path.join(pretrained_path, ckpt_name)
                    if os.path.isfile(candidate_path):
                        ckpt_path = candidate_path
                        break
        # 如果没有预训练模型，则尝试加载中断的检查点
        if ckpt_path is None:
            interrupt_ckpt = os.path.join(log_dir, "model_interrupt.pth") # log_dir 只有 rank 0 有效
            if os.path.isfile(interrupt_ckpt):
                ckpt_path = interrupt_ckpt
        # 如果没有中断的，尝试加载最新的
        if ckpt_path is None:
            latest_ckpt = os.path.join(log_dir, "model_latest.pth")
            if os.path.isfile(latest_ckpt):
                ckpt_path = latest_ckpt


        if ckpt_path and os.path.isfile(ckpt_path):
            logger.info(f"Rank 0 正在加载检查点: {ckpt_path}")
            # 加载到 CPU
            checkpoint_content = torch.load(ckpt_path, map_location="cpu")
            checkpoint = checkpoint_content # 保存检查点内容，稍后用于加载优化器/调度器

            # --- 仅加载模型状态 ---
            if "model_state_dict" in checkpoint_content:
                try:
                    # 处理 DDP 可能添加的 'module.' 前缀
                    model_state_dict = checkpoint_content["model_state_dict"]
                    # 检查是否是 DDP 保存的 (通常不会在 EMA 保存时出现，但在非 EMA 时可能)
                    is_ddp_state = all(k.startswith('module.') for k in model_state_dict.keys())
                    if is_ddp_state:
                         # 移除 'module.' 前缀以匹配本地模型
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in model_state_dict.items():
                            name = k[7:] # remove module.
                            new_state_dict[name] = v
                        model_state_dict = new_state_dict

                    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"加载模型权重时丢失键: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"加载模型权重时出现意外键: {unexpected_keys}")
                except Exception as e:
                    logger.error(f"加载模型状态失败: {e}", exc_info=True)
            else:
                logger.warning("检查点中未找到模型状态。")


            # --- EMA 状态 (如果使用) ---
            # EMA 对象在 fabric.setup 之前初始化，可以直接加载状态
            if ema is not None and "ema_state_dict" in checkpoint_content:
                try:
                    ema.load_state_dict(checkpoint_content["ema_state_dict"])
                except Exception as e:
                    logger.error(f"加载 EMA 状态失败: {e}", exc_info=True)
            elif ema is not None:
                 logger.warning("检查点中未找到 EMA 状态，但配置了 EMA。")

            # 迭代次数和最佳指标
            first_iter = checkpoint_content.get("iteration", 0)
            best_metric = checkpoint_content.get("best_metric", 0.0)
            logger.info(f"Rank 0 成功加载检查点框架。将从迭代 {first_iter + 1} 继续，最佳指标: {best_metric:.4f}")
            logger.info("优化器和调度器状态将在 fabric.setup 后加载。")

        else:
            logger.info("Rank 0 未找到检查点，将从头开始训练。")

    # --- 广播 first_iter 和 best_metric ---
    # 将 first_iter 和 best_metric 打包成 tensor，放在 fabric 设备上进行广播
    state_list = [float(first_iter), best_metric]
    state_tensor_on_device = torch.tensor(state_list, dtype=torch.float64, device=fabric.device)
    fabric.broadcast(state_tensor_on_device, src=0)
    # 所有进程从广播的 tensor 中获取值
    first_iter = int(state_tensor_on_device[0].item())
    best_metric = state_tensor_on_device[1].item()

    # 等待所有进程完成加载/广播
    fabric.barrier()

    # Rank 0 返回检查点内容，其他 rank 返回 None
    # 注意：直接返回 checkpoint 字典可能包含大量数据，更好的做法是只提取需要的 state_dict
    # 但为了简单起见，我们先按原样返回，由调用者处理
    return checkpoint if fabric.is_global_zero else None, first_iter, best_metric


# ----------------------------
# 日志记录函数 (增加 Rank 0 判断)
# ----------------------------
def log_metrics(fabric: Fabric, loss_dict, iteration, current_lr):
    """记录损失指标，仅在 Rank 0 执行"""
    if not fabric.is_global_zero:
        return

    # 准备日志数据
    log_data = {
        "总损失": loss_dict["total_loss"], # item() 已在 train_one_epoch 中处理
        "投影损失": loss_dict["reproj_loss"],
        "SSIM损失": loss_dict.get("ssim_loss", 0.0),
        "TV损失": loss_dict.get("tv_loss", 0.0),
        "XYZ边界正则损失": loss_dict.get("xyz_boundary_reg_loss", 0.0), # 新增
        "学习率": current_lr,
    }

    # 本地日志记录
    log_msg = f"迭代 {iteration} -"
    for key, value in log_data.items():
        log_msg += f" {key}: {value:.4f}" if isinstance(value, (float, int)) else f" {key}: {value}"
    logger.info(log_msg)

    # WandB 日志记录
    if is_wandb_enabled():
        try:
            wandb.log({f"训练/{k}": v for k, v in log_data.items()}, step=iteration)
        except Exception as e:
            logger.error(f"WandB 记录指标失败: {e}")

def log_parameter_histograms(fabric: Fabric, gaussian_splats, iteration):
    """记录高斯参数的直方图，仅在 Rank 0 执行"""
    if not fabric.is_global_zero or not is_wandb_enabled():
        return

    histogram_dict = {}
    # gaussian_splats 此时是字典，值为 (B, N, D) 或 (B, N) 的张量，B 是 local batch size
    # 我们只取 Rank 0 进程上第一个样本 (idx=0) 的数据进行统计
    sample_idx = 0
    if gaussian_splats["xyz"].shape[0] <= sample_idx: # 检查 local batch size 是否足够
        logger.warning(f"Rank 0 的 local batch size ({gaussian_splats['xyz'].shape[0]}) 不足，无法记录直方图。")
        return

    try:
        # 提取 Rank 0 的第一个样本数据
        density_tensor = gaussian_splats["density"][sample_idx].contiguous().detach().cpu()
        offset_tensor = gaussian_splats["offset"][sample_idx].contiguous().detach().cpu()
        scaling_tensor = gaussian_splats["scaling"][sample_idx].contiguous().detach().cpu()
        rotation_tensor = gaussian_splats["rotation"][sample_idx].contiguous().detach().cpu()
        xyz_tensor = gaussian_splats["xyz"][sample_idx].contiguous().detach().cpu() # (N, 3)

        # --- 密度 ---
        density_values = density_tensor.flatten().numpy()
        histogram_dict["参数/密度分布"] = wandb.Histogram(density_values)
        histogram_dict["参数/密度最大值"] = float(density_values.max())
        histogram_dict["参数/密度最小值"] = float(density_values.min())
        histogram_dict["参数/密度平均值"] = float(density_values.mean())

        # --- 偏移 ---
        offset_values = offset_tensor.flatten().numpy()
        histogram_dict["参数/偏移分布"] = wandb.Histogram(offset_values)
        histogram_dict["参数/偏移最大值"] = float(offset_values.max())
        histogram_dict["参数/偏移最小值"] = float(offset_values.min())
        histogram_dict["参数/偏移平均值"] = float(offset_values.mean())

        # --- 缩放 ---
        scaling_values = scaling_tensor.flatten().numpy()
        histogram_dict["参数/缩放分布"] = wandb.Histogram(scaling_values)
        histogram_dict["参数/缩放最大值"] = float(scaling_values.max())
        histogram_dict["参数/缩放最小值"] = float(scaling_values.min())
        histogram_dict["参数/缩放平均值"] = float(scaling_values.mean())

        # --- 旋转 ---
        rotation_values = rotation_tensor.flatten().numpy()
        histogram_dict["参数/旋转分布"] = wandb.Histogram(rotation_values)
        histogram_dict["参数/旋转最大值"] = float(rotation_values.max())
        histogram_dict["参数/旋转最小值"] = float(rotation_values.min())
        histogram_dict["参数/旋转平均值"] = float(rotation_values.mean())

        # --- XYZ 坐标 ---
        x_values = xyz_tensor[:, 0].numpy()
        y_values = xyz_tensor[:, 1].numpy()
        z_values = xyz_tensor[:, 2].numpy()

        histogram_dict["坐标/X分布"] = wandb.Histogram(x_values)
        histogram_dict["坐标/Y分布"] = wandb.Histogram(y_values)
        histogram_dict["坐标/Z分布"] = wandb.Histogram(z_values)

        histogram_dict["坐标/X最大值"] = float(x_values.max())
        histogram_dict["坐标/X最小值"] = float(x_values.min())
        histogram_dict["坐标/X平均值"] = float(x_values.mean())
        histogram_dict["坐标/X范围"] = float(x_values.max() - x_values.min())

        histogram_dict["坐标/Y最大值"] = float(y_values.max())
        histogram_dict["坐标/Y最小值"] = float(y_values.min())
        histogram_dict["坐标/Y平均值"] = float(y_values.mean())
        histogram_dict["坐标/Y范围"] = float(y_values.max() - y_values.min())

        histogram_dict["坐标/Z最大值"] = float(z_values.max())
        histogram_dict["坐标/Z最小值"] = float(z_values.min())
        histogram_dict["坐标/Z平均值"] = float(z_values.mean())
        histogram_dict["坐标/Z范围"] = float(z_values.max() - z_values.min())

        # --- 记录到 WandB ---
        wandb.log(histogram_dict, step=iteration)

    except Exception as e:
        logger.error(f"处理或记录参数直方图时出错: {e}", exc_info=True)


def log_visualizations(fabric: Fabric, loss_dict_vis, iteration, cfg, log_dir):
    """记录渲染图像、GT、体数据切片等可视化结果，仅在 Rank 0 执行"""
    if not fabric.is_global_zero or not is_wandb_enabled():
        return

    log_dict = {}
    try:
        # --- 渲染图像和 GT ---
        # loss_dict_vis 包含的是 rank 0 第一个样本的渲染/gt 结果
        if loss_dict_vis.get("rendered_images") is not None:
            # 假设 rendered_images 是 (N_views, C, H, W) 或 (N_views, H, W)
            render_vis = loss_dict_vis["rendered_images"][0].squeeze().detach().cpu().numpy() # 取第一个视角
            gt_vis = loss_dict_vis["gt_images"][0].squeeze().detach().cpu().numpy() # 取第一个视角

            log_dict["渲染/视角0"] = wandb.Image(render_vis, caption="Rendered Output (View 0)")
            log_dict["渲染/真实视角0"] = wandb.Image(gt_vis, caption="Ground Truth (View 0)")

        # --- 体数据切片 ---
        if (loss_dict_vis.get("vol_pred") is not None and
            isinstance(loss_dict_vis["vol_pred"], torch.Tensor) and
            loss_dict_vis["vol_pred"].dim() >= 3): # 至少是 3D (D, H, W)

            vol_pred = loss_dict_vis["vol_pred"].squeeze() # 移除批次和通道维度 (如果存在)
            if vol_pred.dim() == 3: # 确保是 3D 张量
                mid_z = vol_pred.shape[0] // 2
                mid_y = vol_pred.shape[1] // 2
                mid_x = vol_pred.shape[2] // 2

                axial_slice = vol_pred[mid_z, :, :].detach().cpu().numpy()
                coronal_slice = vol_pred[:, mid_y, :].detach().cpu().numpy()
                sagittal_slice = vol_pred[:, :, mid_x].detach().cpu().numpy()

                # 归一化到 0-1 以便可视化
                def normalize(img):
                    min_val, max_val = img.min(), img.max()
                    return (img - min_val) / max(1e-6, max_val - min_val)

                log_dict["体数据/轴向切片(中心)"] = wandb.Image(normalize(axial_slice), caption="Axial Slice")
                log_dict["体数据/冠状切片(中心)"] = wandb.Image(normalize(coronal_slice), caption="Coronal Slice")
                log_dict["体数据/矢状切片(中心)"] = wandb.Image(normalize(sagittal_slice), caption="Sagittal Slice")

                # --- 保存体数据 NPY 文件 ---
                save_interval = getattr(cfg.logging, "volume_save_interval", 1000)
                if iteration % save_interval == 0 or iteration <= 10:
                    if log_dir: # 确保 log_dir 有效 (在 rank 0 上)
                        volumes_dir = os.path.join(log_dir, "volumes")
                        os.makedirs(volumes_dir, exist_ok=True)
                        vol_data = vol_pred.detach().cpu().numpy()
                        vol_path = os.path.join(volumes_dir, f"volume_iter_{iteration:06d}.npy")
                        try:
                            np.save(vol_path, vol_data)
                            logger.info(f"体数据已保存到: {vol_path}")
                        except Exception as e:
                            logger.error(f"保存体数据失败: {e}")
                    else:
                        logger.warning("log_dir 未设置，无法保存体数据。")
            else:
                 logger.warning(f"vol_pred 维度不为 3 (实际为 {vol_pred.dim()})，无法进行切片可视化。")

        # --- 记录到 WandB ---
        if log_dict:
            wandb.log(log_dict, step=iteration)

    except Exception as e:
        logger.error(f"WandB 可视化记录失败: {e}", exc_info=True)


def save_checkpoint(fabric: Fabric, model, optimizer, scheduler, ema, iteration, metric, log_dir, filename):
    """保存检查点，仅在 Rank 0 执行"""
    if not fabric.is_global_zero:
        return

    if not log_dir:
        logger.error("log_dir 未设置 (非 Rank 0)，无法保存检查点。")
        return

    # --- 获取需要保存的状态字典 ---
    # EMA 模型优先 (如果使用)
    if ema is not None:
        model_state_dict = ema.ema_model.state_dict()
        ema_state_dict = ema.state_dict()
    else:
        # 从 Fabric 获取解包后的模型状态
        # Fabric > 2.0 推荐方式:
        # model_state_dict = fabric.unwrap(model).state_dict()
        # 兼容旧版或直接访问 _forward_module (如果存在且为 DDP)
        unwrapped_model = model
        if hasattr(model, 'module'): # DDP 通常会包裹在 module 里
             unwrapped_model = model.module
        model_state_dict = unwrapped_model.state_dict()
        ema_state_dict = None

    # 优化器状态 (Fabric 会处理，但直接访问 state_dict 也可以)
    optimizer_state_dict = optimizer.state_dict()
    # 调度器状态
    scheduler_state_dict = scheduler.state_dict() if scheduler else None

    # --- 构建检查点字典 ---
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "best_metric": metric, # 保存当前的最佳指标
        "ema_state_dict": ema_state_dict, # 可能为 None
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- 保存文件 ---
    os.makedirs(log_dir, exist_ok=True)
    ckpt_path = os.path.join(log_dir, filename)
    temp_ckpt_path = ckpt_path + ".tmp" # 使用临时文件防止写入中断

    try:
        torch.save(checkpoint, temp_ckpt_path)
        os.replace(temp_ckpt_path, ckpt_path) # 原子替换
        logger.info(f"模型已保存到: {ckpt_path} (迭代: {iteration}, 指标: {metric:.4f})")
    except Exception as e:
        logger.error(f"保存检查点到 {ckpt_path} 失败: {e}", exc_info=True)
        if os.path.exists(temp_ckpt_path):
            os.remove(temp_ckpt_path) # 清理临时文件

# ----------------------------
# 训练循环 (修改以适应 DDP)
# ----------------------------
def train_one_epoch(fabric: Fabric, train_loader, model, optimizer, scheduler, ema, cfg: DictConfig, device, start_iter, log_dir):
    """
    单个 epoch 训练逻辑，使用 Fabric 进行梯度同步。
    返回训练结束时的迭代次数。
    """
    model.train() # 设置模型为训练模式

    accumulation_steps = getattr(cfg.opt, "accumulation_steps", 1) # 梯度累积步数
    mem_clean_interval = getattr(cfg.opt, "mem_clean_interval", 1000) # 内存清理间隔
    total_steps = cfg.opt.iterations # 总目标迭代次数
    log_loss_interval = getattr(cfg.logging, "loss_log", 100)
    log_render_interval = getattr(cfg.logging, "render_log", 500)
    log_hist_interval = getattr(cfg.logging, "histogram_log", 500)
    save_ckpt_interval = getattr(cfg.logging, "ckpt_iterations", 1000)

    # 获取混合精度设置
    mixed_precision = getattr(cfg.general, "mixed_precision", False)

    # 创建梯度缩放器，用于混合精度训练
    scaler = GradScaler(enabled=mixed_precision)

    iteration = start_iter # 当前全局迭代次数 (数据批次迭代)
    # --- 修改: 计算起始的优化器步骤数 ---
    # +1 因为 start_iter 是上一次完成的迭代，我们要从 start_iter+1 开始
    # // accumulation_steps 计算有多少个完整的累积周期已经完成
    start_optimizer_steps = (start_iter + 1) // accumulation_steps
    optimizer_steps = start_optimizer_steps # 当前实际执行的优化器步骤数
    # --- 结束修改 ---

    optimizer.zero_grad() # 在循环开始前清零一次梯度

    # 创建 tqdm 进度条 (仅 Rank 0 显示)
    pbar = None
    pbar = tqdm(total=total_steps, initial=iteration, desc=f"训练 (Rank {fabric.global_rank})",
                disable=not fabric.is_global_zero, dynamic_ncols=True, file=sys.stdout)

    # 训练循环，直到达到目标数据迭代次数
    while iteration <= total_steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            current_epoch = iteration // max(1, len(train_loader))
            train_loader.sampler.set_epoch(current_epoch)

        # 迭代数据加载器
        for batch_idx, data in enumerate(train_loader):
            if iteration > total_steps:
                break # 如果内部循环超过了总步数，则退出

            # ... [数据处理和模型前向传播代码保持不变] ...
            # 获取 local_batch_size, camera_params_list, scanner_cfg_list
            local_batch_size = data["input_images"].shape[0]
            num_input_images = cfg.data.input_images
            camera_params_list = []
            scanner_cfg_list = []
            angles = data["camera_params"]["angle"] # (local_bs, N_total)
            view_to_world = data["camera_params"]["view_to_world"] # (local_bs, N_total, 4, 4)

            for i in range(local_batch_size):
                for j in range(num_input_images):
                     if j < angles.shape[1] and j < view_to_world.shape[1]:
                        cp = {"angle": float(angles[i, j].item()),
                              "view_to_world": view_to_world[i, j]}
                        camera_params_list.append(cp)
                        scanner_cfg_list.append(data["scanner_cfg"][i])
                     else:
                         logger.error(f"索引错误: sample {i}, view {j} 超出范围 ({angles.shape[1]}, {view_to_world.shape[1]})")


            with autocast(enabled=mixed_precision):
                gaussian_splats = model(data["input_images"],
                                        data["source_cv2wT_quat"],
                                        camera_params_list,
                                        scanner_cfg_list)

                # --- 计算损失 (遍历 local_batch 内的样本) ---
                batch_total_loss = 0.0
                batch_reproj_loss = 0.0
                batch_ssim_loss = 0.0
                batch_tv_loss = 0.0
                batch_xyz_boundary_reg_loss = 0.0

                first_sample_rendered_vis = None
                first_sample_gt_vis = None
                first_sample_vol_pred_vis = None

                w_reproj = getattr(cfg.opt, "w_l12", 1.0)
                w_ssim   = getattr(cfg.opt, "w_ssim", 0.0)
                w_tv     = getattr(cfg.opt, "w_tv",   0.0)
                use_l1_loss = cfg.opt.loss == "l1"

                lambda_xyz_boundary = getattr(cfg.opt, "lambda_xyz_boundary", 0.5) 
                min_coord = getattr(cfg.opt, "min_coord", -1.0)
                max_coord = getattr(cfg.opt, "max_coord", 1.0)

                for sample_idx in range(local_batch_size):
                    gaussian_splat_sample = {k: v[sample_idx].contiguous()
                                             for k, v in gaussian_splats.items()}

                    sample_xyz_boundary_reg = 0.0
                    if lambda_xyz_boundary > 0.0 and "xyz" in gaussian_splat_sample:
                        sample_xyz_boundary_reg = xyz_boundary_regularization(
                            gaussian_splat_sample["xyz"], min_coord=min_coord,
                            max_coord=max_coord, lambda_xyz_boundary=lambda_xyz_boundary)
                    batch_xyz_boundary_reg_loss += sample_xyz_boundary_reg

                    cameras = data["cameras"][sample_idx]
                    scanner_cfg = data["scanner_cfg"][sample_idx]
                    bbox = data["bbox"][sample_idx]
                    total_cameras = len(cameras)
                    available_target_indices = list(range(num_input_images, total_cameras))
                    imgs_per_obj = getattr(cfg.opt, "imgs_per_obj", 6)

                    if len(available_target_indices) >= imgs_per_obj and imgs_per_obj > 0:
                        selected_target_indices = np.random.choice(available_target_indices, imgs_per_obj, replace=False)
                    elif len(available_target_indices) > 0: # 如果不够 imgs_per_obj 但仍有可选视图
                        selected_target_indices = available_target_indices
                    else:
                        selected_target_indices = []
                        if fabric.is_global_zero and imgs_per_obj > 0: # 只有在需要渲染视图时才警告
                            logger.warning(f"样本 {sample_idx} 在迭代 {iteration} 没有足够的目标视图 ({len(available_target_indices)}) 进行损失计算 (需要 {imgs_per_obj})。跳过此样本的渲染损失。")
                        # 如果不需要渲染视图计算损失 (e.g., imgs_per_obj=0)，则不警告

                    rendered_images_sample = []
                    gt_images_sample = []
                    vol_preds_sample = []

                    if len(selected_target_indices) > 0: # 仅当有视图需要渲染时执行
                        selected_cameras = [cameras[i] for i in selected_target_indices]
                        for cam in selected_cameras:
                            render_out = render(cam, gaussian_splat_sample)
                            rendered_image = render_out["render"]
                            rendered_images_sample.append(rendered_image.unsqueeze(0))
                            gt_image = cam.original_image.to(fabric.device).unsqueeze(0)
                            gt_images_sample.append(gt_image)

                            if w_tv > 0.0:
                                nVoxel = torch.tensor(scanner_cfg.get("nVoxel", [32, 32, 32]), device=fabric.device)
                                sVoxel = torch.tensor(scanner_cfg.get("sVoxel", [32, 32, 32]), device=fabric.device)
                                tv_vol_center = (bbox[0].to(fabric.device) + sVoxel / 2) + \
                                                (bbox[1].to(fabric.device) - sVoxel - bbox[0].to(fabric.device)) * torch.rand(3, device=fabric.device)
                                query_out = query(gaussian_splat_sample, tv_vol_center, nVoxel, sVoxel)
                                vol_pred = query_out.get("vol")
                                if vol_pred is not None:
                                     vol_preds_sample.append(vol_pred)

                    # 计算损失（即使没有渲染视图，XYZ 正则损失仍然可能存在）
                    sample_reproj = torch.tensor(0.0, device=fabric.device)
                    sample_ssim_val = torch.tensor(0.0, device=fabric.device)
                    sample_tv_val = torch.tensor(0.0, device=fabric.device)

                    if rendered_images_sample: # 只有成功渲染了图像才计算重投影和 SSIM 损失
                        rendered_images_batch = torch.cat(rendered_images_sample, dim=0)
                        gt_images_batch = torch.cat(gt_images_sample, dim=0)

                        if use_l1_loss:
                            sample_reproj = l1_loss(rendered_images_batch, gt_images_batch)
                        else:
                            sample_reproj = l2_loss(rendered_images_batch, gt_images_batch)

                        if w_ssim > 0.0:
                            sample_ssim_val = (1.0 - ssim(rendered_images_batch, gt_images_batch))

                        if sample_idx == 0 and fabric.is_global_zero:
                            first_sample_rendered_vis = rendered_images_batch.detach()
                            first_sample_gt_vis = gt_images_batch.detach()
                            if w_tv > 0.0 and vol_preds_sample:
                                first_sample_vol_pred_vis = vol_preds_sample[0].detach() # 保存第一个 vol 用于可视化

                        del rendered_images_batch, gt_images_batch

                    if w_tv > 0.0 and vol_preds_sample:
                        tv_losses = [tv_3d_loss(vol, reduction="mean") for vol in vol_preds_sample]
                        sample_tv_val = torch.stack(tv_losses).mean()
                        if sample_idx == 0 and fabric.is_global_zero and not first_sample_rendered_vis: # 如果没渲染图像，但计算了TV，也要记录vol
                            first_sample_vol_pred_vis = vol_preds_sample[0].detach()

                    batch_reproj_loss += sample_reproj
                    batch_ssim_loss += sample_ssim_val
                    batch_tv_loss += sample_tv_val

                    del rendered_images_sample, gt_images_sample, vol_preds_sample
                    if w_tv > 0.0 and 'tv_losses' in locals(): del tv_losses
                    if 'query_out' in locals(): del query_out
                    if 'vol_pred' in locals(): del vol_pred


                # --- 平均 Batch 内所有样本的损失 ---
                reproj_loss = batch_reproj_loss / local_batch_size
                ssim_loss = batch_ssim_loss / local_batch_size if w_ssim > 0 else torch.tensor(0.0, device=fabric.device)
                tv_loss = batch_tv_loss / local_batch_size if w_tv > 0 else torch.tensor(0.0, device=fabric.device)
                xyz_boundary_reg_loss = batch_xyz_boundary_reg_loss / local_batch_size if lambda_xyz_boundary > 0 else torch.tensor(0.0, device=fabric.device)

                # --- 组装总损失 ---
                total_loss = (w_reproj * reproj_loss +
                              w_ssim   * ssim_loss +
                              w_tv     * tv_loss +
                              xyz_boundary_reg_loss) # 加入正则损失

            # --- 检查 NaN/Inf ---
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                 logger.error(f"迭代 {iteration} (Rank {fabric.global_rank}) 产生 NaN/Inf 损失，跳过该优化步骤。Loss components: reproj={reproj_loss.item():.4f}, ssim={(ssim_loss.item() if isinstance(ssim_loss, torch.Tensor) else ssim_loss):.4f}, tv={(tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss):.4f}, xyz_reg={(xyz_boundary_reg_loss.item() if isinstance(xyz_boundary_reg_loss, torch.Tensor) else xyz_boundary_reg_loss):.4f}")
                 total_loss = None # 标记为无效，跳过后续步骤
                 # DDP 环境下，一个进程出错可能导致卡死，需要更健壮的处理，例如所有进程都跳过
                 # 或者直接终止训练
                 # fabric.barrier() # 可能需要同步状态

            # --- 反向传播 ---
            if total_loss is not None:
                # 将损失除以累积步数进行缩放
                scaler.scale(total_loss / accumulation_steps).backward()

            # --- 优化器步骤、日志、保存 ---
            # 条件：(当前迭代序号 + 1) 是累积步数的整数倍
            if (iteration + 1) % accumulation_steps == 0:
                if total_loss is not None: # 只有在损失有效时才执行优化步骤
                    scaler.unscale_(optimizer)
                    # fabric.clip_gradients(model, optimizer, max_norm=1.0) # 可选梯度裁剪
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad() # 在优化后清零梯度

                    # --- 修改: 增加优化器步骤计数器 ---
                    optimizer_steps += 1
                    # --- 结束修改 ---

                    if ema is not None and fabric.is_global_zero:
                        ema.update()

                    # --- 日志记录、可视化、保存 (仅 Rank 0) ---
                    if fabric.is_global_zero:
                        current_lr = optimizer.param_groups[0]['lr']
                        loss_dict_log = {
                            "total_loss": total_loss.item(), # 使用累积结束时的损失值
                            "reproj_loss": reproj_loss.item(),
                            "ssim_loss": ssim_loss.item() if isinstance(ssim_loss, torch.Tensor) else ssim_loss,
                            "tv_loss": tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss,
                            "xyz_boundary_reg_loss": xyz_boundary_reg_loss.item() if isinstance(xyz_boundary_reg_loss, torch.Tensor) else xyz_boundary_reg_loss
                        }

                        # --- 修改: 使用 optimizer_steps 进行间隔检查 ---
                        if optimizer_steps % log_loss_interval == 0:
                             # 传递 iteration 用于 WandB step 记录，保持与数据迭代一致
                             log_metrics(fabric, loss_dict_log, iteration, current_lr)

                        if optimizer_steps % log_render_interval == 0 or iteration == start_iter : # 保留 start_iter 时的可视化
                             loss_dict_vis = {
                                 "rendered_images": first_sample_rendered_vis,
                                 "gt_images": first_sample_gt_vis,
                                 "vol_pred": first_sample_vol_pred_vis
                             }
                             # 传递 iteration 用于 WandB step 记录
                             log_visualizations(fabric, loss_dict_vis, iteration, cfg, log_dir)
                             # 清理可视化数据
                             if first_sample_rendered_vis is not None: del first_sample_rendered_vis
                             if first_sample_gt_vis is not None: del first_sample_gt_vis
                             if first_sample_vol_pred_vis is not None: del first_sample_vol_pred_vis
                             first_sample_rendered_vis, first_sample_gt_vis, first_sample_vol_pred_vis = None, None, None
                             # gc.collect() # 可选的更积极清理

                        if optimizer_steps % log_hist_interval == 0:
                             # 传递 iteration 用于 WandB step 记录
                             log_parameter_histograms(fabric, gaussian_splats, iteration)

                        if optimizer_steps % save_ckpt_interval == 0:
                            # 传递 iteration 用于检查点命名和记录
                            save_checkpoint(fabric, model, optimizer, scheduler, ema,
                                            iteration, 0.0, log_dir, "model_latest.pth") # 这里的 metric 通常来自验证
                        # --- 结束修改 ---
                else:
                    # 如果 total_loss 是 None (例如 NaN/Inf)，我们仍然需要清零梯度
                    # 因为之前的 backward 可能已经计算了部分梯度
                    optimizer.zero_grad()
                    if fabric.is_global_zero:
                        logger.warning(f"迭代 {iteration}: 跳过了优化步骤，但已清零梯度。")


            # --- 更新数据迭代计数器 ---
            iteration += 1
            if fabric.is_global_zero: # 只在 Rank 0 更新 pbar
                pbar.update(1)

            # --- 内存清理 ---
            if iteration % mem_clean_interval == 0:
                # 清理当前 Batch 的变量 (移到这里确保每次迭代都清理)
                del data, gaussian_splats, total_loss, reproj_loss, ssim_loss, tv_loss, xyz_boundary_reg_loss
                if 'gaussian_splat_sample' in locals(): del gaussian_splat_sample
                # 在清理后执行 gc 和 empty_cache
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()


    if fabric.is_global_zero: pbar.close() # 关闭进度条
    # 返回最终的数据迭代次数
    return iteration - 1

# ----------------------------
# 验证函数 (假设 evaluate_model 仅在 Rank 0 运行或内部处理了 DDP)
# ----------------------------
def validate(fabric: Fabric, model, optimizer, scheduler, ema, test_loader, cfg: DictConfig, device, iteration, log_dir, current_best_metric):
    """执行验证，仅在 Rank 0 进行评估和保存"""
    if not fabric.is_global_zero:
        # 非 Rank 0 进程只需要知道最佳指标是否更新
        # 广播更新后的 best_metric
        state_tensor = torch.tensor([current_best_metric], dtype=torch.float64, device=fabric.device)
        fabric.broadcast(state_tensor, src=0)
        return state_tensor[0].item()

    logger.info(f"\n{'='*20} 开始验证 - 迭代: {iteration} {'='*20}")
    torch.cuda.empty_cache() # 清理缓存

    # 获取用于评估的模型 (EMA 优先)
    eval_model = ema.ema_model if ema is not None else model
    eval_model.eval() # 设置为评估模式

    # --- 执行评估 ---
    # 假设 evaluate_model 接受模型、数据加载器、配置和设备
    # 如果 evaluate_model 内部没有处理 DDP，确保 test_loader 未使用分布式采样器
    with torch.no_grad():
        val_metrics = evaluate_model(model=eval_model, dataloader=test_loader, cfg=cfg, device=device) # device 可能是 'cuda:0'

    # --- 记录验证结果 ---
    log_dict = {f"验证/{k}": v for k, v in val_metrics.items()}
    if is_wandb_enabled():
        try:
            wandb.log(log_dict, step=iteration)
        except Exception as e:
            logger.error(f"WandB 记录验证结果失败: {e}")

    result_msg = f"验证结果 - 迭代 {iteration}: " + ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
    logger.info(result_msg)

    # --- 保存检查点 (最新和最佳) ---
    # 使用某个关键指标来判断最佳模型，例如 PSNR 或 SSIM
    # 假设 '平均SSIM' 或 'PSNR_novel' 是关键指标
    key_metric_name = "平均SSIM" # 或者 "PSNR_novel" 等，根据 evaluate_model 的输出调整
    current_metric_val = val_metrics.get(key_metric_name, 0.0)

    # 保存最新模型 (包含当前指标)
    save_checkpoint(fabric, model, optimizer, scheduler, ema, iteration, current_metric_val, log_dir, "model_latest.pth")

    new_best_metric = current_best_metric
    # 检查是否是最佳模型
    if current_metric_val > current_best_metric:
        new_best_metric = current_metric_val
        logger.info(f"*** 新最佳模型! {key_metric_name}: {current_best_metric:.4f} -> {new_best_metric:.4f} ***")
        save_checkpoint(fabric, model, optimizer, scheduler, ema, iteration, new_best_metric, log_dir, "model_best.pth")
        # 可以选择保存最佳指标到文件，但 load_checkpoint 会从 model_best.pth 读取
    else:
        logger.info(f"当前 {key_metric_name}: {current_metric_val:.4f} (最佳: {current_best_metric:.4f})")

    eval_model.train() # 恢复训练模式
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"{'='*20} 验证完成 {'='*20}\n")

    # 广播更新后的 best_metric
    state_tensor = torch.tensor([new_best_metric], dtype=torch.float64, device=fabric.device)
    fabric.broadcast(state_tensor, src=0)

    return new_best_metric

# ----------------------------
# 主函数
# ----------------------------
@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):
    start_time = datetime.datetime.now()

    # --- 1. 初始化 Fabric, WandB(rank 0), 日志 ---
    fabric, device, log_dir, scaler = init_training(cfg)  # 接收scaler参数
    # log_dir 只在 rank 0 上有值

    if fabric.is_global_zero:
        logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"使用设备: {fabric.world_size} 个 GPU ({device})")
        # logger.info(f"完整配置:\n{OmegaConf.to_yaml(cfg)}") # 打印配置

    # --- 2. 初始化数据集和数据加载器 ---
    train_loader, test_loader = get_dataloaders(cfg, fabric)

    # --- 3. 初始化模型、优化器、调度器 ---
    # 注意：模型仍在 CPU 上
    model, optimizer, scheduler = init_model_and_opt(cfg, fabric)

    # --- 4. 初始化 EMA (如果使用) ---
    ema = None
    if getattr(cfg.opt.ema, "use", False):
        if fabric.is_global_zero:
            from ema_pytorch import EMA
            ema = EMA(
                model, # 传入原始模型
                beta=cfg.opt.ema.beta,
                update_every=cfg.opt.ema.update_every,
                update_after_step=cfg.opt.ema.update_after_step,
            )
            logger.info("已初始化 EMA (仅 Rank 0)")
        fabric.barrier()

    # --- 5. 加载检查点 (仅 Rank 0 加载模型/EMA状态，广播迭代信息) ---
    loaded_checkpoint, first_iter, best_metric = load_checkpoint(fabric, model, ema, log_dir, cfg)
    if fabric.is_global_zero:
        logger.info(f"从迭代 {first_iter + 1} 开始训练，目标迭代: {cfg.opt.iterations}")
        logger.info(f"当前最佳指标: {best_metric:.4f}")

    # --- 6. 使用 Fabric 设置模型和优化器 (处理 DDP 包装和设备移动) ---
    model, optimizer = fabric.setup(model, optimizer)
    # scheduler 通常不需要 setup

    # --- 6.5 在 setup 之后，加载优化器和调度器状态 (如果检查点存在) ---
    opt_state_dict_recv = None
    sch_state_dict_recv = None

    if fabric.is_global_zero and loaded_checkpoint is not None:
        # Rank 0: 提取状态字典
        opt_state_dict_local = loaded_checkpoint.get("optimizer_state_dict")
        sch_state_dict_local = loaded_checkpoint.get("scheduler_state_dict")
        logger.info("Rank 0 准备广播优化器和调度器状态...")
    else:
        # 其他 Rank: 初始化为 None
        opt_state_dict_local = None
        sch_state_dict_local = None

    # --- 手动广播优化器状态 ---
    if first_iter > 0: # 只有在恢复训练时才需要加载
        # 1. Rank 0 序列化
        buffer = io.BytesIO()
        if fabric.is_global_zero and opt_state_dict_local is not None:
             torch.save(opt_state_dict_local, buffer) # 使用 torch.save 更安全
        buffer_bytes = buffer.getvalue()
        buffer.close()

        # 2. 广播大小
        size_tensor = torch.tensor(len(buffer_bytes), dtype=torch.long, device=fabric.device)
        fabric.broadcast(size_tensor, src=0)
        recv_size = size_tensor.item()

        # 3. 广播字节流
        if recv_size > 0:
             byte_tensor = torch.frombuffer(buffer_bytes, dtype=torch.uint8).to(fabric.device)
             # 如果 Rank 0 没有数据 (opt_state_dict_local is None)，则创建一个空的占位符
             if not fabric.is_global_zero:
                 byte_tensor = torch.empty(recv_size, dtype=torch.uint8, device=fabric.device)
             fabric.broadcast(byte_tensor, src=0)

             # 4. 非 Rank 0 反序列化
             if not fabric.is_global_zero:
                 recv_buffer = io.BytesIO(byte_tensor.cpu().numpy().tobytes())
                 try:
                    opt_state_dict_recv = torch.load(recv_buffer, map_location="cpu") # 加载到 CPU，load_state_dict 会处理设备
                 except Exception as e:
                     logger.error(f"Rank {fabric.global_rank} 反序列化优化器状态失败: {e}", exc_info=True)
                 recv_buffer.close()
             else: # Rank 0 直接使用本地的
                 opt_state_dict_recv = opt_state_dict_local
        elif fabric.is_global_zero and opt_state_dict_local is not None:
             # Rank 0 有数据，但大小为0？这不太可能，但处理一下
             opt_state_dict_recv = opt_state_dict_local
             logger.warning("Rank 0 的优化器状态字典序列化后大小为 0?")
        # 如果 recv_size 为 0 且非 rank 0，则 opt_state_dict_recv 保持 None

        # --- 手动广播调度器状态 ---
        # (与优化器状态类似)
        buffer = io.BytesIO()
        if fabric.is_global_zero and sch_state_dict_local is not None:
             torch.save(sch_state_dict_local, buffer)
        buffer_bytes = buffer.getvalue()
        buffer.close()

        size_tensor = torch.tensor(len(buffer_bytes), dtype=torch.long, device=fabric.device)
        fabric.broadcast(size_tensor, src=0)
        recv_size = size_tensor.item()

        if recv_size > 0:
             byte_tensor = torch.frombuffer(buffer_bytes, dtype=torch.uint8).to(fabric.device)
             if not fabric.is_global_zero:
                 byte_tensor = torch.empty(recv_size, dtype=torch.uint8, device=fabric.device)
             fabric.broadcast(byte_tensor, src=0)

             if not fabric.is_global_zero:
                 recv_buffer = io.BytesIO(byte_tensor.cpu().numpy().tobytes())
                 try:
                     sch_state_dict_recv = torch.load(recv_buffer, map_location="cpu")
                 except Exception as e:
                     logger.error(f"Rank {fabric.global_rank} 反序列化调度器状态失败: {e}", exc_info=True)
                 recv_buffer.close()
             else:
                 sch_state_dict_recv = sch_state_dict_local
        elif fabric.is_global_zero and sch_state_dict_local is not None:
             sch_state_dict_recv = sch_state_dict_local
             logger.warning("Rank 0 的调度器状态字典序列化后大小为 0?")

    # 所有进程加载接收到的状态字典
    if opt_state_dict_recv is not None:
        try:
            # 此时 optimizer 已经被 fabric.setup 处理过，位于正确的设备上
            optimizer.load_state_dict(opt_state_dict_recv)
            # 只在 rank 0 记录日志，避免重复信息
            if fabric.is_global_zero: logger.info("优化器状态已成功加载到所有进程。")
        except Exception as e:
             # 记录所有进程的错误，因为加载失败可能是分布式的
             logger.error(f"Rank {fabric.global_rank} 加载优化器状态失败: {e}", exc_info=True)
    # 如果是从检查点恢复，但没有优化器状态，在 rank 0 记录警告
    elif first_iter > 0 and fabric.is_global_zero: # 确保是恢复状态且 Rank 0
        # 检查 loaded_checkpoint 是否真的没有 optimizer_state_dict
        if loaded_checkpoint is None or "optimizer_state_dict" not in loaded_checkpoint or loaded_checkpoint["optimizer_state_dict"] is None:
            logger.warning("从检查点恢复，但未找到或广播优化器状态。")
        # else: # 如果 Rank 0 有但没广播成功，前面的错误日志会记录

    if sch_state_dict_recv is not None and scheduler is not None:
        try:
            scheduler.load_state_dict(sch_state_dict_recv)
            if fabric.is_global_zero: logger.info("调度器状态已成功加载到所有进程。")
        except Exception as e:
             logger.error(f"Rank {fabric.global_rank} 加载调度器状态失败: {e}", exc_info=True)
    elif first_iter > 0 and scheduler is not None and fabric.is_global_zero: # 确保是恢复状态且 Rank 0
        if loaded_checkpoint is None or "scheduler_state_dict" not in loaded_checkpoint or loaded_checkpoint["scheduler_state_dict"] is None:
            logger.warning("从检查点恢复，但未找到或广播调度器状态。")

    # 确保所有进程都完成了状态加载或跳过
    fabric.barrier()
    if fabric.is_global_zero:
        logger.info("优化器和调度器状态加载流程完成。")


    # --- 7. 训练循环 ---
    iteration = first_iter # 从加载的迭代次数开始 (train_one_epoch 会 +1)
    training_failed = False
    try:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 训练循环由 train_one_epoch 内部管理，直到达到 cfg.opt.iterations
        # 将起始迭代设为 first_iter (表示上一次完成的迭代)
        last_iter = train_one_epoch(fabric, train_loader, model, optimizer, scheduler, ema,
                                    cfg, device, first_iter, log_dir) # 传入 first_iter
        iteration = last_iter + 1 # 更新迭代计数器

        # --- 8. 训练结束后进行最终验证 ---
        # 验证应该在所有 rank 同步后，由 rank 0 进行
        fabric.barrier()
        if fabric.is_global_zero and iteration >= cfg.opt.iterations: # 检查是否达到目标
            logger.info("训练达到目标迭代次数，执行最终验证...")
            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()
            # validate 函数内部会处理 rank 0 逻辑和广播结果
            best_metric = validate(fabric, model, optimizer, scheduler, ema,
                                   test_loader, cfg, device, iteration, log_dir, best_metric)

    except KeyboardInterrupt:
        if fabric.is_global_zero:
            logger.warning("检测到键盘中断，正在尝试保存中断模型...")
            save_checkpoint(fabric, model, optimizer, scheduler, ema, iteration, best_metric, log_dir, "model_interrupt.pth")
            logger.info("中断模型已保存。")
        training_failed = True # 标记训练未正常完成

    except Exception as e:
        logger.error(f"训练过程中发生严重错误 (Rank {fabric.global_rank}): {str(e)}", exc_info=True)
        # 在 DDP 中，一个进程出错通常需要所有进程都退出
        # 可以尝试保存中断模型
        if fabric.is_global_zero:
            logger.error("尝试保存中断模型由于错误...")
            save_checkpoint(fabric, model, optimizer, scheduler, ema, iteration, best_metric, log_dir, "model_error.pth")
        training_failed = True # 标记训练失败

    finally:
        # --- 9. 清理和总结 ---
        fabric.barrier() # 等待所有进程到达
        if fabric.is_global_zero:
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info("-" * 30)
            if training_failed:
                 logger.warning("训练未正常完成。")
            else:
                 logger.info("训练正常完成。")
            logger.info(f"最终迭代次数: {iteration - 1}/{cfg.opt.iterations}")
            logger.info(f"最终最佳指标 ({getattr(cfg.logging, 'key_metric_name', 'N/A')}): {best_metric:.4f}")
            logger.info(f"总训练时长: {duration / 3600.0:.2f} 小时 ({duration:.1f} 秒)")
            logger.info(f"日志保存在: {log_dir}")

            # 结束 WandB 运行
            if is_wandb_enabled() and wandb.run is not None:
                wandb.finish()
                logger.info("WandB 运行已结束。")

        # 清理 GPU 内存
        del model, optimizer, scheduler, train_loader, test_loader
        if ema: del ema
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    # 推荐为 DDP 设置 spawn 启动方法，以避免 CUDA 初始化问题
    # 需要放在 main() 调用之前
    try:
        # 检查是否已设置，避免重复设置引发错误
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
            print("设置多进程启动方法为: spawn")
        else:
            print(f"多进程启动方法已设置为: {multiprocessing.get_start_method()}")
    except RuntimeError as e:
        print(f"设置多进程启动方法时发生错误 (可能已设置): {e}")

    main()