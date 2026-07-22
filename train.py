# -*- coding: utf-8 -*-
# CUDA_VISIBLE_DEVICES=0,2
import os
import gc
import datetime
import math
import logging
import random  # 添加random模块
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import hydra
import wandb
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader
# from lightning.fabric import Fabric # 移除 Fabric
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
from scene.gaussian_predictor_multichannel import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset
from eval import evaluate_model # 假设 evaluate_model 兼容 DDP 或在 rank 0 运行

# ----------------------------
# 新增的正则化函数 (保持不变)
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
# 使用 Rank 区分，避免日志冲突
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
    初始化分布式环境, 设备, WandB (仅 Rank 0), 和日志。
    """
    torch.set_float32_matmul_precision('high') # 设置矩阵乘法精度

    # --- 分布式环境初始化 (使用 torch.distributed) ---
    if 'RANK' not in os.environ:
        # 单卡训练或未通过 torchrun 启动
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # Or any free port
        rank = 0
        world_size = 1
        local_rank = 0
        dist.init_process_group(backend='gloo', init_method='env://') # Gloo for single node/CPU
        print("Running in single-process mode (or launched without torchrun).")
    else:
        # 由 torchrun 启动
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Initializing distributed training: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

    # --- 设备设置 ---
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"Rank {rank} using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Rank {rank} using CPU.")

    # --- 混合精度设置 ---
    mixed_precision = getattr(cfg.general, "mixed_precision", False)
    scaler = GradScaler(enabled=mixed_precision and torch.cuda.is_available())

    # --- 日志和 WandB (仅 Rank 0) ---
    log_dir = None
    if rank == 0: # 仅在全局 Rank 0 进程上执行
        run_name = getattr(cfg.logging, "wandb_run_name", "ddp_run") # 运行名称
        log_dir = os.path.join(os.getcwd(), "logs", run_name) # 日志目录
        setup_logging(log_dir, rank) # 设置日志，传入 rank

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
    is_disabled_tensor = torch.tensor(int(os.environ.get("WANDB_DISABLED", "false").lower() == "true"), device=device)
    dist.broadcast(is_disabled_tensor, src=0)
    if is_disabled_tensor.item() == 1:
        os.environ["WANDB_DISABLED"] = "true"

    return rank, local_rank, world_size, device, log_dir, scaler # 返回分布式信息, 设备, 和日志目录

# ----------------------------
# 数据加载与 collate (保持不变，使用 DistributedSampler)
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
    folder_names  = [sample["folder_name"] for sample in batch] # 新增：收集文件夹名称

    return {
        "input_images":      input_images,
        "camera_params":     {"angle": angles, "view_to_world": view2world},
        "source_cv2wT_quat": quats,
        "scanner_cfg":       scanner_cfgs,
        "vol":               vols,
        # "vol_mask":          vols_mask,
        "scene_scale":       scene_scales,
        "bbox":              bboxes,
        "cameras":           cameras_lists,
        "folder_name":       folder_names # 新增：返回文件夹名称列表
    }



def get_dataloaders(cfg: DictConfig, rank, world_size):
    """获取训练和测试数据加载器，使用 DistributedSampler"""
    if rank == 0:
        logger.info("初始化数据集...")
    data_path = hydra.utils.to_absolute_path(str(cfg.data.data_path))
    train_dataset = get_dataset(data_path=data_path, type="train")
    test_dataset = get_dataset(data_path=data_path, type="test")
    if rank == 0:
        logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # global_batch_size 是所有 GPU 上的总批次大小
    global_batch_size = getattr(cfg.data, "batch_size", 4)
    # 每个 GPU 的批次大小
    per_gpu_batch_size = global_batch_size // world_size
    if rank == 0:
        logger.info(f"全局批次大小: {global_batch_size}, 每个 GPU 批次大小: {per_gpu_batch_size}")

    num_workers = getattr(cfg.data, "num_workers", 0)
    pin_memory = getattr(cfg.data, "pin_memory", False) and torch.cuda.is_available()

    # 简单的内存检查 (仅 Rank 0 打印信息)
    if rank == 0 and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_mem < 24:
            logger.info(f"GPU 0 内存 ({gpu_mem:.1f}GB) 可能较小，请确保全局批次大小 ({global_batch_size}) 合适。")
        if num_workers > 0 and multiprocessing.get_start_method(allow_none=True) != 'spawn':
             logger.warning("多进程数据加载 (num_workers > 0) 在 DDP 中推荐使用 'spawn' 启动方法，请检查或设置 multiprocessing.set_start_method('spawn', force=True)")

    import functools
    collate_fn_with_cfg = functools.partial(ct_collate_fn, cfg=cfg)

    # --- 使用 DistributedSampler ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # 验证集通常也需要 sampler 来确保每个 rank 得到不同的数据子集
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # 创建 DataLoader 时 batch_size 使用每个 GPU 的大小
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size, # 使用 per-GPU 批次大小
        sampler=train_sampler, # 传入 sampler
        shuffle=False, # sampler 会处理 shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_with_cfg,
        persistent_workers=num_workers > 0, # 如果使用 worker，可以持久化
        multiprocessing_context='spawn' if num_workers > 0 else None
    )

    # 验证集 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # 验证时通常用单样本或适合验证的批次
        sampler=test_sampler, # 传入 sampler
        shuffle=False,
        num_workers=0, # 验证时一般不用多 worker
        pin_memory=False,
        collate_fn=collate_fn_with_cfg
    )

    return train_loader, test_loader

# ----------------------------
# 模型与优化器 (EMA 初始化移到 main 函数中)
# ----------------------------
def init_model_and_opt(cfg: DictConfig, rank, device):
    """初始化模型和优化器，并将模型移动到指定设备"""
    if rank == 0:
        logger.info("初始化模型和优化器...")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True # 启用 cudnn 优化
            torch.backends.cudnn.enabled = True

    model = GaussianSplatPredictor(cfg)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数 (单个副本): {total_params}")

    # --- 移动模型到设备 ---
    model.to(device)
    # 内存格式 (如果适用) 和梯度初始化
    # model = model.to(memory_format=torch.channels_last) # DDP 可能不支持 channels_last
    for param in model.parameters():
        param.grad = None # 节约内存

    # --- 优化器设置 ---
    # 确保优化器参数组与当前检查点保存和恢复逻辑一致。
    param_groups = [{'params': model.network.parameters(), 'lr': cfg.opt.base_lr}] # 使用 model.network.parameters()
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
    use_fixed_lr = getattr(cfg.opt, "use_fixed_lr", True)

    def lr_lambda(current_step):
        if use_fixed_lr:
            return 1.0
        else:
            if current_step < warmup_steps:
                return 0.1 + 0.9 * (current_step / max(1, warmup_steps))
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if rank == 0:
        if use_fixed_lr:
            logger.info(f"将使用固定学习率: {cfg.opt.base_lr}")
        else:
            logger.info(f"将使用余弦退火学习率调度，最小学习率因子: {min_lr_factor}")
        if getattr(cfg.general, "mixed_precision", False) and torch.cuda.is_available():
            logger.info("已配置混合精度训练。")

    return model, optimizer, scheduler

def load_checkpoint(rank, device, model, # 模型已在设备上
                    ema, log_dir, cfg: DictConfig):
    """
    在 Rank 0 上查找检查点，加载模型/EMA权重，并广播起始迭代和最佳指标。
    返回检查点内容 (rank 0才有值) 和迭代信息。
    """
    first_iter = 0
    best_metric = 0.0
    checkpoint_content_rank0 = None # 用于存储 Rank 0 加载的检查点内容

    # 仅 Rank 0 查找和加载检查点文件
    if rank == 0:
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
        if ckpt_path is None and log_dir:
            interrupt_ckpt = os.path.join(log_dir, "model_interrupt.pth")
            if os.path.isfile(interrupt_ckpt):
                ckpt_path = interrupt_ckpt
        # 如果没有中断的，尝试加载最新的
        if ckpt_path is None and log_dir:
            latest_ckpt = os.path.join(log_dir, "model_latest.pth")
            if os.path.isfile(latest_ckpt):
                ckpt_path = latest_ckpt

        if ckpt_path and os.path.isfile(ckpt_path):
            logger.info(f"Rank 0 正在加载检查点: {ckpt_path}")
            # 加载到 CPU，稍后加载到模型/EMA
            checkpoint_content_rank0 = torch.load(ckpt_path, map_location="cpu")

            # --- 加载模型状态 ---
            # 在 DDP 包装之前加载
            if "model_state_dict" in checkpoint_content_rank0:
                try:
                    model_state_dict = checkpoint_content_rank0["model_state_dict"]
                    # 处理可能的 DDP 'module.' 前缀 (虽然保存时应该已经去掉了)
                    is_ddp_state = all(k.startswith('module.') for k in model_state_dict.keys())
                    if is_ddp_state:
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
                    logger.info("Rank 0 模型权重已加载。")
                except Exception as e:
                    logger.error(f"Rank 0 加载模型状态失败: {e}", exc_info=True)
            else:
                logger.warning("检查点中未找到模型状态。")

            # --- EMA 状态 (如果使用) ---
            # EMA 对象在 DDP 包装之前初始化，可以直接加载状态
            if ema is not None and "ema_state_dict" in checkpoint_content_rank0:
                try:
                    ema.load_state_dict(checkpoint_content_rank0["ema_state_dict"])
                    logger.info("Rank 0 EMA 状态已加载。")
                except Exception as e:
                    logger.error(f"Rank 0 加载 EMA 状态失败: {e}", exc_info=True)
            elif ema is not None:
                 logger.warning("检查点中未找到 EMA 状态，但配置了 EMA。")

            # 迭代次数和最佳指标
            first_iter = checkpoint_content_rank0.get("iteration", 0)
            best_metric = checkpoint_content_rank0.get("best_metric", 0.0)
            logger.info(f"Rank 0 成功加载检查点框架。将从迭代 {first_iter + 1} 继续，最佳指标: {best_metric:.5f}")
            logger.info("优化器和调度器状态将在 DDP setup 后加载和广播。")

        else:
            logger.info("Rank 0 未找到检查点，将从头开始训练。")

    # --- 广播 first_iter 和 best_metric ---
    state_list = [float(first_iter), best_metric]
    state_tensor_on_device = torch.tensor(state_list, dtype=torch.float64, device=device)
    dist.broadcast(state_tensor_on_device, src=0)
    # 所有进程从广播的 tensor 中获取值
    first_iter = int(state_tensor_on_device[0].item())
    best_metric = state_tensor_on_device[1].item()

    # 等待所有进程完成加载/广播
    dist.barrier()

    # Rank 0 返回检查点内容，其他 rank 返回 None
    return checkpoint_content_rank0 if rank == 0 else None, first_iter, best_metric


# ----------------------------
# 日志记录函数 (增加 Rank 0 判断)
# ----------------------------
def log_metrics(rank, loss_dict, iteration, current_lr):
    """记录损失指标，仅在 Rank 0 执行"""
    if rank != 0:
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
        # --- 修改格式化字符串 ---
        log_msg += f" {key}: {value:.5f}" if isinstance(value, (float, int)) else f" {key}: {value}"
    logger.info(log_msg)

    # WandB 日志记录
    if is_wandb_enabled():
        try:
            wandb.log({f"训练/{k}": v for k, v in log_data.items()}, step=iteration)
        except Exception as e:
            logger.error(f"WandB 记录指标失败: {e}")

def log_parameter_histograms(rank, gaussian_splats, iteration):
    """记录高斯参数的直方图，仅在 Rank 0 执行"""
    if rank != 0 or not is_wandb_enabled():
        return

    histogram_dict = {}
    # gaussian_splats 此时是字典，值为 (LocalBS, N, D)
    # 我们只取 Rank 0 进程上第一个样本 (idx=0) 的数据进行统计
    sample_idx = 0
    if gaussian_splats["xyz"].shape[0] <= sample_idx: # 检查 local batch size 是否足够
        logger.warning(f"Rank 0 的 local batch size ({gaussian_splats['xyz'].shape[0]}) 不足，无法记录直方图。")
        return

    try:
        # 提取 Rank 0 的第一个样本数据 (转移到 CPU)
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


def log_visualizations(rank, loss_dict_vis, folder_name_vis, image_name_vis, iteration, cfg, log_dir):
    """记录渲染图像、GT、体数据切片等可视化结果，仅在 Rank 0 执行"""
    if rank != 0 or not is_wandb_enabled():
        return

    log_dict = {}
    try:
        # --- 渲染图像和 GT ---
        # loss_dict_vis 包含的是 rank 0 第一个样本的渲染/gt 结果
        if loss_dict_vis.get("rendered_images") is not None and loss_dict_vis.get("gt_images") is not None:
            render_vis = loss_dict_vis["rendered_images"].squeeze().numpy() # 已经移到 CPU
            gt_vis = loss_dict_vis["gt_images"].squeeze().numpy() # 已经移到 CPU

            # --- 修改 Caption --- #
            base_caption = f"(Iter: {iteration}, View 0)"
            folder_info = f"Folder: {folder_name_vis}" if folder_name_vis else ""
            image_info = f"Target: {image_name_vis}" if image_name_vis else ""
            separator = " - " if folder_info and image_info else ""

            # 组合 caption
            full_caption = f"{base_caption} - {folder_info}{separator}{image_info}".strip(" - ")

            # --- 确保图像是 HxW 灰度图 --- #
            if render_vis.ndim == 3 and render_vis.shape[0] in [3, 4]: # 处理 3 或 4 通道
                render_vis = render_vis[0] # 取第一个通道
            if gt_vis.ndim == 3 and gt_vis.shape[0] in [3, 4]: # 处理 3 或 4 通道
                gt_vis = gt_vis[0] # 取第一个通道

            # --- 归一化图像到 [0, 1] 范围 --- #
            def normalize_img(img):
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    return (img - min_val) / (max_val - min_val)
                else:
                    return np.zeros_like(img) # 如果图像是恒定值，返回全黑

            render_vis_norm = normalize_img(render_vis)
            gt_vis_norm = normalize_img(gt_vis)

            log_dict["渲染/视角0"] = wandb.Image(render_vis_norm, caption=f"Rendered {full_caption}")
            log_dict["渲染/真实视角0"] = wandb.Image(gt_vis_norm, caption=f"GT {full_caption}")

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
                save_interval = getattr(cfg.logging, "volume_save_interval", 1e9) # 默认值改大
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


def save_checkpoint(rank, model, optimizer, scheduler, ema, iteration, metric, log_dir, filename):
    """保存检查点，仅在 Rank 0 执行"""
    if rank != 0:
        return

    if not log_dir:
        logger.error("log_dir 未设置 (非 Rank 0)，无法保存检查点。")
        return

    # --- 获取需要保存的状态字典 ---
    # EMA 模型优先 (如果使用)
    if ema is not None:
        # 保存 EMA 模型参数和 EMA 状态本身
        model_state_dict = ema.ema_model.state_dict()
        ema_state_dict = ema.state_dict()
    else:
        # 获取 DDP 解包后的模型状态
        model_to_save = model.module if isinstance(model, DDP) else model
        model_state_dict = model_to_save.state_dict()
        ema_state_dict = None

    # 优化器状态
    optimizer_state_dict = optimizer.state_dict()
    # 调度器状态
    scheduler_state_dict = scheduler.state_dict() if scheduler else None

    # --- 构建检查点字典 ---
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model_state_dict, # 已解包或来自 EMA
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
        logger.info(f"模型已保存到: {ckpt_path} (迭代: {iteration}, 指标: {metric:.5f})")
    except Exception as e:
        logger.error(f"保存检查点到 {ckpt_path} 失败: {e}", exc_info=True)
        if os.path.exists(temp_ckpt_path):
            os.remove(temp_ckpt_path) # 清理临时文件

# ----------------------------
# 训练循环 (修改以适应 DDP和梯度累积优化)
# ----------------------------
def train_one_epoch(rank, world_size, train_loader, model, optimizer, scheduler, scaler, #传入 scaler
                    ema, cfg: DictConfig, device, start_iter, log_dir):
    """
    单个 epoch 训练逻辑，使用 DDP 进行梯度同步。
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
    mixed_precision = getattr(cfg.general, "mixed_precision", False) and torch.cuda.is_available()

    iteration = start_iter # 当前全局迭代次数 (数据批次迭代)
    # 计算起始的优化器步骤数
    start_optimizer_steps = (start_iter + 1) // accumulation_steps
    optimizer_steps = start_optimizer_steps # 当前实际执行的优化器步骤数

    optimizer.zero_grad() # 在循环开始前清零一次梯度

    # 创建 tqdm 进度条 (仅 Rank 0 显示)
    pbar = None
    if rank == 0:
        pbar = tqdm(total=total_steps, initial=iteration, desc=f"训练 (Rank {rank})",
                    disable=False, dynamic_ncols=True, file=sys.stdout)

    # 训练循环，直到达到目标数据迭代次数
    while iteration < total_steps:
        print(f"设置 sampler 的 epoch iteration")
        # --- 设置 sampler 的 epoch ---
        if isinstance(train_loader.sampler, DistributedSampler):
            # 计算当前 epoch (基于迭代次数和 loader 长度)
            # 如果 loader 长度为 0 或未知，可以简单使用 iteration // accumulation_steps 作为近似
            loader_len = len(train_loader) if len(train_loader) > 0 else 1
            current_epoch = iteration // max(1, loader_len)
            train_loader.sampler.set_epoch(current_epoch)
            # if rank == 0 and iteration % loader_len == 0:
            #     logger.debug(f"Setting sampler epoch to {current_epoch} at iteration {iteration}")
        print("开始迭代数据加载器")
        # 迭代数据加载器
        for batch_idx, data in enumerate(train_loader):
            if iteration >= total_steps:
                break # 如果内部循环超过了总步数，则退出

            # --- 数据移动到设备 ---
            # 尽量只移动需要的张量
            input_images = data["input_images"].to(device)
            source_cv2wT_quat = data["source_cv2wT_quat"].to(device)
            # angles = data["camera_params"]["angle"].to(device) # 在循环内处理
            # view_to_world = data["camera_params"]["view_to_world"].to(device) # 在循环内处理
            # bbox = data["bbox"].to(device) # 在循环内处理

            local_batch_size = input_images.shape[0]
            num_input_images = cfg.data.input_images
            camera_params_list = []
            scanner_cfg_list = []
            angles_batch = data["camera_params"]["angle"] # (local_bs, N_total)
            view_to_world_batch = data["camera_params"]["view_to_world"] # (local_bs, N_total, 4, 4)

            for i in range(local_batch_size):
                for j in range(num_input_images):
                     if j < angles_batch.shape[1] and j < view_to_world_batch.shape[1]:
                        # 保持 tensor 在原始设备或 CPU，因为 model forward 可能需要
                        cp = {"angle": float(angles_batch[i, j].item()),
                              "view_to_world": view_to_world_batch[i, j]} # 保持 tensor
                        camera_params_list.append(cp)
                        scanner_cfg_list.append(data["scanner_cfg"][i]) # 这个是 dict，不需要移动
                     else:
                         # 只在 rank 0 打印错误减少冗余
                         if rank == 0:
                            logger.error(f"索引错误: sample {i}, view {j} 超出范围 ({angles_batch.shape[1]}, {view_to_world_batch.shape[1]})")


            # --- 前向传播和损失计算 (在 autocast 内) ---
            # DDP 下不需要 no_sync context manager
            with autocast(enabled=mixed_precision):
                # 前向传播 (DDP 模型会自动处理输入和输出)
                gaussian_splats = model(input_images,
                                        source_cv2wT_quat,
                                        camera_params_list, # 列表包含 tensor，应在模型内部处理设备
                                        scanner_cfg_list) # 列表包含 dict

                # --- 计算损失 (遍历 local_batch 内的样本) ---
                batch_total_loss = 0.0
                batch_reproj_loss = 0.0
                batch_ssim_loss = 0.0
                batch_tv_loss = 0.0
                batch_xyz_boundary_reg_loss = 0.0

                # 仅 Rank 0 需要的可视化数据
                first_sample_rendered_vis = None
                first_sample_gt_vis = None
                first_sample_vol_pred_vis = None
                first_sample_gt_image_name = None # 新增：用于存储图像名称

                w_reproj = getattr(cfg.opt, "w_l12", 1.0)
                w_ssim   = getattr(cfg.opt, "w_ssim", 0.0)
                w_tv     = getattr(cfg.opt, "w_tv",   0.0)
                use_l1_loss = cfg.opt.loss == "l1"

                # 根据迭代次数动态调整 lambda_xyz_boundary
                base_lambda_xyz_boundary = getattr(cfg.opt, "lambda_xyz_boundary", 0.5)
                if iteration < 10000:
                    lambda_xyz_boundary = 500.0  # 前1000次迭代使用大权重
                else:
                    lambda_xyz_boundary = base_lambda_xyz_boundary  # 后续使用配置中的权重或默认0.5

                min_coord = getattr(cfg.opt, "min_coord", -1.0)
                max_coord = getattr(cfg.opt, "max_coord", 1.0)

                for sample_idx in range(local_batch_size):
                    # 从 DDP 输出中提取单个样本的数据
                    # .contiguous() 确保内存连续，有时是必要的
                    gaussian_splat_sample = {k: v[sample_idx].contiguous()
                                             for k, v in gaussian_splats.items()}

                    sample_xyz_boundary_reg = torch.tensor(0.0, device=device)
                    if lambda_xyz_boundary > 0.0 and "xyz" in gaussian_splat_sample:
                        # 确保 xyz 在当前设备
                        xyz_sample = gaussian_splat_sample["xyz"].to(device)
                        sample_xyz_boundary_reg = xyz_boundary_regularization(
                            xyz_sample, min_coord=min_coord,
                            max_coord=max_coord, lambda_xyz_boundary=lambda_xyz_boundary)
                    batch_xyz_boundary_reg_loss += sample_xyz_boundary_reg

                    # 获取原始 cameras 列表 (非 tensor)
                    cameras = data["cameras"][sample_idx]
                    scanner_cfg = data["scanner_cfg"][sample_idx] # dict
                    bbox_sample = data["bbox"][sample_idx].to(device) # 移动到设备
                    total_cameras = len(cameras)
                    available_target_indices = list(range(num_input_images, total_cameras))
                    imgs_per_obj = getattr(cfg.opt, "imgs_per_obj", 6)

                    selected_target_indices = []
                    if len(available_target_indices) >= imgs_per_obj and imgs_per_obj > 0:
                        # 使用 numpy 在 CPU 上选择，避免 CUDA 随机状态问题
                        selected_target_indices = np.random.choice(available_target_indices, imgs_per_obj, replace=False).tolist()
                    elif len(available_target_indices) > 0:
                        selected_target_indices = available_target_indices
                    else:
                        # 仅 Rank 0 警告
                        if rank == 0 and imgs_per_obj > 0:
                            logger.warning(f"样本 {sample_idx} 在迭代 {iteration} 没有足够的目标视图 ({len(available_target_indices)}) 进行损失计算 (需要 {imgs_per_obj})。跳过此样本的渲染损失。")

                    rendered_images_sample = []
                    gt_images_sample = []
                    vol_preds_sample = []

                    if len(selected_target_indices) > 0: # 仅当有视图需要渲染时执行
                        selected_cameras = [cameras[i] for i in selected_target_indices]
                        for cam in selected_cameras:
                             # 渲染函数需要处理设备一致性
                             # 假设 render 函数内部会将 cam 的相关 tensor 移到与 gaussian_splat_sample 相同的设备
                            render_out = render(cam, gaussian_splat_sample) # gaussian_splat_sample 在 device 上
                            rendered_image = render_out["render"] # 假设输出在 device 上
                            rendered_images_sample.append(rendered_image.unsqueeze(0))
                            # 将 GT 图像移到设备
                            gt_image = cam.original_image.to(device).unsqueeze(0)
                            gt_images_sample.append(gt_image)

                            if w_tv > 0.0:
                                nVoxel = torch.tensor(scanner_cfg.get("nVoxel", [32, 32, 32]), device=device)
                                sVoxel = torch.tensor(scanner_cfg.get("sVoxel", [32, 32, 32]), device=device)
                                # 确保 bbox_sample 在 device 上
                                tv_vol_center = (bbox_sample[0] + sVoxel / 2) + \
                                                (bbox_sample[1] - sVoxel - bbox_sample[0]) * torch.rand(3, device=device)
                                # 假设 query 函数内部处理设备
                                query_out = query(gaussian_splat_sample, tv_vol_center, nVoxel, sVoxel)
                                vol_pred = query_out.get("vol") # 假设输出在 device 上
                                if vol_pred is not None:
                                     vol_preds_sample.append(vol_pred)

                    # 计算损失（所有张量应在同一设备上）
                    sample_reproj = torch.tensor(0.0, device=device)
                    sample_ssim_val = torch.tensor(0.0, device=device)
                    sample_tv_val = torch.tensor(0.0, device=device)

                    if rendered_images_sample: # 只有成功渲染了图像才计算重投影和 SSIM 损失
                        rendered_images_batch = torch.cat(rendered_images_sample, dim=0)
                        gt_images_batch = torch.cat(gt_images_sample, dim=0)

                        if use_l1_loss:
                            sample_reproj = l1_loss(rendered_images_batch, gt_images_batch)
                        else:
                            sample_reproj = l2_loss(rendered_images_batch, gt_images_batch)

                        if w_ssim > 0.0:
                            sample_ssim_val = (1.0 - ssim(rendered_images_batch, gt_images_batch))

                        # 仅 Rank 0 保存可视化数据
                        if sample_idx == 0 and rank == 0:
                            first_sample_rendered_vis = rendered_images_batch.detach() # .cpu() ? 稍后处理
                            first_sample_gt_vis = gt_images_batch.detach()
                            first_sample_gt_image_name = selected_cameras[0].image_name # 获取 GT 对应的原始相机名称

                            if w_tv > 0.0 and vol_preds_sample:
                                first_sample_vol_pred_vis = vol_preds_sample[0].detach()

                        del rendered_images_batch, gt_images_batch # 及时释放

                    if w_tv > 0.0 and vol_preds_sample:
                        tv_losses = [tv_3d_loss(vol, reduction="mean") for vol in vol_preds_sample]
                        sample_tv_val = torch.stack(tv_losses).mean()
                        # 如果没渲染图像，但计算了TV，也要记录vol (仅 Rank 0)
                        if sample_idx == 0 and rank == 0 and first_sample_rendered_vis is None:
                            first_sample_vol_pred_vis = vol_preds_sample[0].detach()

                    batch_reproj_loss += sample_reproj
                    batch_ssim_loss += sample_ssim_val
                    batch_tv_loss += sample_tv_val

                    # 清理循环变量
                    del rendered_images_sample, gt_images_sample, vol_preds_sample
                    if w_tv > 0.0 and 'tv_losses' in locals(): del tv_losses
                    if 'query_out' in locals(): del query_out
                    if 'vol_pred' in locals(): del vol_pred
                    del gaussian_splat_sample, sample_reproj, sample_ssim_val, sample_tv_val, sample_xyz_boundary_reg

                # --- 平均 Batch 内所有样本的损失 ---
                # 这些张量已经在 device 上
                reproj_loss = batch_reproj_loss / local_batch_size
                ssim_loss = batch_ssim_loss / local_batch_size if w_ssim > 0 else torch.tensor(0.0, device=device)
                tv_loss = batch_tv_loss / local_batch_size if w_tv > 0 else torch.tensor(0.0, device=device)
                xyz_boundary_reg_loss = batch_xyz_boundary_reg_loss / local_batch_size if lambda_xyz_boundary > 0 else torch.tensor(0.0, device=device)

                # --- 组装总损失 ---
                total_loss = (w_reproj * reproj_loss +
                              w_ssim   * ssim_loss +
                              w_tv     * tv_loss +
                              xyz_boundary_reg_loss) # 加入正则损失

                # --- 反向传播 ---
                # DDP 会自动同步梯度，只需在 loss 上调用 backward
                # loss 需要除以累积步数
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if rank == 0: # 只在 Rank 0 记录错误
                        logger.error(f"迭代 {iteration} (Rank {rank}) 产生 NaN/Inf 损失，跳过该优化步骤。 Loss: {total_loss.item()}")
                    # 清零梯度，防止影响下一步
                    if (iteration + 1) % accumulation_steps == 0:
                         optimizer.zero_grad()
                else:
                    # 使用 scaler 进行反向传播
                    scaler.scale(total_loss / accumulation_steps).backward()

            # --- 优化器步骤、日志、保存 ---
            # 在累积周期的最后一步执行
            if (iteration + 1) % accumulation_steps == 0:
                 # 只有在损失有效时才执行 unscale 和 step
                if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                    scaler.unscale_(optimizer) # Unscale 梯度
                    # 可以选择性地加入梯度裁剪
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer) # 优化器更新参数
                    scaler.update() # 更新 scaler 状态
                    scheduler.step() # 更新学习率
                    if ema is not None:
                        ema.update() # 更新 EMA 模型

                # 在优化器步骤后清零梯度
                optimizer.zero_grad()

                # 记录优化器步数
                optimizer_steps += 1

                # --- 日志记录、可视化、保存 (仅 Rank 0) ---
                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    # 将损失值移到 CPU 以进行记录 (.item())
                    loss_dict_log = {
                        "total_loss": total_loss.item(),
                        "reproj_loss": reproj_loss.item(),
                        "ssim_loss": ssim_loss.item() if isinstance(ssim_loss, torch.Tensor) else ssim_loss,
                        "tv_loss": tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss,
                        "xyz_boundary_reg_loss": xyz_boundary_reg_loss.item() if isinstance(xyz_boundary_reg_loss, torch.Tensor) else xyz_boundary_reg_loss
                    }

                    # 使用优化器步数计算日志间隔
                    if optimizer_steps % log_loss_interval == 0:
                        log_metrics(rank, loss_dict_log, iteration, current_lr)

                    # 可视化数据应在记录前移到 CPU
                    if optimizer_steps % log_render_interval == 0 or iteration == start_iter:
                         vis_rendered = first_sample_rendered_vis.cpu() if first_sample_rendered_vis is not None else None
                         vis_gt = first_sample_gt_vis.cpu() if first_sample_gt_vis is not None else None
                         vis_vol = first_sample_vol_pred_vis.cpu() if first_sample_vol_pred_vis is not None else None
                         # Pass the image name
                         vis_image_name = first_sample_gt_image_name # 获取图像名称
                         vis_folder_name = data["folder_name"][0] if data["folder_name"] else "N/A" # 获取文件夹名称

                         loss_dict_vis = {
                            "rendered_images": vis_rendered,
                            "gt_images": vis_gt,
                            "vol_pred": vis_vol
                         }
                         # Pass image name and folder name to log_visualizations
                         log_visualizations(rank, loss_dict_vis, vis_folder_name, vis_image_name, iteration, cfg, log_dir) # 传递文件夹和图像名称
                         # 清理可视化数据 (CPU 上的)
                         del vis_rendered, vis_gt, vis_vol, loss_dict_vis, vis_image_name, vis_folder_name # 清理名称

                    if optimizer_steps % log_hist_interval == 0:
                        # log_parameter_histograms 会将数据移到 CPU
                        log_parameter_histograms(rank, gaussian_splats, iteration)

                    if optimizer_steps % save_ckpt_interval == 0:
                        # save_checkpoint 只在 rank 0 执行
                        save_checkpoint(rank, model, optimizer, scheduler, ema,
                                        iteration, 0.0, log_dir, "model_latest.pth") # 指标后面会在 validate 更新

                    # 清理 Rank 0 独有的可视化变量
                    if first_sample_rendered_vis is not None: del first_sample_rendered_vis
                    if first_sample_gt_vis is not None: del first_sample_gt_vis
                    if first_sample_vol_pred_vis is not None: del first_sample_vol_pred_vis
                    if first_sample_gt_image_name is not None: del first_sample_gt_image_name # 清理名称
                    # 清理文件夹名称变量 (虽然它不是在这里定义的，但保持一致性)
                    first_sample_rendered_vis, first_sample_gt_vis, first_sample_vol_pred_vis, first_sample_gt_image_name = None, None, None, None

            # --- 更新数据迭代计数器 ---
            iteration += 1
            if rank == 0: # 只在 Rank 0 更新 pbar
                pbar.update(1)

            # --- 内存清理 ---
            if iteration % mem_clean_interval == 0:
                # 清理当前 Batch 的变量
                del data, input_images, source_cv2wT_quat, gaussian_splats, total_loss, reproj_loss, ssim_loss, tv_loss, xyz_boundary_reg_loss
                # if 'gaussian_splat_sample' in locals(): del gaussian_splat_sample # 已在循环内清理
                # 在内存清理部分完成循环清理
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    if rank == 0:
        pbar.close() # 关闭进度条

    # 等待所有进程完成 epoch
    dist.barrier()
    return iteration # 返回完成的总迭代次数

# ----------------------------
# 验证函数 (仅在 Rank 0 运行评估，然后广播结果)
# ----------------------------
def validate(rank, world_size, model, optimizer, scheduler, ema, test_loader, cfg: DictConfig, device, iteration, log_dir, current_best_metric):
    """执行验证，仅在 Rank 0 进行评估和保存，然后广播最佳指标"""

    new_best_metric = current_best_metric

    if rank == 0:
        logger.info(f"\n{'='*20} 开始验证 - 迭代: {iteration} {'='*20}")
        if torch.cuda.is_available(): torch.cuda.empty_cache() # 清理缓存

        # 获取用于评估的模型 (EMA 优先)
        if ema is not None:
            eval_model = ema.ema_model # EMA 模型已经在正确的设备上
        else:
             # 从 DDP 模型中获取原始模型
             eval_model = model.module if isinstance(model, DDP) else model
        eval_model.eval() # 设置为评估模式

        # --- 执行评估 ---
        # test_loader 使用了 DistributedSampler，每个 rank 只评估一部分数据
        # 要获得完整的验证结果，通常需要在 rank 0 收集所有结果或使用非分布式 sampler
        # 简化：假设 evaluate_model 能处理这种情况或只在 rank 0 上评估完整数据集
        # (更健壮的做法是使用 dist.gather 收集指标)
        # 这里我们假设 evaluate_model 返回 rank 0 上的指标字典
        val_metrics = {}
        with torch.no_grad():
            # 确保传入 device 给 evaluate_model
            # 如果 test_loader 是分布式的，evaluate_model 需要能正确处理
            # 或者修改 test_loader 不使用 sampler 仅在 rank 0 运行
            # 暂时假设 evaluate_model 只在 rank 0 运行或返回 rank 0 指标
            try:
                 # 传递 device='cuda:0' 或 'cpu'
                 val_metrics = evaluate_model(model=eval_model, dataloader=test_loader, cfg=cfg, device=device)
            except Exception as e:
                 logger.error(f"Rank 0 评估失败: {e}", exc_info=True)


        # --- 记录验证结果 ---
        if val_metrics:
            log_dict = {f"验证/{k}": v for k, v in val_metrics.items()}
            if is_wandb_enabled():
                try:
                    wandb.log(log_dict, step=iteration)
                except Exception as e:
                    logger.error(f"WandB 记录验证结果失败: {e}")

            # --- 修改格式化字符串 ---
            result_msg = f"验证结果 - 迭代 {iteration}: " + ", ".join([f"{k}={v:.5f}" for k, v in val_metrics.items()])
            logger.info(result_msg)

            # --- 保存检查点 (最新和最佳) ---
            key_metric_name = "平均SSIM" # 或者 "PSNR_novel" 等，根据 evaluate_model 的输出调整
            current_metric_val = val_metrics.get(key_metric_name, 0.0)

            # 保存最新模型 (包含当前指标) - save_checkpoint 只在 rank 0 执行
            save_checkpoint(rank, model, optimizer, scheduler, ema, iteration, current_metric_val, log_dir, "model_latest.pth")

            # 检查是否是最佳模型
            if current_metric_val > current_best_metric:
                new_best_metric = current_metric_val
                # --- 修改格式化字符串 ---
                logger.info(f"*** 新最佳模型! {key_metric_name}: {current_best_metric:.5f} -> {new_best_metric:.5f} ***")
                save_checkpoint(rank, model, optimizer, scheduler, ema, iteration, new_best_metric, log_dir, "model_best.pth")
            else:
                # --- 修改格式化字符串 ---
                logger.info(f"当前 {key_metric_name}: {current_metric_val:.5f} (最佳: {current_best_metric:.5f})")
        else:
             logger.warning("Rank 0 未能获取验证指标，无法判断最佳模型或保存。")


        eval_model.train() # 恢复训练模式
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"{'='*20} 验证完成 {'='*20}\n")

    # --- 广播更新后的 best_metric ---
    # 将 new_best_metric (rank 0 上的值) 广播给所有进程
    best_metric_tensor = torch.tensor([new_best_metric], dtype=torch.float64, device=device)
    dist.broadcast(best_metric_tensor, src=0)
    # 所有进程更新 best_metric
    best_metric_synced = best_metric_tensor[0].item()

    # 等待所有进程完成验证步骤（包括广播）
    dist.barrier()

    return best_metric_synced

# ----------------------------
# 主函数
# ----------------------------
@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):
    # ----- 统一随机种子 (关键改动 #1) -----
    SEED = getattr(cfg, "seed", 42)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    start_time = datetime.datetime.now()

    # --- 1. 初始化分布式环境, WandB(rank 0), 日志 ---
    rank, local_rank, world_size, device, log_dir, scaler = init_training(cfg)
    logger.info(f"Rank {rank} Completed init_training")

    if rank == 0:
        logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"使用设备: {world_size} 个 GPU ({device})")
        logger.info(f"随机种子已设定为 {SEED}")

    # --- 2. 初始化数据集和数据加载器 ---
    logger.info(f"Rank {rank} Starting get_dataloaders...")
    train_loader, test_loader = get_dataloaders(cfg, rank, world_size)
    logger.info(f"Rank {rank} Completed get_dataloaders")

    # --- 3. 初始化模型、优化器、调度器 (模型已移到 device) ---
    logger.info(f"Rank {rank} Starting init_model_and_opt...")
    model, optimizer, scheduler = init_model_and_opt(cfg, rank, device)
    logger.info(f"Rank {rank} Completed init_model_and_opt")

    # --- 4. 初始化 EMA (如果使用) ---
    # 在模型加载权重之前但在移动到设备之后
    logger.info(f"Rank {rank} Initializing EMA (if enabled)...")
    ema = None
    if getattr(cfg.opt.ema, "use", False):
        from ema_pytorch import EMA
        ema = EMA(
            model, # 传入已在 device 上的模型
            beta=cfg.opt.ema.beta,
            update_every=cfg.opt.ema.update_every,
            update_after_step=cfg.opt.ema.update_after_step,
            use_ema_parameter_kwargs = {'device': device} # 确保 EMA 状态在正确设备
        ).to(device) # 确保 EMA 对象本身在设备上
        logger.info(f"Rank {rank} EMA Initialized on device {device}.")
    else:
        logger.info(f"Rank {rank} EMA Disabled.")

    # 同步点：确保模型和 EMA (如果使用) 都已初始化
    dist.barrier()

    # --- 5. 加载检查点 (模型/EMA 权重在 Rank 0 加载，信息广播) ---
    logger.info(f"Rank {rank} Starting load_checkpoint...")
    # model 和 ema 已在正确设备上，load_checkpoint 会在 rank 0 加载 state_dict
    loaded_checkpoint_rank0, first_iter, best_metric = load_checkpoint(rank, device, model, ema, log_dir, cfg)
    logger.info(f"Rank {rank} Completed load_checkpoint. first_iter={first_iter}, best_metric={best_metric:.5f}")

    if rank == 0:
        logger.info(f"从迭代 {first_iter + 1} 开始训练，目标迭代: {cfg.opt.iterations}")
        logger.info(f"当前最佳指标: {best_metric:.5f}")

    # 同步点：确保所有 rank 的模型权重一致 (rank 0 加载，其他 rank 保持初始化)
    # DDP 会在初始化时同步模型参数，所以这里理论上不需要手动广播模型权重
    dist.barrier()
    logger.info(f"Rank {rank} passed model weight sync barrier.")

    # --- 6. 使用 DDP 包装模型 ---
    # 在加载模型权重之后，加载优化器状态之前
    logger.info(f"Rank {rank} BEFORE DDP wrapping...")
    # find_unused_parameters=True 可能需要，如果模型有部分参数未在 forward 中使用
    find_unused = getattr(cfg.model, "ddp_find_unused_parameters", True) # Default to True
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused)
    logger.info(f"Rank {rank} AFTER DDP wrapping. find_unused_parameters={find_unused}")

    # --- 7. 加载优化器和调度器状态 ---
    # Rank 0 从检查点获取状态，然后广播给所有进程
    logger.info(f"Rank {rank} entering optimizer/scheduler state loading...")
    if first_iter > 0: # 只有在恢复训练时才需要加载
        opt_state_list = [None]
        sch_state_list = [None]

        if rank == 0:
            logger.info("Rank 0 preparing optimizer/scheduler state for broadcast...")
            if loaded_checkpoint_rank0:
                opt_state_list[0] = loaded_checkpoint_rank0.get("optimizer_state_dict")
                sch_state_list[0] = loaded_checkpoint_rank0.get("scheduler_state_dict")
                if opt_state_list[0] is None: logger.warning("Optimizer state not found in checkpoint.")
                if sch_state_list[0] is None: logger.warning("Scheduler state not found in checkpoint.")
            else:
                # 这个情况理论上不应该发生，因为 loaded_checkpoint_rank0 只在 rank 0 有值
                logger.error("Rank 0: Resuming requested but no checkpoint content available.")

        # 使用 broadcast_object_list 进行广播
        logger.info(f"Rank {rank} broadcasting optimizer state...")
        dist.broadcast_object_list(opt_state_list, src=0)
        logger.info(f"Rank {rank} broadcasting scheduler state...")
        dist.broadcast_object_list(sch_state_list, src=0)
        logger.info(f"Rank {rank} finished broadcasting states.")

        opt_state_dict_recv = opt_state_list[0]
        sch_state_dict_recv = sch_state_list[0]

        # 所有进程加载接收到的状态字典
        if opt_state_dict_recv:
            try:
                optimizer.load_state_dict(opt_state_dict_recv)
                # logger.info(f"Rank {rank} loaded optimizer state.") # 可能太啰嗦
                if rank == 0: logger.info("Optimizer state loaded by all ranks.")
            except ValueError as e:
                # 捕获参数组不匹配的错误
                if "parameter group that doesn't match" in str(e):
                    logger.warning(f"Rank {rank} failed to load optimizer state due to parameter group mismatch. Optimizer state will be reset. Error: {e}")
                else:
                    # 其他 ValueError 仍然报错
                    logger.error(f"Rank {rank} failed to load optimizer state with unexpected ValueError: {e}", exc_info=True)
                    raise e
            except Exception as e:
                # 其他所有加载错误
                logger.error(f"Rank {rank} failed to load optimizer state: {e}", exc_info=True)
                # 根据需要决定是否在这里重新抛出错误或允许继续
                # raise e # 如果希望加载失败时停止训练
        elif rank == 0: # 只在 Rank 0 记录未加载的警告
             logger.warning("Optimizer state was not loaded (not found or broadcast failed).")

        # if sch_state_dict_recv and scheduler:
        #     try:
        #         scheduler.load_state_dict(sch_state_dict_recv)
        #         if rank == 0: logger.info("Scheduler state loaded by all ranks.")
        #     except Exception as e:
        #          logger.error(f"Rank {rank} failed to load scheduler state: {e}", exc_info=True)
        # elif rank == 0 and scheduler: # 只在 Rank 0 记录未加载的警告
        #     logger.warning("Scheduler state was not loaded (not found or broadcast failed).")
        if rank == 0 and scheduler: # 可以加一句日志，说明没有加载旧状态
            logger.info("Scheduler state from checkpoint was intentionally skipped to use the new config.")

        # 清理 Rank 0 的检查点内容
        if rank == 0:
            del loaded_checkpoint_rank0
        del opt_state_list, sch_state_list, opt_state_dict_recv, sch_state_dict_recv

    # 同步点：确保所有进程都尝试加载了状态
    dist.barrier()
    if rank == 0:
        logger.info("Optimizer and scheduler state loading/broadcast finished. Scheduler state will use new config.")

    # --- 8. 训练循环 ---
    iteration = first_iter # 从加载的迭代次数开始
    training_failed = False
    try:
        # 清理内存
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 训练循环直到达到 cfg.opt.iterations
        last_iter = train_one_epoch(rank, world_size, train_loader, model, optimizer, scheduler, scaler,
                                    ema, cfg, device, first_iter, log_dir)
        iteration = last_iter # train_one_epoch 返回的是完成的迭代次数

        # --- 9. 训练结束后进行最终验证 ---
        dist.barrier() # 确保所有 rank 完成训练
        if rank == 0 and iteration >= cfg.opt.iterations: # 检查是否达到目标
            logger.info("训练达到目标迭代次数，执行最终验证...")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # validate 函数内部会处理 rank 0 逻辑和广播结果
            best_metric = validate(rank, world_size, model, optimizer, scheduler, ema,
                                   test_loader, cfg, device, iteration, log_dir, best_metric)

    except KeyboardInterrupt:
        if rank == 0:
            logger.warning("检测到键盘中断，正在尝试保存中断模型...")
            save_checkpoint(rank, model, optimizer, scheduler, ema, iteration, best_metric, log_dir, "model_interrupt.pth")
            logger.info("中断模型已保存。")
        training_failed = True # 标记训练未正常完成

    except Exception as e:
        logger.error(f"训练过程中发生严重错误 (Rank {rank}): {str(e)}", exc_info=True)
        # 尝试保存错误模型
        if rank == 0:
            logger.error("尝试保存错误模型...")
            try:
                save_checkpoint(rank, model, optimizer, scheduler, ema, iteration, best_metric, log_dir, "model_error.pth")
                logger.info("错误模型已尝试保存。")
            except Exception as save_e:
                 logger.error(f"保存错误模型失败: {save_e}")
        training_failed = True # 标记训练失败

    finally:
        # --- 10. 清理和总结 ---
        dist.barrier() # 等待所有进程到达
        if rank == 0:
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info("-" * 30)
            if training_failed:
                 logger.warning("训练未正常完成。")
            else:
                 logger.info("训练正常完成。")
            logger.info(f"最终迭代次数: {iteration}/{cfg.opt.iterations}") # iteration 是最后完成的步骤
            logger.info(f"最终最佳指标 ({getattr(cfg.logging, 'key_metric_name', 'N/A')}): {best_metric:.5f}")
            logger.info(f"总训练时长: {duration / 3600.0:.2f} 小时 ({duration:.1f} 秒)")
            if log_dir: logger.info(f"日志保存在: {log_dir}")

            # 结束 WandB 运行
            if is_wandb_enabled() and wandb.run is not None:
                wandb.finish()
                logger.info("WandB 运行已结束。")

        # 清理 GPU 内存和分布式环境
        del model, optimizer, scheduler, train_loader, test_loader
        if ema: del ema
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        dist.destroy_process_group()
        logger.info(f"Rank {rank} finished cleanup and destroyed process group.")


if __name__ == "__main__":
    # 检查是否需要设置 spawn (通常由 torchrun 处理，但保险起见)
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("设置多进程启动方法为: spawn")
        except RuntimeError as e:
            print(f"设置多进程启动方法时发生错误 (可能已设置或环境不支持): {e}")

    main()
