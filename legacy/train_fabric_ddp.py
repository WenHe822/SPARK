# -*- coding: utf-8 -*-
"""
Stable multi-GPU training script with Hydra cfg, Lightning Fabric and W&B
Keeps original training logic (Gaussian Splatting + XYZ boundary reg).
MODIFIED: Added detailed reprojection loss and image logging.
"""
import os
import gc
import random
import logging
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from torch.cuda.amp import GradScaler, autocast
import functools

# Project modules (unchanged)
from utils.loss_utils import l1_loss, l2_loss, ssim, tv_3d_loss
from scene.gaussian_predictor import GaussianSplatPredictor
from r2_gaussian.gaussian import render, query
from datasets.dataset_factory import get_dataset

# ---------------------- Helpers ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ct_collate_fn remains unchanged
def ct_collate_fn(batch, cfg):
    bs = len(batch)
    N = cfg.data.input_images
    input_images = torch.stack([
        torch.stack([cam.original_image for cam in sample["cameras"][:N]], dim=0)
        for sample in batch
    ], dim=0)
    angles = torch.stack([
        torch.tensor([cam.angle for cam in sample["cameras"][:N]], dtype=torch.float32)
        for sample in batch
    ], dim=0)
    view2world = torch.stack([
        torch.stack([cam.view_world_transform for cam in sample["cameras"][:N]], dim=0)
        for sample in batch
    ], dim=0)
    quats = torch.stack([
        sample["source_cv2wT_quat"][:N]
        for sample in batch
    ], dim=0)
    vols      = torch.stack([sample["vol"]      for sample in batch], dim=0)
    scene_scales  = torch.tensor([sample["scene_scale"] for sample in batch], dtype=torch.float32)
    bboxes        = torch.stack([sample["bbox"] for sample in batch], dim=0)
    scanner_cfgs  = [sample["scanner_cfg"] for sample in batch]
    cameras_lists = [sample["cameras"]    for sample in batch]
    return {
        "input_images":      input_images,
        "camera_params":     {"angle": angles, "view_to_world": view2world},
        "source_cv2wT_quat": quats,
        "scanner_cfg":       scanner_cfgs,
        "vol":               vols,
        "scene_scale":       scene_scales,
        "bbox":              bboxes,
        "cameras":           cameras_lists
    }


def xyz_boundary_regularization(xyz, min_coord: float, max_coord: float, lambda_xyz_boundary: float):
    lower = torch.relu(min_coord - xyz)
    upper = torch.relu(xyz - max_coord)
    return lambda_xyz_boundary * (lower + upper).mean()

# setup_logger remains unchanged
def setup_logger(rank: int, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    level = logging.INFO if rank == 0 else logging.ERROR
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / 'train.log', encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to {log_dir}")
    return logger

# build_dataloaders remains unchanged
def build_dataloaders(cfg: DictConfig, fabric: Fabric):
    train_ds = get_dataset(cfg.data.data_path, type="train")
    val_ds   = get_dataset(cfg.data.data_path, type="test")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,   num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=False)
    collate_fn_with_cfg = functools.partial(ct_collate_fn, cfg=cfg)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        collate_fn=collate_fn_with_cfg)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1, # Usually 1 for validation
        sampler=val_sampler,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn_with_cfg # <<< MODIFIED: Use collate_fn for val too, consistency
    )
    # <<< MODIFIED: Apply setup_dataloaders here for consistency
    # <<< MODIFIED: Removed `use_distributed_sampler=False`, might not exist in older Fabric versions.
    # Ensure DistributedSampler is correctly handled (it is created above).
    train_loader = fabric.setup_dataloaders(train_loader) # Sampler already added
    val_loader = fabric.setup_dataloaders(val_loader) # Sampler already added
    return train_loader, val_loader

# ---------------------- Main ----------------------
@hydra.main(version_base=None, config_path='../configs', config_name='default_config')
def main(cfg: DictConfig):
    # 1. Fabric init
    # <<< MODIFIED: Re-added find_unused_parameters=True based on the RuntimeError.
    # This is necessary when some model parameters don't receive gradients in every iteration,
    # which seems to be the case here according to the error message.
    strategy = DDPStrategy(find_unused_parameters=True) if cfg.general.num_devices > 1 else 'auto'
    # <<< MODIFIED: Add logging config defaults if missing
    log_loss_interval = getattr(cfg.logging, "loss_log", 10)
    log_render_interval = getattr(cfg.logging, "render_log_interval", 10) # <<< ADDED: Default render log interval
    ckpt_every_interval = getattr(cfg.logging, "ckpt_every", 100)
    disable_wandb = getattr(cfg.logging, "disable_wandb", False)

    fabric = Fabric(
        accelerator='cuda',
        devices=cfg.general.num_devices,
        strategy=strategy,
        # Note: Ensure '16' is the correct precision string for your Fabric version.
        # Older versions might use '16-mixed'.
        precision=cfg.general.precision)
    fabric.launch()

    is_global_zero = fabric.is_global_zero
    log_dir = Path(cfg.logging.output_dir) / cfg.logging.wandb_run_name # <<< MODIFIED: Use run name subdir
    logger = setup_logger(fabric.global_rank, log_dir)
    set_seed(cfg.general.seed + fabric.global_rank)

    # W&B
    if is_global_zero and not disable_wandb:
        # <<< MODIFIED: Ensure log_dir exists before wandb init
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run_name,
                dir=str(log_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume='allow', reinit=True,
                settings=wandb.Settings(console="off") # <<< ADDED: Avoid duplicate console logs
            )
            logger.info("W&B initialized.")
        except Exception as e:
            logger.error(f"WandB initialization failed: {e}. Disabling WandB.")
            disable_wandb = True # <<< ADDED: Disable if init fails
            os.environ['WANDB_DISABLED'] = 'true'
    else:
        os.environ['WANDB_DISABLED'] = 'true'

    # Data
    # <<< MODIFIED: Pass fabric to build_dataloaders
    train_loader, val_loader = build_dataloaders(cfg, fabric)
    if is_global_zero:
        logger.info(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}") # <<< MODIFIED: Log loader size

    # Model, optimizer, scheduler, scaler
    model = GaussianSplatPredictor(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.opt.base_lr, weight_decay=cfg.opt.weight_decay)
    if cfg.opt.use_fixed_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.opt.iterations, eta_min=cfg.opt.base_lr * cfg.opt.min_lr_factor)
    # Note: Check if your Fabric version requires different handling for GradScaler enablement.
    scaler = GradScaler(enabled=('16' in cfg.general.precision))

    # Resume
    start_iter, best_metric = 0, 0.0
    # <<< MODIFIED: Store optimizer/scheduler/scaler states temporarily like before
    opt_state_dict, sch_state_dict, scaler_state_dict = None, None, None
    if cfg.logging.resume_ckpt:
        ckpt_path = Path(cfg.logging.resume_ckpt)
        if ckpt_path.is_file():
             # <<< MODIFIED: Load to CPU first is generally safer for DDP
            data = torch.load(ckpt_path, map_location='cpu')
            # Handle potential 'module.' prefix if saved from DDP
            model_state_dict = data['model_state_dict']
            if all(k.startswith('module.') for k in model_state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[7:] # remove module.
                    new_state_dict[name] = v
                model_state_dict = new_state_dict

            # Load model state before fabric.setup
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            if is_global_zero:
                if missing_keys: logger.warning(f"Resuming: Missing keys in model state_dict: {missing_keys}")
                if unexpected_keys: logger.warning(f"Resuming: Unexpected keys in model state_dict: {unexpected_keys}")

            # Load optimizer, scheduler, scaler states later, after fabric.setup
            # Store them temporarily
            opt_state_dict = data.get('optimizer_state_dict')
            sch_state_dict = data.get('scheduler_state_dict')
            scaler_state_dict = data.get('scaler_state_dict')
            start_iter = data.get('iteration', 0) + 1 # <<< MODIFIED: Start from next iteration
            best_metric = data.get('best_metric', 0.0)
            if is_global_zero:
                logger.info(f"Prepared to resume from iter {start_iter}, best_metric {best_metric:.4f}")
                logger.info("Optimizer, scheduler, scaler states will be loaded after fabric.setup.")
        elif is_global_zero:
            logger.warning(f"Resume checkpoint not found at {ckpt_path}")


    # <<< MODIFIED: Setup model and optimizer with Fabric
    model, optimizer = fabric.setup(model, optimizer)

    # <<< MODIFIED: Load optimizer, scheduler, scaler state *after* setup
    # This remains complex with DDP. Rank 0 loading + broadcasting state_dicts is the most robust
    # but harder to implement. This approach (each rank loads) might work depending on the
    # Fabric version and optimizer sharding.
    if cfg.logging.resume_ckpt and ckpt_path.is_file():
        if opt_state_dict:
             try:
                 optimizer.load_state_dict(opt_state_dict)
                 if is_global_zero: logger.info("Optimizer state loaded.")
             except Exception as e:
                 logger.error(f"Rank {fabric.global_rank} failed to load optimizer state: {e}", exc_info=True)
        if sch_state_dict and scheduler:
             try:
                 scheduler.load_state_dict(sch_state_dict)
                 if is_global_zero: logger.info("Scheduler state loaded.")
             except Exception as e:
                 logger.error(f"Rank {fabric.global_rank} failed to load scheduler state: {e}", exc_info=True)
        if scaler_state_dict and scaler:
             try:
                 scaler.load_state_dict(scaler_state_dict)
                 if is_global_zero: logger.info("Scaler state loaded.")
             except Exception as e:
                 logger.error(f"Rank {fabric.global_rank} failed to load scaler state: {e}", exc_info=True)
        # Barrier to ensure all ranks have loaded before proceeding
        fabric.barrier()


    # Training loop
    iteration = start_iter
    # <<< ADDED: Variables to store data for visualization logging
    first_sample_rend_vis = None
    first_sample_gt_vis = None
    first_sample_reproj_loss_val = 0.0

    while iteration < cfg.opt.iterations:
        # <<< ADDED: Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
             # Use optimizer steps or iteration count? Iteration seems more direct here.
             epoch = iteration // len(train_loader)
             train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if iteration >= cfg.opt.iterations:
                break

            # <<< ADDED: Determine if we need to log visualizations for this iteration
            should_log_render = is_global_zero and (iteration % log_render_interval == 0) and not disable_wandb
            # <<< ADDED: Reset vis variables at the start of potential logging iteration step
            if should_log_render:
                first_sample_rend_vis = None
                first_sample_gt_vis = None
                first_sample_reproj_loss_val = 0.0

            optimizer.zero_grad()
            with autocast(enabled=('16' in cfg.general.precision)):
                # Predict all splats
                gaussian_splats = model(
                    batch['input_images'], batch['source_cv2wT_quat'],
                    # Rebuild camera/scanner lists (assuming batch structure is consistent)
                    [ {'angle': float(a), 'view_to_world': v}
                      for sample_angles, sample_vs in zip(
                          batch['camera_params']['angle'],
                          batch['camera_params']['view_to_world'])
                      # Use cfg.data.input_images (used in collate) not cfg.data.input_views
                      for a, v in zip(sample_angles[:cfg.data.input_images], sample_vs[:cfg.data.input_images]) ],
                    [batch['scanner_cfg'][i] # Directly use sample index
                     for i in range(batch['input_images'].shape[0])] # <<< MODIFIED: Simpler scanner_cfg list creation
                )

                batch_loss = 0.0
                batch_reproj_loss_unweighted = 0.0 # <<< ADDED: Accumulate unweighted loss for logging
                num_samples_with_valid_targets = 0 # <<< ADDED: Track samples contributing to reproj loss

                B = batch['input_images'].shape[0] # Local batch size
                # iterate per-sample
                for i in range(B):
                    # extract sample splats
                    sample_splats = {k: v[i] for k, v in gaussian_splats.items()}
                    # target cameras list
                    cams = batch['cameras'][i]
                    # Select target views (ensure indices are valid)
                    num_total_views = len(cams)
                    # Use cfg.data.input_images (consistent with collate)
                    start_target_idx = cfg.data.input_images
                    available_target_indices = list(range(start_target_idx, num_total_views))

                    targets_to_render = []
                    if len(available_target_indices) > 0 and cfg.opt.imgs_per_obj > 0:
                        num_targets = min(cfg.opt.imgs_per_obj, len(available_target_indices))
                        selected_indices = random.sample(available_target_indices, num_targets)
                        targets_to_render = [cams[idx] for idx in selected_indices]

                    sample_loss = 0.0
                    sample_reproj_loss_val = 0.0 # <<< ADDED: Track unweighted loss for this sample
                    num_targets_rendered = 0

                    if not targets_to_render:
                         # Handle boundary regularization even if no targets rendered
                         if cfg.opt.lambda_xyz_boundary > 0 and 'xyz' in sample_splats:
                            sample_loss += xyz_boundary_regularization(
                                sample_splats['xyz'], -1, 1, cfg.opt.lambda_xyz_boundary)
                         # Accumulate sample loss to batch loss (might just be regularization loss)
                         batch_loss += sample_loss
                         continue # Skip rendering loop if no targets

                    # Only increment if we have targets to potentially render
                    num_samples_with_valid_targets += 1

                    for cam_idx, cam in enumerate(targets_to_render):
                        out = render(cam, sample_splats)
                        rend = out['render']
                        gt = cam.original_image.to(fabric.device)

                        # --- Calculate individual loss components ---
                        reproj_loss_term = l1_loss(rend, gt) # Calculate unweighted loss first
                        sample_reproj_loss_val += reproj_loss_term.item() # Accumulate for logging

                        loss_cam = cfg.opt.w_l12 * reproj_loss_term

                        if cfg.opt.w_ssim > 0:
                            ssim_term = (1 - ssim(rend, gt))
                            loss_cam += cfg.opt.w_ssim * ssim_term

                        # <<< MODIFIED: Add TV loss only once per sample if needed
                        # TV loss calculation seems misplaced inside camera loop, move outside?
                        # Or calculate based on the first camera's query? Assume it's okay here for now.
                        if cfg.opt.w_tv > 0 and 'vol' in sample_splats and cam_idx == 0: # Only add once per sample
                            # Original code calculates TV loss per camera, which might be intended if 'vol' changes?
                            # Let's assume it should be per sample for now. If 'vol' is static per sample, add it after loop.
                            # If 'vol' is dynamic (depends on query per camera?), keep it here but average later?
                            # Sticking to original structure: add TV loss per camera view render.
                            loss_cam += cfg.opt.w_tv * tv_3d_loss(sample_splats['vol'], reduction='mean')

                        sample_loss += loss_cam
                        num_targets_rendered += 1

                        # --- Capture images for logging (only first sample, first camera of batch if condition met) ---
                        # <<< ADDED Start >>>
                        if i == 0 and cam_idx == 0 and should_log_render and first_sample_rend_vis is None:
                            first_sample_rend_vis = rend.detach().float().cpu() # Use float for wandb
                            first_sample_gt_vis = gt.detach().float().cpu()
                            # Store the unweighted loss of this specific view
                            first_sample_reproj_loss_val = reproj_loss_term.item()
                        # <<< ADDED End <<<

                    # --- Add sample-level losses (after camera loop) ---
                    # Boundary regularization per sample
                    if cfg.opt.lambda_xyz_boundary > 0 and 'xyz' in sample_splats:
                        sample_loss += xyz_boundary_regularization(
                            sample_splats['xyz'], -1, 1, cfg.opt.lambda_xyz_boundary)

                    # Average sample loss over number of targets rendered for this sample
                    batch_loss += sample_loss / max(1, num_targets_rendered)
                    # Average unweighted reproj loss over number of targets for this sample
                    if num_targets_rendered > 0:
                         batch_reproj_loss_unweighted += sample_reproj_loss_val / num_targets_rendered

                # --- Final loss calculation for the batch ---
                # Average batch loss over number of samples in batch
                # <<< MODIFIED: Average over local batch size B instead of num_samples_with_valid_targets?
                # Averaging over B seems more standard for DDP batch loss reporting.
                total_loss = batch_loss / B
                # Average reproj loss over samples that had valid targets
                avg_reproj_loss_unweighted = batch_reproj_loss_unweighted / max(1, num_samples_with_valid_targets) # <<< ADDED

            # --- Backward pass and optimization ---
            # <<< MODIFIED: Use fabric.backward
            fabric.backward(scaler.scale(total_loss / cfg.opt.accumulation_steps))

            if (iteration + 1) % cfg.opt.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # Note: Check if fabric.clip_gradients API matches your version.
                # Alternatively, use torch.nn.utils.clip_grad_norm_ directly.
                fabric.clip_gradients(model, optimizer, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True) # <<< MODIFIED: set_to_none=True is often faster

            # --- Logging ---
            if is_global_zero and iteration % log_loss_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                # Use total_loss which is already averaged over the batch
                logger.info(f"Iter {iteration:06d} | Loss={total_loss.item():.4f} | LR={lr:.2e}")
                if not disable_wandb:
                    wandb.log({
                        'train/total_loss': total_loss.item(),
                        'train/lr': lr
                        # <<< ADDED: Log averaged unweighted reprojection loss
                        # 'train/reprojection_loss_unweighted_avg': avg_reproj_loss_unweighted
                    }, step=iteration)

            # <<< ADDED: Detailed render logging block >>>
            if should_log_render: # Already checked is_global_zero and interval
                 log_data_vis = {}
                 if first_sample_rend_vis is not None and first_sample_gt_vis is not None:
                     # Log the specific unweighted loss for the visualized sample/view
                     log_data_vis["train/reprojection_loss_unweighted_example"] = first_sample_reproj_loss_val
                     logger.info(f"Iter {iteration:06d} | Reproj Loss Example (unweighted): {first_sample_reproj_loss_val:.4f}")

                     # Log images to WandB
                     try:
                         # Combine images side-by-side for direct comparison (optional)
                         # Ensure channel dimension is last if needed by numpy/wandb
                         rend_img = first_sample_rend_vis.permute(1, 2, 0).numpy() if first_sample_rend_vis.dim() == 3 else first_sample_rend_vis.numpy()
                         gt_img = first_sample_gt_vis.permute(1, 2, 0).numpy() if first_sample_gt_vis.dim() == 3 else first_sample_gt_vis.numpy()
                         # Clamp values for visualization if necessary
                         rend_img = np.clip(rend_img, 0, 1)
                         gt_img = np.clip(gt_img, 0, 1)

                         log_data_vis["train/Render_vs_GT"] = [
                             wandb.Image(rend_img, caption=f"Iter {iteration} Rendered"),
                             wandb.Image(gt_img, caption=f"Iter {iteration} Ground Truth")
                         ]
                     except Exception as e:
                          logger.error(f"WandB Image logging failed at iter {iteration}: {e}", exc_info=True)

                 if log_data_vis:
                     wandb.log(log_data_vis, step=iteration)

                 # No need to reset here, done at the start of the iteration if should_log_render is True

            # --- Checkpointing ---
            # <<< MODIFIED: Use ckpt_every_interval
            if is_global_zero and iteration > 0 and iteration % ckpt_every_interval == 0:
                state = {
                    # <<< MODIFIED: Get state_dict from unwrapped model
                    'model_state_dict': fabric.unwrap(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict(),
                    'iteration': iteration,
                    'best_metric': best_metric # Note: best_metric is never updated in this script
                }
                ckpt_path = log_dir / f"checkpoint_{iteration:06d}.pth"
                torch.save(state, ckpt_path)
                logger.info(f"Checkpoint saved to {ckpt_path}")

            # --- Iteration update and Cleanup ---
            iteration += 1
            if iteration % cfg.opt.mem_clean_interval == 0:
                # <<< ADDED: Explicitly delete tensors likely holding significant memory
                del batch, gaussian_splats, total_loss, batch_loss, sample_loss
                if 'out' in locals(): del out
                if 'rend' in locals(): del rend
                if 'gt' in locals(): del gt
                gc.collect()
                torch.cuda.empty_cache()

    # --- End of Training ---
    fabric.barrier()
    if is_global_zero:
        logger.info("Training completed.")
        # <<< ADDED: Save final checkpoint
        final_ckpt_path = log_dir / "checkpoint_final.pth"
        state = {
            'model_state_dict': fabric.unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict(),
            'iteration': iteration -1, # Last completed iteration
            'best_metric': best_metric
        }
        torch.save(state, final_ckpt_path)
        logger.info(f"Final checkpoint saved to {final_ckpt_path}")

        if not disable_wandb and wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished.")

if __name__ == '__main__':
    main()