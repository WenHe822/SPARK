import os
import torch
import argparse
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
import logging
import sys
# --- 新增导入 ---
try:
    from PIL import Image
except ImportError:
    logging.error("请安装 Pillow 包 (pip install Pillow) 以保存渲染对比图像。")
    Image = None
try:
    import torchvision.transforms.functional as TF
except ImportError:
     logging.error("请安装 torchvision 包 (pip install torchvision) 以保存渲染对比图像。")
     TF = None
# --- 结束新增导入 ---

# 确保项目根目录在 sys.path 中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from train_network import setup_logging # setup_logging 在这里不需要，使用基本的 logging 配置
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_readers_ct import readBlenderInfo
# from datasets.dataset_ct import DatasetCT # DatasetCT 在推理时不需要
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import matrix_to_quaternion
# 导入 r2_gaussian 函数
try:
    from r2_gaussian.gaussian import render, query
except ImportError:
    logging.error("无法导入 r2_gaussian 包。请确保它已安装并且在 PYTHONPATH 中。")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_model(cfg, checkpoint_path, device):
    # 直接使用 cfg 初始化模型，假设 cfg 与训练时一致
    model = GaussianSplatPredictor(cfg)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型状态字典，处理可能的 DDP 'module.' 前缀或 EMA 权重
    state_dict_key = "model_state_dict"
    if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
        # 优先加载 EMA 状态 (如果训练时使用了 EMA 且保存了)
        # EMA 保存的是模型参数，不是 EMA 对象状态
        state_dict_key = "ema_state_dict" # 注意：确认 EMA 保存的是否是模型参数字典
        logger.info("检测到 EMA 状态，尝试加载 EMA 模型权重...")
        # 如果 ema_state_dict 保存的是 EMA 对象本身的状态，需要不同的加载方式
        # 假设保存的是可以直接加载到 model 的 state_dict
        model_state_dict = checkpoint[state_dict_key]
        # 可能需要移除 EMA 状态字典中的 'module.' 前缀（如果EMA包装了DDP模型）
        # 或者直接加载 model_state_dict (如果训练脚本保存的是原始模型或EMA模型参数)

    elif "model_state_dict" in checkpoint:
         model_state_dict = checkpoint["model_state_dict"]
         # 处理 DDP 保存的 'module.' 前缀
         if all(k.startswith('module.') for k in model_state_dict.keys()):
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in model_state_dict.items():
                 name = k[7:] # remove module.
                 new_state_dict[name] = v
             model_state_dict = new_state_dict
             logger.info("移除了模型状态字典中的 'module.' 前缀。")
    else:
        logger.error(f"检查点 {checkpoint_path} 中未找到 'model_state_dict' 或 'ema_state_dict'。")
        sys.exit(1)

    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"加载模型权重时丢失键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"加载模型权重时出现意外键: {unexpected_keys}")
        logger.info(f"已从检查点加载模型权重: {checkpoint_path} (使用键: {state_dict_key})")
    except Exception as e:
        logger.error(f"加载模型状态失败: {e}", exc_info=True)
        sys.exit(1)

    model.eval()
    return model

def save_volume(vol_tensor, save_path):
    vol_np = vol_tensor.detach().cpu().numpy()
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(vol_np, affine)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib.save(nii_img, save_path)
    logger.info(f"体积已保存为: {save_path}，形状: {vol_np.shape}")

def main():
    parser = argparse.ArgumentParser(description="高斯体推理脚本（适用于通用训练模型）") # 修改描述
    parser.add_argument('--proj_dir', type=str, required=True, help='包含投影和相机参数的测试数据目录路径')
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml', help='模型配置文件路径 (必须与训练时使用的配置完全一致)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='通用训练保存的模型权重路径 (例如 model_latest.pth 或 model_best.pth)')
    parser.add_argument('--output_path', type=str, default='output/volume_pred_general.nii.gz', help='输出体积保存路径') # 修改默认输出名
    parser.add_argument('--nVoxel', type=int, nargs=3, default=[256,256,256], help='体素化分辨率 [X, Y, Z]')
    parser.add_argument('--input_image_idx', type=int, default=108, help='选择测试数据集中用作输入的视角索引')
    # --- 新增参数 ---
    parser.add_argument('--render_view_indices', type=int, nargs='+', default=[0, 100, 108, 200, 300, 400, 500], help='指定要渲染并与 GT 对比的视角索引列表')
    # --- 结束新增参数 ---
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # --- 修改配置加载逻辑 ---
    try:
        # 加载基础配置文件
        cfg = OmegaConf.load(args.config_path)
        logger.info(f"加载基础配置文件: {args.config_path}")

        # 手动检查并合并 cam_embd 默认配置
        if not hasattr(cfg, 'cam_embd'):
            logger.info("基础配置中未找到 'cam_embd'，尝试加载默认配置...")
            config_dir = os.path.dirname(args.config_path)
            cam_embd_defaults_path = os.path.join(config_dir, "cam_embd", "defaults.yaml")
            if os.path.exists(cam_embd_defaults_path):
                cam_embd_cfg_content = OmegaConf.load(cam_embd_defaults_path)
                # 将加载的 cam_embd 配置添加到主 cfg 对象下
                cfg.cam_embd = cam_embd_cfg_content
                logger.info(f"成功加载并合并 cam_embd 默认配置: {cam_embd_defaults_path}")
            else:
                logger.error(f"必需的 cam_embd 默认配置文件未找到: {cam_embd_defaults_path}")
                raise FileNotFoundError(f"Required defaults file not found: {cam_embd_defaults_path}")
        else:
             logger.info("基础配置中已包含 'cam_embd'。")

        # 验证合并后的配置是否完整 (可选但推荐)
        if not hasattr(cfg, 'model') or not hasattr(cfg.model, 'num_gaussians') or not hasattr(cfg, 'cam_embd') or not hasattr(cfg.cam_embd, 'dimension'):
            logger.warning(f"加载/合并后的配置似乎仍不完整，请检查 {args.config_path} 和默认配置。")
            # 可能需要更严格的错误处理

    except FileNotFoundError as e:
        logger.error(f"配置文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"加载或解析配置文件失败: {e}", exc_info=True) # 添加 exc_info=True
        sys.exit(1)
    # --- 配置加载结束 ---

    # 加载模型
    model = load_model(cfg, args.ckpt_path, device)

    # 读取投影和相机参数
    logger.info(f"从目录加载场景信息: {args.proj_dir}")
    scene_info = readBlenderInfo(args.proj_dir)
    if not scene_info or not scene_info.cameras:
        logger.error(f"无法从 {args.proj_dir} 加载有效的场景信息或相机。")
        return
    cameras = cameraList_from_camInfos(scene_info.cameras)
    scanner_cfg = scene_info.scanner_cfg
    bbox = torch.stack([
        torch.tensor(scanner_cfg["offOrigin"]) - torch.tensor(scanner_cfg["sVoxel"]) / 2,
        torch.tensor(scanner_cfg["offOrigin"]) + torch.tensor(scanner_cfg["sVoxel"]) / 2,
    ], dim=0).to(device)

    # 选择输入视角
    input_image_idx = args.input_image_idx
    if not (0 <= input_image_idx < len(cameras)):
        logger.error(f"指定的输入视角索引 {input_image_idx} 超出范围 (共 {len(cameras)} 个视角)")
        return
    logger.info(f"使用测试数据集中的输入视角索引: {input_image_idx}")

    # 构造输入数据 (B=1, N_views=1)
    input_camera = cameras[input_image_idx]
    sample_image_tensor = input_camera.original_image.to(device)
    input_images = sample_image_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, C, H, W]
    logger.info(f"输入图像形状: {input_images.shape}")

    view_to_world = input_camera.view_world_transform.to(device)
    rotation_matrix_t = view_to_world[:3,:3].T
    source_quat = matrix_to_quaternion(rotation_matrix_t).unsqueeze(0)
    source_cv2wT_quat = source_quat.unsqueeze(0) # [1, 1, 4]
    logger.info(f"输入相机位姿四元数形状: {source_cv2wT_quat.shape}")

    # camera_params: 包含单个相机参数的列表 (因为输入视图数为1)
    camera_params = [{
        "angle": float(input_camera.angle),
        "view_to_world": view_to_world
    }]

    # scanner_cfg_list: 包含单个 scanner 配置的列表 (因为输入视图数为1)
    scanner_cfg_list = [scanner_cfg]

    # 推理
    logger.info("开始模型推理...")
    vol_pred = None # 初始化 vol_pred
    splats_for_query = None # 初始化
    with torch.no_grad():
        gaussian_splats = model(input_images, source_cv2wT_quat, camera_params, scanner_cfg_list)
        if not gaussian_splats or 'xyz' not in gaussian_splats:
             logger.error("模型输出为空或缺少 'xyz' 键。")
             return
        num_points = gaussian_splats['xyz'].shape[1]
        logger.info(f"模型输出高斯点数量: {num_points}")
        if num_points == 0:
            logger.warning("模型输出了0个高斯点，无法进行体素化和渲染对比。")
            # 即使没有点，也可能需要继续执行完脚本的其他部分，或者在这里退出
            # return # 如果没有点就退出
        else:
             # 准备用于体素化和渲染的 splats (提取 batch=0 的数据)
             splats_for_query = {k: v[0].to(device) for k, v in gaussian_splats.items()}
             if not splats_for_query:
                logger.error("无法为体素化和渲染准备高斯参数。")
                return

             # 体素化 (如果 splats 有效)
             logger.info("开始体素化...")
             center = (bbox[0] + bbox[1]) / 2
             sVoxel = (bbox[1] - bbox[0]).to(device)
             nVoxel = torch.tensor(args.nVoxel, device=device, dtype=torch.long)
             logger.info(f"体素化参数: 中心={center.cpu().numpy()}, 大小={sVoxel.cpu().numpy()}, 分辨率={nVoxel.cpu().numpy()}")

             vol_output = query(
                 gaussian_splats=splats_for_query,
                 center=center,
                 nVoxel=nVoxel,
                 sVoxel=sVoxel,
                 scaling_modifier=1.0
             )
             if "vol" not in vol_output or vol_output["vol"] is None:
                 logger.error("体素化函数未能返回有效的 'vol'。")
                 # 即使体素化失败，仍可尝试渲染对比
             else:
                 vol_pred = vol_output["vol"]
                 logger.info(f"体素化完成，输出体积形状: {vol_pred.shape}")

        # --- 新增：渲染选定视角并与 GT 对比 ---
        if splats_for_query and args.render_view_indices and Image and TF:
            logger.info(f"开始渲染选定视角进行对比: {args.render_view_indices}")
            comparison_output_dir = os.path.join(os.path.dirname(args.output_path), "render_comparison")
            os.makedirs(comparison_output_dir, exist_ok=True)

            for view_idx in args.render_view_indices:
                if not (0 <= view_idx < len(cameras)):
                    logger.warning(f"跳过无效的渲染视角索引: {view_idx} (总共 {len(cameras)} 个视角)")
                    continue

                target_camera = cameras[view_idx]
                # 假设 target_camera.original_image 是 [C, H, W] 范围 [0, 1] 的 Tensor on CPU
                gt_image_tensor = target_camera.original_image.cpu()

                try:
                    # 注意：假设 render 函数能处理 target_camera 的设备（或内部处理）
                    # splats_for_query 已经在 device 上
                    render_output = render(target_camera, splats_for_query) # 传入目标相机和预测点云

                    if "render" not in render_output or render_output["render"] is None:
                        logger.warning(f"视角 {view_idx} 的渲染函数未能返回有效的 'render'。")
                        continue

                    # 将渲染结果移到 CPU，确保范围在 [0, 1]
                    rendered_image_tensor = render_output['render'].detach().clamp(0, 1).cpu()

                    # 转换为 PIL Image (需要 torchvision)
                    gt_image_pil = TF.to_pil_image(gt_image_tensor)
                    rendered_image_pil = TF.to_pil_image(rendered_image_tensor)

                    # 保存图像
                    gt_path = os.path.join(comparison_output_dir, f"gt_view_{view_idx:03d}.png")
                    render_path = os.path.join(comparison_output_dir, f"render_view_{view_idx:03d}.png")
                    gt_image_pil.save(gt_path)
                    rendered_image_pil.save(render_path)

                    # 可选：创建并保存并排对比图
                    combined_img = Image.new('RGB', (gt_image_pil.width * 2, gt_image_pil.height))
                    combined_img.paste(gt_image_pil, (0, 0))
                    combined_img.paste(rendered_image_pil, (gt_image_pil.width, 0))
                    combined_path = os.path.join(comparison_output_dir, f"comparison_view_{view_idx:03d}.png")
                    combined_img.save(combined_path)

                    logger.info(f"已保存视角 {view_idx} 的对比图像于 {comparison_output_dir}")

                except Exception as e:
                    logger.error(f"渲染或保存视角 {view_idx} 时出错: {e}", exc_info=True)

        elif not (Image and TF):
             logger.warning("未安装 Pillow 或 torchvision，无法执行渲染对比和保存图像。")
        # --- 结束新增渲染对比逻辑 ---

    # 保存体积 (仅当体素化成功时)
    if vol_pred is not None:
        save_volume(vol_pred, args.output_path)
    else:
        logger.warning("由于体素化未成功或未执行，最终体积未保存。")

if __name__ == "__main__":
    main()
