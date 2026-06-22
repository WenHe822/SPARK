import os
import torch
import argparse
import numpy as np
import pickle
import logging
import sys
from omegaconf import OmegaConf

# 确保项目根目录在 sys.path 中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_readers_ct import readBlenderInfo
from utils.camera_utils import cameraList_from_camInfos

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

    # 确保返回与输入设备一致的张量
    return torch.tensor([r, x, y, z], dtype=torch.float32, device=M.device)

def load_model(cfg, checkpoint_path, device):
    # 直接使用 cfg 初始化模型，假设 cfg 与训练时一致
    model = GaussianSplatPredictor(cfg)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型状态字典，处理可能的 DDP 'module.' 前缀或 EMA 权重
    state_dict_key = "model_state_dict"
    if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
        # 优先加载 EMA 状态 (如果训练时使用了 EMA 且保存了)
        state_dict_key = "ema_state_dict"
        logger.info("检测到 EMA 状态，尝试加载 EMA 模型权重...")
        model_state_dict = checkpoint[state_dict_key]
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

def save_gaussian_params(gaussian_splats, save_path, save_format="npy"):
    """
    保存高斯参数到指定格式
    args:
        gaussian_splats: 网络输出的高斯参数字典
        save_path: 保存路径
        save_format: 保存格式，可以是 "npy" 或 "pickle"
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 提取第一个批次的数据 (B=0)
    xyz = gaussian_splats['xyz'][0].detach().cpu().numpy()
    density = gaussian_splats['density'][0].detach().cpu().numpy()
    scaling = gaussian_splats['scaling'][0].detach().cpu().numpy()
    rotation = gaussian_splats['rotation'][0].detach().cpu().numpy()
    
    # 对数据进行转换，确保每个点云点有位置和密度
    n_points = xyz.shape[0]
    logger.info(f"保存 {n_points} 个高斯点")
    
    if save_format == "npy":
        # NPY格式: 只保存位置和密度 [x, y, z, density]
        # 按照initialize_pcd.py中的格式，将密度维度合并到最后一列
        point_cloud = np.concatenate([xyz, density], axis=-1)
        np.save(save_path, point_cloud)
        logger.info(f"高斯点参数已保存为 NPY 格式: {save_path}")
    elif save_format == "pickle":
        # Pickle格式: 保存所有参数
        gaussian_data = {
            "xyz": xyz,
            "density": density,
            "scaling": scaling,
            "rotation": rotation
        }
        with open(save_path, 'wb') as f:
            pickle.dump(gaussian_data, f)
        logger.info(f"高斯点参数已保存为 Pickle 格式: {save_path}")
    else:
        logger.error(f"不支持的保存格式: {save_format}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="高斯点参数推理和保存脚本")
    parser.add_argument('--proj_dir', type=str, required=True, help='包含投影和相机参数的测试数据目录路径')
    parser.add_argument('--config_path', type=str, required=True, help='模型配置文件路径 (必须与训练时使用的配置完全一致)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='保存的模型权重路径 (例如 model_latest.pth 或 model_best.pth)')
    parser.add_argument('--output_path', type=str, required=True, help='输出高斯点参数保存路径，支持 .npy 或 .pickle 后缀')
    parser.add_argument('--input_image_idx', type=int, default=108, help='选择测试数据集中用作输入的视角索引')
    args = parser.parse_args()

    # 检查输出格式
    output_ext = os.path.splitext(args.output_path)[1].lower()
    if output_ext == '.npy':
        save_format = 'npy'
    elif output_ext in ['.pickle', '.pkl']:
        save_format = 'pickle'
        args.output_path = os.path.splitext(args.output_path)[0] + '.pickle'  # 确保使用 .pickle 后缀
    else:
        logger.error(f"不支持的输出格式 {output_ext}，请使用 .npy 或 .pickle")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

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

    except Exception as e:
        logger.error(f"加载或解析配置文件失败: {e}", exc_info=True)
        sys.exit(1)

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
    with torch.no_grad():
        import time
        start_time = time.time()  # 记录开始时间
        gaussian_splats = model(input_images, source_cv2wT_quat, camera_params, scanner_cfg_list)
        inference_time = time.time() - start_time  # 计算推理时间
        logger.info(f"推理时间: {inference_time:.4f} 秒")  # 记录推理时间
        if not gaussian_splats or 'xyz' not in gaussian_splats:
             logger.error("模型输出为空或缺少 'xyz' 键。")
             return
        num_points = gaussian_splats['xyz'].shape[1]
        logger.info(f"模型输出高斯点数量: {num_points}")
        if num_points == 0:
            logger.warning("模型输出了0个高斯点，无法保存高斯参数。")
            return
        
        # 保存高斯参数
        save_gaussian_params(gaussian_splats, args.output_path, save_format)

if __name__ == "__main__":
    main() 
#python infference_save_pcd.py --proj_dir data/TCIA_projections_512res_600numproj/train/patient_00001_img_03_cone --ckpt_path /Disk_16TB/zhouhaowei/code/network_GAS/TMI/experiments_out/2025-04-25/10-24-50/logs/1/model_latest.pth --config_path configs/default_config.yaml --output_path data/TCIA_projections_512res_600numproj/train/patient_00001_img_03_cone/patient_00001_img_03_cone.npy   