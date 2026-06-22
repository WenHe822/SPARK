import os  # 用于操作系统相关的功能，如路径处理
import sys  # 用于系统相关的功能，如修改模块搜索路径
from typing import NamedTuple  # 用于定义结构化的数据类（类似结构体）
import numpy as np  # 强大的数学库，用于数组和矩阵运算
import os.path as osp  # os.path 的别名，方便处理文件路径
import json  # 用于读取和解析 JSON 文件
import torch  # PyTorch 库，用于张量操作和 GPU 计算
import pickle  # 用于读取 Python 序列化数据（如 .pkl 文件）

# 将当前目录添加到模块搜索路径，以便导入自定义模块
sys.path.append("./")
# 从项目工具模块中导入点云相关函数（可能用于后续渲染或处理）
from r2_gaussian.utils.graphics_utils import BasicPointCloud, fetchPly

# 定义扫描模式的 ID 映射，parallel 表示平行束，cone 表示锥形束
mode_id = {
    "parallel": 0,  # 平行束模式
    "cone": 1,     # 锥形束模式
}

# 定义相机信息的数据结构，使用 NamedTuple 以确保字段不可变且类型明确
class CameraInfo(NamedTuple):
    uid: int              # 相机唯一标识符
    R: np.array           # 旋转矩阵（3x3），表示相机朝向（世界到相机）
    T: np.array           # 平移向量（3x1），表示相机位置
    angle: float          # 相机旋转角度（通常是射线源绕物体的角度）
    FovY: np.array        # 垂直视野角度（弧度）
    FovX: np.array        # 水平视野角度（弧度）
    image: np.array       # 投影图像数据（2D 数组）
    image_path: str       # 投影图像的文件路径
    image_name: str       # 图像文件名（不含扩展名）
    width: int            # 图像宽度（探测器像素数）
    height: int           # 图像高度（探测器像素数）
    mode: int             # 扫描模式（0 或 1，对应 mode_id）
    scanner_cfg: dict     # 扫描仪配置字典，包含几何参数
    mask_image: np.array = None  # mask投影图像数据（2D 数组），默认为None
    mask_image_path: str = None  # mask投影图像的文件路径，默认为None

# 定义场景信息的数据结构，封装整个场景的数据
class SceneInfo(NamedTuple):
    cameras: list   # 相机列表（CameraInfo 对象）

    vol: torch.tensor     # 体数据张量（3D 体素网格）
    vol_mask: torch.tensor# mask体数据张量（3D 体素网格），默认为None
    scanner_cfg: dict     # 扫描仪配置字典
    scene_scale: float    # 场景缩放因子，用于统一单位和范围

# 读取 Blender 格式的 CT 数据
def readBlenderInfo(path):
    """Read blender format CT data from a given path.
    Args:
        path (str): projections目录路径
        eval (bool): 是否用于评估（不再使用）
    Returns:
        SceneInfo: 包含相机和体数据的场景信息对象
    """
    # 拼接元数据文件路径并读取 JSON 文件
    meta_data_path = osp.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)  # 加载元数据，包含扫描仪配置和投影信息

    # 更新体数据的完整路径
    meta_data["vol"] = osp.join(path, meta_data["vol"])
    
    # 如果存在mask体数据，更新其完整路径
    vol_mask = None
    if "vol_mask" in meta_data:
        meta_data["vol_mask"] = osp.join(path, meta_data["vol_mask"])

    # 如果元数据中缺少体素大小 (dVoxel)，则通过总尺寸 (sVoxel) 和数量 (nVoxel) 计算
    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(
            np.array(meta_data["scanner"]["sVoxel"])  # 体素总尺寸
            / np.array(meta_data["scanner"]["nVoxel"])  # 体素数量
        )
    # 如果元数据中缺少探测器像素大小 (dDetector)，则通过总尺寸 (sDetector) 和数量 (nDetector) 计算
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(
            np.array(meta_data["scanner"]["sDetector"])  # 探测器总尺寸
            / np.array(meta_data["scanner"]["nDetector"])  # 探测器像素数
        )

    # 计算场景缩放因子，使得体数据的最大维度缩放到 [-1, 1]^3 范围内
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    # 对需要缩放的关键参数应用缩放因子，保持单位一致
    for key_to_scale in [
        "dVoxel",      # 体素大小
        "sVoxel",      # 体素总尺寸
        "sDetector",   # 探测器总尺寸
        "dDetector",   # 探测器像素大小
        "offOrigin",   # 物体中心的偏移
        "offDetector", # 探测器的偏移
        "DSD",         # 源到探测器距离
        "DSO",         # 源到物体中心距离
    ]:
        meta_data["scanner"][key_to_scale] = (
            np.array(meta_data["scanner"][key_to_scale]) * scene_scale
        ).tolist()  # 转为列表存储

    # 读取相机信息
    cam_infos = readCTameras(meta_data, path, scene_scale)

    # 读取体数据
    vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float()
    
    # 如果存在mask体数据，读取它
    vol_mask = None
    if "vol_mask" in meta_data:
        vol_mask = torch.from_numpy(np.load(meta_data["vol_mask"])).float()

    # 创建场景信息对象，封装所有数据
    scene_info = SceneInfo(
        cameras=cam_infos,  # 所有相机信息
        scanner_cfg=meta_data["scanner"],
        vol=vol_gt,
        vol_mask=vol_mask,
        scene_scale=scene_scale,
    )
    return scene_info

# 读取相机信息的核心函数
def readCTameras(meta_data, source_path, scene_scale=1.0):
    """Read camera info from CT metadata.
    Args:
        meta_data (dict): 元数据，包含扫描仪配置和投影信息
        source_path (str): projections目录路径
        scene_scale (float): 场景缩放因子
    Returns:
        list: 相机信息列表
    """
    # 提取扫描仪配置
    cam_cfg = meta_data["scanner"]
    cam_infos = []

    # 获取投影信息列表
    proj_info = meta_data["projections"]  
    n_projs = len(proj_info)
    
    # 检查是否有mask投影数据
    has_mask = "mask_projections" in meta_data
    mask_proj_info = meta_data.get("mask_projections", [])

    # 遍历每个投影
    for i_proj in range(n_projs):


        # 获取当前投影的元信息
        frame_info = proj_info[i_proj]
        frame_angle = frame_info["angle"]

        # 计算相机变换矩阵
        c2w = angle2pose(cam_cfg["DSO"], frame_angle)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        # 拼接投影图像的完整路径并加载数据
        image_path = osp.join(source_path, frame_info["file_path"])
        image = np.load(image_path) * scene_scale
        
        # 初始化mask相关变量
        mask_image = None
        mask_image_path = None
        
        # 如果有mask投影数据，读取对应的mask
        if has_mask and i_proj < len(mask_proj_info):
            mask_frame_info = mask_proj_info[i_proj]
            # 确保mask和投影的角度一致
            if abs(mask_frame_info["angle"] - frame_angle) < 1e-6:
                mask_image_path = osp.join(source_path, mask_frame_info["file_path"])
                mask_image = np.load(mask_image_path) * scene_scale

        # 计算视野角度
        FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
        FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2

        # 获取扫描模式
        mode = mode_id[cam_cfg["mode"]]

        # 创建相机信息对象
        cam_info = CameraInfo(
            uid=i_proj,
            R=R,
            T=T,
            angle=frame_angle,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=osp.basename(image_path).split(".")[0],
            width=cam_cfg["nDetector"][1],
            height=cam_cfg["nDetector"][0],
            mode=mode,
            scanner_cfg=cam_cfg,
            mask_image=mask_image,
            mask_image_path=mask_image_path,
        )
        cam_infos.append(cam_info)


    return cam_infos

# 将角度转换为相机姿态的变换矩阵
def angle2pose(DSO, angle):
    """Transfer angle to pose (c2w) based on scanner geometry.
    Args:
        DSO (float): 源到物体中心的距离
        angle (float): 旋转角度（弧度）
    Returns:
        np.array: 4x4 相机到世界的变换矩阵
    """
    # 第一步：绕 X 轴旋转 -90 度（固定轴）
    phi1 = -np.pi / 2
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi1), -np.sin(phi1)],
        [0.0, np.sin(phi1), np.cos(phi1)],
    ])
    # 第二步：绕 Z 轴旋转 90 度（固定轴）
    phi2 = np.pi / 2
    R2 = np.array([
        [np.cos(phi2), -np.sin(phi2), 0.0],
        [np.sin(phi2), np.cos(phi2), 0.0],
        [0.0, 0.0, 1.0],
    ])
    # 第三步：绕 Z 轴旋转指定角度（固定轴）
    R3 = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    # 组合三个旋转：R = R3 * R2 * R1
    rot = np.dot(np.dot(R3, R2), R1)
    # 计算平移向量：射线源沿圆周分布
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])

    # # 当angle=0时，相机应位于(DSO, 0, 0)位置
    # assert np.allclose(trans, [DSO, 0, 0], atol=1e-6), f"平移计算错误，得到{trans}"
    # # 当DSO=0时，相机应位于原点
    # if DSO == 0:
    #     assert np.allclose(trans, [0, 0, 0], atol=1e-6), "DSO为零时平移非零"
    # 创建 4x4 变换矩阵
    transform = np.eye(4)  # 初始化为单位矩阵
    transform[:3, :3] = rot  # 设置旋转部分
    transform[:3, 3] = trans  # 设置平移部分
    return transform

# 读取 NAF 格式的 CT 数据
def readNAFInfo(path, eval):
    """Read NAF format CT data from a given path.
    Args:
        path (str): projections目录路径
        eval (bool): 是否用于评估
    Returns:
        SceneInfo: 包含相机和体数据的场景信息对象
    """
    # 读取 pickle 文件
    with open(path, "rb") as f:
        data = pickle.load(f)  # 加载序列化数据

    # 定义扫描仪配置，单位从毫米转换为米（除以 1000）
    scanner_cfg = {
        "DSD": data["DSD"] / 1000,  # 源到探测器距离
        "DSO": data["DSO"] / 1000,  # 源到物体中心距离
        "nVoxel": data["nVoxel"],   # 体素数量
        "dVoxel": (np.array(data["dVoxel"]) / 1000).tolist(),  # 体素大小
        "sVoxel": (np.array(data["nVoxel"]) * np.array(data["dVoxel"]) / 1000).tolist(),  # 体素总尺寸
        "nDetector": data["nDetector"],  # 探测器像素数
        "dDetector": (np.array(data["dDetector"]) / 1000).tolist(),  # 探测器像素大小
        "sDetector": (np.array(data["nDetector"]) * np.array(data["dDetector"]) / 1000).tolist(),  # 探测器总尺寸
        "offOrigin": (np.array(data["offOrigin"]) / 1000).tolist(),  # 物体中心偏移
        "offDetector": (np.array(data["offDetector"]) / 1000).tolist(),  # 探测器偏移
        "totalAngle": data["totalAngle"],  # 总旋转角度
        "startAngle": data["startAngle"],  # 起始角度
        "accuracy": data["accuracy"],      # 精度参数
        "mode": data["mode"],              # 扫描模式
        "filter": None,                    # 滤波器（未使用）
    }

    # 计算场景缩放因子，使体数据缩放到 [-1, 1]^3
    scene_scale = 2 / max(scanner_cfg["sVoxel"])
    # 对几何参数应用缩放
    for key_to_scale in [
        "dVoxel", "sVoxel", "sDetector", "dDetector", "offOrigin", "offDetector", "DSD", "DSO",
    ]:
        scanner_cfg[key_to_scale] = (
            np.array(scanner_cfg[key_to_scale]) * scene_scale
        ).tolist()

    # 初始化相机信息
    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]
    cam_infos = {"train": [], "test": []}

    # 检查是否有mask数据
    has_mask = "mask" in data or "image_mask" in data
    mask_key = "mask" if "mask" in data else "image_mask" if "image_mask" in data else None

    # 遍历每个分割
    for split in splits:
        # 设置分割参数
        if split == "test":
            uid_offset = data["numTrain"]
            n_split = data["numVal"]
        else:
            uid_offset = 0
            n_split = data["numTrain"]
        # 选择数据来源
        if split == "test" and "val" in data:
            data_split = data["val"]
        else:
            data_split = data[split]
        angles = data_split["angles"]      # 投影角度列表
        projs = data_split["projections"]  # 投影图像列表
        
        # 获取mask投影（如果存在）
        mask_projs = None
        if has_mask and f"{split}_mask" in data:
            mask_projs = data[f"{split}_mask"]["projections"]
        elif has_mask and "mask_projections" in data_split:
            mask_projs = data_split["mask_projections"]

        # 遍历每个投影
        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_angle = angles[i_split]  # 当前投影角度
            c2w = angle2pose(scanner_cfg["DSO"], frame_angle)  # 相机到世界变换
            w2c = np.linalg.inv(c2w)  # 世界到相机变换
            R = np.transpose(w2c[:3, :3])  # 旋转矩阵（转置）
            T = w2c[:3, 3]  # 平移向量

            image = projs[i_split] * scene_scale  # 投影图像数据，应用缩放
            
            # 获取mask投影（如果存在）
            mask_image = None
            if mask_projs is not None and i_split < len(mask_projs):
                mask_image = mask_projs[i_split] * scene_scale

            # 计算视野角度
            FovX = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
            FovY = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2

            mode = mode_id[scanner_cfg["mode"]]  # 扫描模式 ID

            # 创建相机信息对象
            cam_info = CameraInfo(
                uid=i_split + uid_offset,
                R=R,
                T=T,
                angle=frame_angle,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=None,  # NAF 格式无文件路径
                image_name=f"{i_split + uid_offset:04d}",  # 用 ID 作为名称
                width=scanner_cfg["nDetector"][1],
                height=scanner_cfg["nDetector"][0],
                mode=mode,
                scanner_cfg=scanner_cfg,
                mask_image=mask_image,
                mask_image_path=None,  # NAF 格式无文件路径
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")

    # 提取训练和测试相机信息
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]
    
    # 读取体数据并转换为张量
    vol_gt = torch.from_numpy(data["image"]).float()
    
    # 读取mask体数据（如果存在）
    vol_mask = None
    if has_mask and mask_key is not None:
        vol_mask = torch.from_numpy(data[mask_key]).float()
    
    # 创建场景信息对象
    scene_info = SceneInfo(
        cameras=train_cam_infos + test_cam_infos,  # 合并训练和测试相机列表
        scanner_cfg=scanner_cfg,
        vol=vol_gt,
        vol_mask=vol_mask,
        scene_scale=scene_scale,
    )
    return scene_info

# 定义场景加载函数的映射，支持不同格式的数据读取
sceneLoadTypeCallbacks = {
    "Blender": readBlenderInfo,  # Blender 格式读取函数
    "NAF": readNAFInfo,          # NAF 格式读取函数
}