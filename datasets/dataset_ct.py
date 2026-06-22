import os
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.dataset_readers_ct import readBlenderInfo, SceneInfo
from datasets.shared_dataset import SharedDataset
from utils.camera_utils import cameraList_from_camInfos
import numpy as np
import random
from torch.utils.data import DataLoader
# class DatasetCT(SharedDataset):
#     def __init__(self, data_path, type):
#         """
#         Args:
#             data_path (str): 数据根目录路径
#             type (str): 'train' 或 'test'，指定加载哪个数据集
#         """
#         self.type = type
#         self.data_path = data_path
        
#         # 获取所有投影子目录
#         self.proj_dirs = []
#         type_path = os.path.join(data_path, type)
#         for item in os.listdir(type_path):
#             item_path = os.path.join(type_path, item)
#             # if os.path.isdir(item_path) and item.startswith('projections'):
#             self.proj_dirs.append(item_path)
                
#         print(f"找到 {len(self.proj_dirs)} 个{type}投影数据集")
        
#         # 设置每个对象返回的图片数量和输入图片数量
#         self.imgs_per_obj = 10  # 每个对象返回的总图片数
#         self.input_images = 2   # 用作条件输入的图片数
        
#         # 对于测试集，设置固定的输入图片索引
#         self.test_input_idxs = [0]  # 始终使用第一张图片作为输入

#     def __len__(self):
#         return len(self.proj_dirs)
    
#     def __getitem__(self, idx):


#         proj_dir = self.proj_dirs[idx]
        
#         # 读取场景信息
#         scene_info = readBlenderInfo(proj_dir)
#         cameras = cameraList_from_camInfos(scene_info.cameras)#一个列表，每个元素是一个Camera对象，一个Camera对象对应一个投影的信息
#         vol_gt = scene_info.vol
#         vol_mask = scene_info.vol_mask  # 获取mask体数据
#         scanner_cfg = scene_info.scanner_cfg
#         scene_scale = scene_info.scene_scale
#         bbox = torch.stack(
#             [
#                 torch.tensor(scanner_cfg["offOrigin"])
#                 - torch.tensor(scanner_cfg["sVoxel"]) / 2,
#                 torch.tensor(scanner_cfg["offOrigin"])
#                 + torch.tensor(scanner_cfg["sVoxel"]) / 2,
#             ],
#             dim=0,
#         )

#         # 选择要返回的相机/图片
#         if self.type == "train":
#             # 对于训练集，随机选择 imgs_per_obj 张图片
#             total_cameras = len(cameras)
#             if total_cameras > self.imgs_per_obj:
#                 # 随机选择索引
#                 indices = torch.randperm(total_cameras)[:self.imgs_per_obj]
#                 # 确保前 input_images 张图片被包含
#                 indices = torch.cat([indices[:self.input_images], indices], dim=0)
#                 # 根据索引选择相机
#                 cameras = [cameras[i] for i in indices]
#         else:
#             # 对于测试集，使用固定的输入图片索引
#             total_cameras = len(cameras)
#             # 修正：动态计算中间相机的索引
#             input_indices = self.test_input_idxs + [total_cameras // 2]  # 使用第一张和中间的图片
#             other_indices = [i for i in range(total_cameras) if i not in input_indices]
#             indices = input_indices + other_indices
#             cameras = [cameras[i] for i in indices]

#         # 收集所有相机的变换矩阵
#         world_view_transforms = []
#         view_to_world_transforms = []
#         full_proj_transforms = []
#         camera_centers = []

#         for cam in cameras:
#             world_view_transforms.append(cam.world_view_transform)
#             view_to_world_transforms.append(cam.world_view_transform.inverse())
#             full_proj_transforms.append(cam.full_proj_transform)
#             camera_centers.append(cam.camera_center)

#         # 转换为张量
#         world_view_transforms = torch.stack(world_view_transforms)
#         view_to_world_transforms = torch.stack(view_to_world_transforms)
#         full_proj_transforms = torch.stack(full_proj_transforms)
#         camera_centers = torch.stack(camera_centers)

#         # 构建相机姿态字典
#         camera_poses = {
#             "world_view_transforms": world_view_transforms,
#             "view_to_world_transforms": view_to_world_transforms,
#             "full_proj_transforms": full_proj_transforms,
#             "camera_centers": camera_centers
#         }

#         # 转换为相对位姿
#         #camera_poses = self.make_poses_relative_to_first(camera_poses)
        
#         # 计算四元数表示
#         source_cv2wT_quat = self.get_source_cw2wT(camera_poses["view_to_world_transforms"])

#         # # 更新相机参数
#         # for i, cam in enumerate(cameras):
#         #     cam.world_view_transform = camera_poses["world_view_transforms"][i]
#         #     cam.full_proj_transform = camera_poses["full_proj_transforms"][i]
#         #     cam.camera_center = camera_poses["camera_centers"][i]
            
#         return {
#             "cameras": cameras,
#             "scanner_cfg": scanner_cfg,
#             "vol": vol_gt,
#             "vol_mask": vol_mask,  # 添加mask体数据到返回字典
#             "scene_scale": scene_scale,
#             "bbox": bbox,
#             "source_cv2wT_quat": source_cv2wT_quat
#         }

class DatasetCT(SharedDataset):
    def __init__(self, data_path, type):
        """
        Args:
            data_path (str): 数据根目录路径
            type (str): 'train' 或 'test'，指定加载哪个数据集
        """
        self.type = type
        self.data_path = data_path
        
        # 获取所有投影子目录
        self.proj_dirs = []
        type_path = os.path.join(data_path, type)
        for item in os.listdir(type_path):
            item_path = os.path.join(type_path, item)
            self.proj_dirs.append(item_path)
                
        # 设置每个对象返回的图片数量和输入图片数量
        self.imgs_per_obj = 8  # 每个对象返回的总图片数
        self.input_images = 1   # 用作条件输入的图片数
        
        # 设置固定的输入投影为projection81.npy
        self.target_projection = "projection81"

    def __len__(self):
        return len(self.proj_dirs)
    
    def __getitem__(self, idx):
        proj_dir = self.proj_dirs[idx]
        
        # 读取场景信息
        scene_info = readBlenderInfo(proj_dir)
        cameras = cameraList_from_camInfos(scene_info.cameras)
        vol_gt = scene_info.vol
        vol_mask = scene_info.vol_mask 
        scanner_cfg = scene_info.scanner_cfg
        scene_scale = scene_info.scene_scale
        bbox = torch.stack(
            [
                torch.tensor(scanner_cfg["offOrigin"])
                - torch.tensor(scanner_cfg["sVoxel"]) / 2,
                torch.tensor(scanner_cfg["offOrigin"])
                + torch.tensor(scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

        # 选择要返回的相机/图片
        total_cameras = len(cameras)
        
        # 查找目标投影的索引
        target_idx = -1
        for i, cam in enumerate(cameras):
            if self.target_projection in cam.image_name:
                target_idx = i
                break
        
        # 如果找不到目标投影，使用默认第一张
        if target_idx == -1:
            print(f"警告：在项目 {idx} 中找不到 {self.target_projection}，使用第一张投影代替")
            target_idx = 0
            
        # 准备其他相机索引
        other_indices = [i for i in range(total_cameras) if i != target_idx]
        
        if self.type == "train":
            # 对于训练集，随机选择其他投影
            if len(other_indices) > self.imgs_per_obj - 1:
                selected_others = random.sample(other_indices, self.imgs_per_obj - 1)
            else:
                selected_others = other_indices
        else:
            # 对于测试集，使用所有其他投影
            selected_others = other_indices
            
        # 组合索引，确保目标投影在第一位
        indices = [target_idx] + selected_others
        # 限制返回的投影数量
        if len(indices) > self.imgs_per_obj:
            indices = indices[:self.imgs_per_obj]
            
        # 根据索引选择相机
        cameras = [cameras[i] for i in indices]

        # 收集所有相机的变换矩阵
        world_view_transforms = []
        view_to_world_transforms = []
        full_proj_transforms = []
        camera_centers = []

        for cam in cameras:
            world_view_transforms.append(cam.world_view_transform)
            view_to_world_transforms.append(cam.world_view_transform.inverse())
            full_proj_transforms.append(cam.full_proj_transform)
            camera_centers.append(cam.camera_center)

        # 转换为张量
        world_view_transforms = torch.stack(world_view_transforms)#(imgs_per_obj,4,4)
        view_to_world_transforms = torch.stack(view_to_world_transforms)#(imgs_per_obj,4,4)
        full_proj_transforms = torch.stack(full_proj_transforms)#(imgs_per_obj,4,4)
        camera_centers = torch.stack(camera_centers)#(imgs_per_obj,3)

        # 构建相机姿态字典
        camera_poses = {
            "world_view_transforms": world_view_transforms,
            "view_to_world_transforms": view_to_world_transforms,
            "full_proj_transforms": full_proj_transforms,
            "camera_centers": camera_centers
        }
        
        # 计算四元数表示
        source_cv2wT_quat = self.get_source_cw2wT(camera_poses["view_to_world_transforms"])#(imgs_per_obj,4)

        # 获取文件夹名称
        folder_name = os.path.basename(proj_dir)

        return {
            "cameras": cameras,
            "scanner_cfg": scanner_cfg,
            "vol": vol_gt,
            "vol_mask": vol_mask,
            "scene_scale": scene_scale,
            "bbox": bbox,
            "source_cv2wT_quat": source_cv2wT_quat,
            "folder_name": folder_name # 添加文件夹名称
        }

'''返回的字典 (dict)
├── cameras (list of Camera 对象)
│   └── 每个 Camera 对象包含的字段：
│       ├── uid: int
│       ├── colmap_id: int
│       ├── R: torch.Tensor (shape: [3, 3])
│       ├── T: torch.Tensor (shape: [3])
│       ├── angle: float
│       ├── FoVx: float
│       ├── FoVy: float
│       ├── mode: int
│       ├── image_name: str
│       ├── original_image: torch.Tensor (shape: [1, H, W])
│       ├── mask_image: torch.Tensor (shape: [1, H, W]) 
│       ├── image_width: int
│       ├── image_height: int
│       ├── trans: np.array
│       ├── scale: float
│       ├── world_view_transform: torch.Tensor (shape: [4, 4])
│       ├── projection_matrix: torch.Tensor (shape: [4, 4])
│       ├── full_proj_transform: torch.Tensor (shape: [4, 4])
│       └── camera_center: torch.Tensor (shape: [3])
├── scanner_cfg (dict)
│   └── 包含扫描仪配置参数（字段举例）：
│       ├── dVoxel: list
│       ├── sVoxel: list
│       ├── sDetector: list
│       ├── dDetector: list
│       ├── offOrigin: list
│       ├── offDetector: list
│       ├── DSD: float
│       ├── DSO: float
│       └── ...（其它相关配置参数）
├── vol (torch.Tensor)
│   └── CT 体数据张量（形状依数据而定，如 [D, H, W]）
├── vol_mask (torch.Tensor)
│   └── CT mask体数据张量（形状与vol相同）
├── scene_scale (float)
│   └── 场景缩放因子（用于归一化至 [-1, 1]^3）
├── bbox (torch.Tensor)
│   └── 二维张量，形状为 [2, 3]
│       ├── 第一行：下界 = offOrigin - sVoxel/2
│       └── 第二行：上界 = offOrigin + sVoxel/2
└── source_cv2wT_quat (torch.Tensor 或类似结构)
    └── 表示首个输入相机的视角到世界变换的四元数（通常包含 4 个元素）
'''
if __name__ == "__main__":


    import matplotlib.pyplot as plt
    import os
    import random
    import torch
    
    # 创建数据集实例
    train_dataset = DatasetCT(data_path="/home/haowei_zhou/Project/Gaussian_splatting/TMI/data/Singleprojections_withmask", type="train")
    
    # 创建保存图像的目录
    save_dir = "visualization_samples"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 随机选择一个数据样本（一个CT体的投影集合）
    sample_idx = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[sample_idx]
    
    # 获取该样本的所有相机（投影）
    cameras = sample["cameras"]
    print(f"从体积 {sample_idx} 中选择了 {len(cameras)} 个投影")
    
    # 随机选择5个投影（如果投影数量不足5个，则使用所有投影）
    num_projections = min(5, len(cameras))
    selected_indices = random.sample(range(len(cameras)), num_projections)
    
    for i, cam_idx in enumerate(selected_indices):
        camera = cameras[cam_idx]
        
        # 获取图像
        original_image = camera.original_image.squeeze().cpu().numpy()  # 形状为 [H, W]
        
        # 保存原始图像
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f"投影 {cam_idx} - 角度: {camera.angle:.2f}")
        plt.colorbar()
        
        # 检查是否有mask图像并保存
        if hasattr(camera, 'mask_image') and camera.mask_image is not None:
            mask_image = camera.mask_image.squeeze().cpu().numpy()
            plt.subplot(1, 2, 2)
            plt.imshow(mask_image, cmap='gray')
            plt.title(f"Mask {cam_idx}")
            plt.colorbar()
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"projection_{sample_idx}_{cam_idx}.png"))
        plt.close()
        
        print(f"保存了投影 {cam_idx} 的图像和mask")
    
    print(f"完成。图像已保存到 {save_dir} 目录")