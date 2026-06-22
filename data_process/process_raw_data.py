#!/usr/bin/env python3
"""
处理 nii.gz 格式数据：
1. 读取 nii.gz 文件，归一化到 [0, 1]；
2. 将体数据重采样为目标立方体尺寸（默认 256³）；
3. 保存为 .npy 文件。
"""

import os
import os.path as osp
import glob
import argparse

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from tqdm import tqdm


def process_nii(file_path, target_size, mode=None):
    """
    处理单个 nii 或 nii.gz 文件。
    
    参数：
      file_path: nii 文件路径
      target_size: 输出立方体的边长（例如 256）
      mode: 重塑模式，可选值为 "crop"（裁剪）或 "expand"（填充），默认为 None（直接缩放）
      
    返回：
      处理后的 3D numpy 数组，值在 [0, 1] 范围内
    """
    # 读取 nii 文件
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # 数据归一化到 [0, 1]
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)
    data = np.clip(data, 0.0, 1.0)
    
    # 获取 spacing 信息（体素尺寸），取前三个维度
    spacing = img.header.get_zooms()[:3]
    
    # 根据模式对体数据进行重塑
    data = reshape_vol(data, spacing, target_size, mode)
    data = np.clip(data, 0.0, 1.0)
    return data


def reshape_vol(image, spacing, target_size, mode=None):
    """
    重塑体数据为立方体。
    
    如果 mode 不为 None，则先对物理尺寸进行 resample，
    并根据 mode 对数据进行裁剪或扩展，再统一缩放到 target_size。
    
    参数：
      image: 原始 3D 数组
      spacing: 原始体素尺寸信息
      target_size: 目标立方体边长
      mode: None 或 "crop" 或 "expand"
      
    返回：
      处理后的 3D 数组
    """
    if mode is not None:
        image, _ = resample(image, spacing, [1, 1, 1])
        if mode == "crop":
            image = crop_to_cube(image)
        elif mode == "expand":
            image = expand_to_cube(image)
        else:
            raise ValueError("Unsupported reshape mode!")
    image_new = resize(image, target_size)
    return image_new


def resample(image, spacing, new_spacing=[1, 1, 1]):
    """
    根据新的体素尺寸，对图像进行 resample，保持物理尺寸稳定。
    
    参数：
      image: 原始 3D 数组
      spacing: 原始体素尺寸（数组或列表）
      new_spacing: 目标体素尺寸，默认为 [1, 1, 1]
      
    返回：
      resample 后的图像和新的 spacing
    """
    spacing = np.array(list(spacing))
    new_spacing = np.array(new_spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = np.array(image.shape) * resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    real_resize_factor = new_shape / np.array(image.shape)
    new_spacing = spacing / real_resize_factor
    image = ndimage.zoom(image, real_resize_factor, mode="nearest")
    return image, new_spacing


def crop_to_cube(array):
    """
    从数组中心裁剪出立方体区域（取最小尺寸）。
    """
    min_dim = min(array.shape)
    start_indices = [(dim_size - min_dim) // 2 for dim_size in array.shape]
    end_indices = [start + min_dim for start in start_indices]
    cubic_region = array[
        start_indices[0]:end_indices[0],
        start_indices[1]:end_indices[1],
        start_indices[2]:end_indices[2],
    ]
    return cubic_region


def expand_to_cube(array):
    """
    将数组扩展为立方体，边缘使用 0 填充。
    """
    max_dim = max(array.shape)
    # 计算每个维度两侧的 padding
    padding = [(max_dim - s) // 2 for s in array.shape]
    padding = [(pad, max_dim - s - pad) for pad, s in zip(padding, array.shape)]
    cubic_array = np.pad(array, padding, mode="constant", constant_values=0)
    return cubic_array


def resize(scan, target_size):
    """
    将体数据缩放到目标立方体尺寸。
    
    参数：
      scan: 原始 3D 数组
      target_size: 目标尺寸（立方体边长）
      
    返回：
      缩放后的 3D 数组
    """
    scan_x, scan_y, scan_z = scan.shape
    zoom_x = target_size / scan_x
    zoom_y = target_size / scan_y
    zoom_z = target_size / scan_z

    # 如果缩放比例不为 1，则进行缩放
    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        scan = ndimage.zoom(scan, (zoom_x, zoom_y, zoom_z), mode="nearest")
    return scan


def main():
    parser = argparse.ArgumentParser(
        description="处理 nii/nii.gz 数据，归一化并重塑为立方体后保存为 .npy 文件"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 nii/nii.gz 文件所在的文件夹路径（文件应平铺在此文件夹下）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 .npy 文件保存的文件夹路径",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="输出体数据的目标立方体边长，默认 256",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[None, "crop", "expand"],
        help="重塑模式：'crop' 表示从中心裁剪，'expand' 表示扩展填充；默认直接缩放",
    )
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    
    # 搜索 nii.gz 与 nii 文件
    nii_files = glob.glob(osp.join(input_folder, "*.nii.gz")) + glob.glob(osp.join(input_folder, "*.nii"))
    if len(nii_files) == 0:
        print("在输入文件夹中未找到 nii 或 nii.gz 文件。")
        return

    for file_path in tqdm(nii_files, desc="Processing nii files"):
        data = process_nii(file_path, args.target_size, args.mode)
        basename = os.path.basename(file_path)
        # 去掉扩展名
        if basename.endswith(".nii.gz"):
            basename = basename[:-7]
        elif basename.endswith(".nii"):
            basename = basename[:-4]
        output_file = osp.join(output_folder, basename + ".npy")
        np.save(output_file, data.astype(np.float32))


if __name__ == "__main__":
    main()
