import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
import scipy.ndimage as ndimage
import os

def resize(scan, target_size):
    scan_x, scan_y, scan_z = scan.shape
    zoom_x = target_size / scan_x
    zoom_y = target_size / scan_y
    zoom_z = target_size / scan_z

    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        scan = ndimage.zoom(scan, (zoom_x, zoom_y, zoom_z), mode="nearest")
    return scan

def resample(image, spacing, new_spacing=[1, 1, 1]):
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
    min_dim = min(array.shape)
    start_indices = [(dim_size - min_dim) // 2 for dim_size in array.shape]
    end_indices = [start + min_dim for start in start_indices]
    return array[
        start_indices[0]:end_indices[0],
        start_indices[1]:end_indices[1],
        start_indices[2]:end_indices[2],
    ]

def expand_to_cube(array):
    max_dim = max(array.shape)
    padding = [(max_dim - s) // 2 for s in array.shape]
    padding = [(pad, max_dim - s - pad) for pad, s in zip(padding, array.shape)]
    return np.pad(array, padding, mode="constant", constant_values=0)

def process_mask(mask_data, spacing, target_size=256, mode=None):
    if mode is not None:
        mask_data, _ = resample(mask_data, spacing, [1, 1, 1])
        if mode == "crop":
            mask_data = crop_to_cube(mask_data)
        elif mode == "expand":
            mask_data = expand_to_cube(mask_data)
    mask_data = resize(mask_data, target_size)
    mask_data = (mask_data > 0.5).astype(np.float32)
    return mask_data

def segment_single_file(input_path, output_path, target_size=256, mode=None):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件 {input_path} 不存在")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载CT图像
    input_img = nib.load(input_path)
    spacing = input_img.header.get_zooms()[:3]
    
    # 执行身体分割
    output_seg = totalsegmentator(input_path, task="body",output=output_path)
    
    # 处理分割结果
    if isinstance(output_seg, dict):
        mask_data = output_seg.get("body", list(output_seg.values())[0]).get_fdata()
    else:
        mask_data = output_seg.get_fdata() if hasattr(output_seg, "get_fdata") else np.array(output_seg)
    
    # 后处理mask
    processed_mask = process_mask(mask_data, spacing, target_size, mode)
    
    # 保存结果
    np.save(output_path, processed_mask)
    print(f"分割结果已保存至: {output_path}")

if __name__ == "__main__":
    # 示例用法
    input_file = "/Disk_16TB/zhouhaowei/code/network_GAS/TMI/data/patient_00001_img_01.nii.gz"
    output_file = "/Disk_16TB/zhouhaowei/code/network_GAS/TMI/data/patient_00001_img_01mask.nii.gz"
    
    segment_single_file(
        input_path=input_file,
        output_path=output_file,
        target_size=256,  # 可自定义目标尺寸
        mode=None        # 可选None/"crop"/"expand"
    )