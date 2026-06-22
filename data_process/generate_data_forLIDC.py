import os
import os.path as osp
import tigre
from tigre.utilities import gpu
from tigre.utilities import CTnoise
import numpy as np
import yaml
import json
import h5py
import argparse
from tqdm import tqdm
import sys

# 确保可以找到 r2_gaussian 包
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from r2_gaussian.utils.ct_utils import get_geometry_tigre # 假设这个函数可用且符合预期

def load_and_preprocess_ct(h5_file_path):
    """加载HDF5文件中的CT数据，进行截断和归一化。"""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'ct' not in f:
                print(f"警告: 'ct' dataset not found in {h5_file_path}. Skipping.")
                return None
            ct_data = f['ct'][:]
            # LIDC 数据通常是 Hounsfield units (HU)
            # 截断 [750, 1500] HU 并归一化到 [0, 1]
            ct_data = np.clip(ct_data, 750, 1500)
            ct_data = (ct_data - 750.0) / (1500.0 - 750.0)
        return ct_data.astype(np.float32)
    except Exception as e:
        print(f"错误:无法加载或处理 {h5_file_path}. 原因: {e}")
        return None

def main(args):
    """遍历LIDC数据集，为每个CT病例生成投影。"""
    input_dir = args.input_dir
    output_dir = args.output_dir
    scanner_cfg_path = args.scanner_cfg
    projections_num = args.projections_num

    # 1. 加载扫描仪配置
    try:
        with open(scanner_cfg_path, "r") as handle:
            scanner_cfg = yaml.safe_load(handle)
    except Exception as e:
        print(f"错误:无法加载扫描仪配置文件 {scanner_cfg_path}. 原因: {e}")
        return

    # 2. 获取TIGRE几何对象
    try:
        geo = get_geometry_tigre(scanner_cfg)
    except Exception as e:
        print(f"错误:无法从配置创建TIGRE几何对象. 原因: {e}")
        return

    # 3. 查找所有 HDF5 文件
    h5_files = []
    print(f"正在扫描输入目录: {input_dir}")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".h5") or file == "ct_xray_data.h5": # 兼容两种可能的命名
                 # 获取上一级目录名作为病例名称
                case_name = osp.basename(root)
                h5_files.append((osp.join(root, file), case_name))

    if not h5_files:
        print(f"警告:在 {input_dir} 中未找到HDF5文件。")
        return

    print(f"找到 {len(h5_files)} 个HDF5文件，开始处理...")

    # 4. 遍历每个HDF5文件并处理
    for h5_path, case_id in tqdm(h5_files, desc="处理CT病例"):
        # a. 加载和预处理CT数据
        vol = load_and_preprocess_ct(h5_path)
        if vol is None:
            continue # 跳过加载失败的文件

        # b. 定义输出路径
        case_save_path = osp.join(output_dir, f"{case_id}_{scanner_cfg['mode']}")
        os.makedirs(case_save_path, exist_ok=True)

        # c. 准备 TIGRE 输入 (TIGRE 通常期望 Z, Y, X 顺序)
        # 参考 generate_data.py 的做法
        vol_tigre = vol

        # d. 生成投影
        angles = (
            np.linspace(0, scanner_cfg["totalAngle"] / 180 * np.pi, projections_num + 1)[:-1]
            + scanner_cfg["startAngle"] / 180 * np.pi
        )
        try:
            projections = tigre.Ax(vol_tigre, geo, angles)
             # 同样参考 generate_data.py, 可能需要反转探测器维度

        except Exception as e:
            print(f"错误:为 {case_id} 生成投影失败. 原因: {e}")
            continue

        # e. 添加噪声 (如果配置中指定)
        if scanner_cfg.get("noise", False): # 使用 .get 提供默认值
            try:
                projections = CTnoise.add(
                    projections,
                    Poisson=scanner_cfg.get("possion_noise", 1e5), # 提供默认值
                    Gaussian=np.array(scanner_cfg.get("gaussian_noise", [0, 1])), # 提供默认值
                )
                projections[projections < 0.0] = 0.0
            except Exception as e:
                print(f"警告:为 {case_id} 添加噪声失败. 原因: {e}")
                # 即使加噪声失败，也继续保存无噪声的投影

        # f. 保存结果
        # 保存预处理后的原始CT体数据 (未转置的)
        np.save(osp.join(case_save_path, "vol_gt.npy"), vol)

        # 保存投影和创建元数据
        file_path_dict = []
        for i_proj in range(projections.shape[0]):
            proj = projections[i_proj]
            frame_save_name = f"projection{i_proj+1}.npy"
            np.save(osp.join(case_save_path, frame_save_name), proj)
            file_path_dict.append(
                {
                    "file_path": frame_save_name,
                    "angle": float(angles[i_proj]) # 转换为Python float类型以便JSON序列化
                }
            )

        # 创建元数据 (使用与 generate_data.py 类似的结构)
        meta = {
            "scanner": scanner_cfg,
            "vol": "vol_gt.npy",
             # 使用标准化边界框，因为我们归一化了体积
            "bbox": [[-1, -1, -1], [1, 1, 1]], # 或使用原始物理尺寸? 暂时用 0-1
            "original_h5_path": h5_path, # 记录原始文件路径
            "projections": file_path_dict
        }
        try:
            with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
        except Exception as e:
             print(f"错误:保存 {case_id} 的元数据失败. 原因: {e}")

    print(f"处理完成。结果保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为LIDC数据集生成X射线投影")

    parser.add_argument("--input_dir", default="/Disk_16TB/zhouhaowei/dataset/LIDC-HDF5-256", type=str, help="包含LIDC HDF5文件的根目录路径。")
    parser.add_argument("--output_dir", default="/Disk_10TB/zhouhaowei/data", type=str, help="保存生成数据的输出目录路径。")
    parser.add_argument("--scanner_cfg", default="data_process/scanner/cone_beam.yml", type=str, help="扫描仪配置文件的路径。")
    parser.add_argument("--projections_num", default=400, type=int, help="要为每个CT生成的投影数量。")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)


    main(args)
