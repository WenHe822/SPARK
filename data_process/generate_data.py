import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
import plotly.graph_objects as go
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise
import json
import matplotlib.pyplot as plt
import tigre.algorithms as algs
import argparse
import open3d as o3d
import cv2
import pickle
import copy

import sys

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume


def main(args):
    """Assume CT is in a unit cube. We synthesize X-ray projections."""
    vol_path = args.vol
    scanner_cfg_path = args.scanner
    projections_num = args.projections_num
    vol_name = osp.basename(vol_path)[:-4]
    output_path = args.output

    # Load configuration
    with open(scanner_cfg_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)

    case_name = f"{vol_name}_{scanner_cfg['mode']}"
    print(f"Generate data for case {case_name}")
    geo = get_geometry_tigre(scanner_cfg)

    # Load volume
    vol = np.load(vol_path).astype(np.float32)

    # Generate projections
    angles = (
        np.linspace(0, scanner_cfg["totalAngle"] / 180 * np.pi, projections_num + 1)[:-1]
        + scanner_cfg["startAngle"] / 180 * np.pi
    )
    projections = tigre.Ax(
        np.transpose(vol, (2, 1, 0)).copy(), geo, angles
    )[:, ::-1, :]
    
    if scanner_cfg["noise"]:
        projections = CTnoise.add(
            projections,
            Poisson=scanner_cfg["possion_noise"],
            Gaussian=np.array(scanner_cfg["gaussian_noise"]),
        )
        projections[projections < 0.0] = 0.0

    # Save
    case_save_path = osp.join(output_path, case_name)
    os.makedirs(case_save_path, exist_ok=True)
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
                "angle": float(angles[i_proj])
            }
        )
    
    meta = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "bbox": [[-1, -1, -1], [1, 1, 1]],
        "projections": file_path_dict
    }
    with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"Generate data for case {case_name} complete!")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Data generator parameters")
    
    parser.add_argument("--vol", default="data_generator/volume_gt/0_chest.npy", type=str, help="Path to volume.")
    parser.add_argument("--scanner", default="data_generator/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_projections", type=str, help="Path to output.")
    parser.add_argument("--projections_num", default=150, type=int, help="Number of projections to generate.")
    # fmt: on

    args = parser.parse_args()
    main(args)
