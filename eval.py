import torch
import os
import json
import tqdm
import numpy as np

from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from datasets.dataset_factory import get_dataset
from scene.gaussian_predictor import GaussianSplatPredictor
from r2_gaussian.gaussian import render, query
from utils.loss_utils import l1_loss, l2_loss, ssim, tv_3d_loss

def evaluate_model(model, dataloader, cfg, device, save_vis=0, out_folder=None):
    """
    对传入的CBCT数据集做评估, 返回dict包含平均 L1/L2, SSIM, (可选) TV Loss等
    """
    model.eval()
    all_l12 = []
    all_ssim = []
    all_tv  = []

    if save_vis > 0 and out_folder is not None:
        os.makedirs(out_folder, exist_ok=True)

    for idx, data in enumerate(tqdm.tqdm(dataloader, desc="评估中")):
        data = data[0]
        cameras = data["cameras"]
        scanner_cfg = data["scanner_cfg"]
        bbox = data["bbox"]

        # 1. 构造输入
        input_images = []
        camera_params_list = []
        for cam_idx in range(cfg.data.input_images):
            input_images.append(cameras[cam_idx].original_image)
            cam_dict = {
                "angle": float(cameras[cam_idx].angle),
                "view_to_world": cameras[cam_idx].view_world_transform
            }
            camera_params_list.append(cam_dict)
        input_images = torch.stack(input_images, dim=0).unsqueeze(0).to(device)
        source_cv2wT_quat = data["source_cv2wT_quat"][:cfg.data.input_images]
        # 2. 前向预测
        gaussian_splats = model(input_images,source_cv2wT_quat, camera_params_list, scanner_cfg)
        for k, v in gaussian_splats.items():
            gaussian_splats[k] = v.squeeze(0)

        # 3. 渲染所有视角
        rendered_images = []
        gt_images = []
        for cam_idx in range(len(cameras)):
            render_out = render(cameras[cam_idx], gaussian_splats)
            rendered_images.append(render_out["render"].unsqueeze(0))
            gt_images.append(cameras[cam_idx].original_image.unsqueeze(0))
        rendered_images = torch.cat(rendered_images, dim=0).to(device)
        gt_images = torch.cat(gt_images, dim=0).to(device)

        # 4. 计算 L1/L2
        if cfg.opt.loss == "l2":
            l12_val = l2_loss(rendered_images, gt_images).item()
        else:
            l12_val = l1_loss(rendered_images, gt_images).item()

        # 计算SSIM
        ssim_val = ssim(rendered_images, gt_images).item()

        # (可选) 计算3D TV
        # 按照当前训练流程中的方式查询体数据
        # 使用与训练一致的方式计算TV损失
        vol_pred = query(
            gaussian_splats,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"]
        )["vol"]  # 注意这里键名应该是"vol"而不是"volume"
        
        tv_val = tv_3d_loss(vol_pred, reduction="mean").item()  # 添加reduction参数保持一致

        all_l12.append(l12_val)
        all_ssim.append(ssim_val)
        all_tv.append(tv_val)

        # (可选) 可视化存图
        if save_vis > 0 and idx < save_vis and out_folder is not None:
            # 存前几个投影的渲染对比
            case_folder = os.path.join(out_folder, f"案例_{idx}")
            os.makedirs(case_folder, exist_ok=True)
            
            # 保存投影图像
            np.save(os.path.join(case_folder, "渲染视角0.npy"),
                    rendered_images[0].squeeze().detach().cpu().numpy())
            np.save(os.path.join(case_folder, "真实视角0.npy"),
                    gt_images[0].squeeze().detach().cpu().numpy())
            
            # 保存体数据的中心切片
            if vol_pred.dim() >= 4:
                mid_z = vol_pred.shape[-3] // 2
                mid_y = vol_pred.shape[-2] // 2
                mid_x = vol_pred.shape[-1] // 2
                
                axial_slice = vol_pred[..., mid_z, :, :].squeeze().detach().cpu().numpy()
                coronal_slice = vol_pred[..., :, mid_y, :].squeeze().detach().cpu().numpy()
                sagittal_slice = vol_pred[..., :, :, mid_x].squeeze().detach().cpu().numpy()
                
                np.save(os.path.join(case_folder, "轴向切片.npy"), axial_slice)
                np.save(os.path.join(case_folder, "冠状切片.npy"), coronal_slice)
                np.save(os.path.join(case_folder, "矢状切片.npy"), sagittal_slice)

    # 5. 计算平均指标
    mean_l12  = float(np.mean(all_l12))
    mean_ssim = float(np.mean(all_ssim))
    mean_tv   = float(np.mean(all_tv))

    # 返回最终统计
    return {
        "平均L1/L2损失": mean_l12,
        "平均SSIM": mean_ssim,
        "平均TV损失": mean_tv
    }

@torch.no_grad()
def main_eval(cfg_path, ckpt_path, data_path, device_idx=0, save_vis=0):
    """
    主评估函数，加载模型和数据集，执行评估
    
    参数:
        cfg_path: 配置文件路径
        ckpt_path: 模型检查点路径
        data_path: 测试数据集路径
        device_idx: GPU设备索引
        save_vis: 保存可视化结果的样本数量
    """
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

    # 加载cfg
    cfg = OmegaConf.load(cfg_path)

    # 加载模型
    model = GaussianSplatPredictor(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("已加载模型:", ckpt_path)

    # 构造测试集
    test_dataset = get_dataset(data_path=data_path, type="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg.data.num_workers, pin_memory=True)

    # 评估
    scores = evaluate_model(model, test_loader, cfg, device, save_vis=save_vis, out_folder="评估结果")
    print("评估结果:", scores)

    # 存json
    out_json = os.path.join("评估结果", "评分.json")
    os.makedirs("评估结果", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)
    print("已保存评估结果到", out_json)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data', type=str, required=True, help='测试数据集路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备索引')
    parser.add_argument('--vis', type=int, default=0, help='保存可视化结果的样本数量')
    
    args = parser.parse_args()
    
    main_eval(args.cfg, args.ckpt, args.data, device_idx=args.gpu, save_vis=args.vis)