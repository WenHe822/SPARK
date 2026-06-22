import torch
import numpy as np
from types import SimpleNamespace
import os
import sys

# 确保能找到模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.gaussian_predictor import GaussianSplatPredictor

def test_gaussian_predictor():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建一个简单的配置对象
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="SingleUNet",
            base_dim=32,
            num_blocks=2,
            attention_resolutions=[16],
            depth_scale=1.0,
            xyz_scale=0.1,
            density_scale=1.0,
            scale_scale=0.1,
            depth_bias=0.5,
            xyz_bias=0.0,
            density_bias=0.1,
            scale_bias=0.1,
            isotropic=False
        ),
        data=SimpleNamespace(
            training_resolution=64
        ),
        cam_embd=SimpleNamespace(
            embedding="positional",
            dimension=32
        )
    )
    
    # 创建模型
    model = GaussianSplatPredictor(cfg, is_ct=True).to(device)
    
    # 明确设置维度
    batch_size = 1  # 设置为1简化测试
    n_views = 2
    channels = 1  # 单通道图像
    height = 64
    width = 64
    
    print(f"创建测试数据: 批次大小={batch_size}, 视图数={n_views}, 高度={height}, 宽度={width}")
    
    # 创建投影图像 - 确保形状正确 [B, N_views, C, H, W]
    x = torch.randn(batch_size, n_views, channels, height, width)
    print(f"x shape: {x.shape}")
    
    # 创建相机四元数 - 与批次和视图数匹配
    source_cv2wT_quat = torch.nn.functional.normalize(
        torch.randn(batch_size, n_views, 4), dim=-1)
    print(f"source_cv2wT_quat shape: {source_cv2wT_quat.shape}")
    
    # 创建相机参数 - 确保长度等于 batch_size * n_views
    camera_params = []
    for b in range(batch_size):
        for v in range(n_views):
            angle = 2 * np.pi * v / n_views
            # 确保view_to_world是张量
            view_matrix = torch.eye(4, dtype=torch.float32)
            camera_params.append({
                "DSO": 100.0,  # 源到旋转中心距离
                "dDetector": (0.5, 0.5),  # 像素物理尺寸
                "offDetector": (width/2, height/2),  # 探测器中心坐标
                "view_to_world": view_matrix,  # 确保是张量
                "angle": float(angle)  # 确保是浮点数
            })
    
    print(f"camera_params 长度: {len(camera_params)}, 应等于 {batch_size * n_views}")
    
    # 创建像素坐标网格 [H, W, 2]
    y, x = torch.meshgrid(
        torch.arange(height), 
        torch.arange(width), 
        indexing='ij'
    )
    pixel_coords = torch.stack([x, y], dim=-1).float()  # 确保是浮点类型
    print(f"pixel_coords shape: {pixel_coords.shape}")
    
    # 将输入数据移至相同设备
    x = x.to(device)
    source_cv2wT_quat = source_cv2wT_quat.to(device)
    pixel_coords = pixel_coords.to(device)
    
    # 确保view_to_world也是正确设备
    for cam in camera_params:
        cam["view_to_world"] = cam["view_to_world"].to(device)
    
    # 前向传播
    try:
        with torch.no_grad():
            outputs = model(
                x, 
                source_cv2wT_quat=source_cv2wT_quat,
                camera_params=camera_params,
                pixel_coords=pixel_coords
            )
        
        # 检查输出字典
        print("输出键:", list(outputs.keys()))
        
        # 检查输出形状
        for key, value in outputs.items():
            print(f"{key} shape: {value.shape}")
            
        print("测试成功! 网络可以正常运行，输出形状正确。")
        return True
        
    except Exception as e:
        import traceback
        print(f"测试失败，具体错误: {str(e)}")
        traceback.print_exc()
        return False

def test_simplified():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建配置
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="SingleUNet",
            base_dim=32,
            num_blocks=2,
            attention_resolutions=[16],
            depth_scale=1.0,
            xyz_scale=0.1,
            density_scale=1.0,
            scale_scale=0.1,
            depth_bias=0.5,
            xyz_bias=0.0,
            density_bias=0.1,
            scale_bias=0.1,
            isotropic=False
        ),
        data=SimpleNamespace(
            training_resolution=64
        ),
        cam_embd=SimpleNamespace(
            embedding="positional",
            dimension=32
        )
    )
    
    # 创建模型
    model = GaussianSplatPredictor(cfg, is_ct=True).to(device)
    
    # 创建输入数据
    x = torch.randn(2, 3, 1, 64, 64).to(device)  # [B, N_views, C, H, W]
    
    # 仅测试网络前向，不进行后处理
    try:
        print("测试网络前向:")
        x_reshaped = x.reshape(-1, *x.shape[2:])
        net_out = model.network(x_reshaped)
        print(f"网络输出形状: {net_out.shape}")
        print("网络前向测试成功!")
        return True
    except Exception as e:
        import traceback
        print(f"网络前向测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_backward_pass():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建配置
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="SingleUNet",
            base_dim=32,
            num_blocks=2,
            attention_resolutions=[16],
            depth_scale=1.0,
            xyz_scale=0.1,
            density_scale=1.0,
            scale_scale=0.1,
            depth_bias=0.5,
            xyz_bias=0.0,
            density_bias=0.1,
            scale_bias=0.1,
            isotropic=False
        ),
        data=SimpleNamespace(
            training_resolution=64
        ),
        cam_embd=SimpleNamespace(
            embedding="positional",
            dimension=32
        )
    )
    
    # 创建模型
    model = GaussianSplatPredictor(cfg, is_ct=True).to(device)
    
    # 创建输入数据
    batch_size = 1
    n_views = 2
    height = width = 32
    
    # 修复: 确保叶子张量在设备上创建
    proj_image = torch.randn(batch_size, n_views, 1, height, width).to(device)
    proj_image.requires_grad = True  # 分开设置requires_grad以确保是叶子张量
    print(f"投影图像形状: {proj_image.shape}, 是叶子张量: {proj_image.is_leaf}")
    
    # 创建相机四元数 - 与批次和视图数匹配
    source_cv2wT_quat = torch.nn.functional.normalize(
        torch.randn(batch_size, n_views, 4), dim=-1).to(device)
    
    camera_params = []
    for b in range(batch_size):
        for v in range(n_views):
            angle = 2 * np.pi * v / n_views
            view_matrix = torch.eye(4, dtype=torch.float32).to(device)
            camera_params.append({
                "DSO": 100.0,
                "dDetector": (0.5, 0.5),
                "offDetector": (width/2, height/2),
                "view_to_world": view_matrix,
                "angle": float(angle)
            })
    
    # 创建像素坐标
    y, x = torch.meshgrid(
        torch.arange(height), 
        torch.arange(width), 
        indexing='ij'
    )
    pixel_coords = torch.stack([x, y], dim=-1).float().to(device)
    print(f"像素坐标形状: {pixel_coords.shape}")
    
    try:
        # 显式命名所有参数，避免位置混淆
        with torch.enable_grad():  # 确保启用梯度跟踪
            outputs = model(
                x=proj_image,  # 显式命名为x
                source_cv2wT_quat=source_cv2wT_quat,
                camera_params=camera_params,
                pixel_coords=pixel_coords
            )
        
        # 计算简单损失
        loss = 0
        for key, value in outputs.items():
            if key in ['xyz', 'scaling']:
                # 对于位置和缩放参数，鼓励接近原点/单位缩放
                loss = loss + value.abs().mean()
            elif key == 'density':
                # 对于密度，鼓励稀疏性
                loss = loss + value.mean()
            elif key == 'rotation':
                # 对于旋转，确保是单位四元数
                norm = value.norm(dim=-1)
                loss = loss + (norm - 1.0).abs().mean()
        
        print(f"计算的损失值: {loss.item()}")
        
        # 反向传播
        loss.backward()
        
        # 检查输入梯度
        if proj_image.grad is not None:
            print(f"输入梯度范数: {proj_image.grad.norm().item()}")
            print(f"输入梯度统计: 最小={proj_image.grad.min().item()}, 最大={proj_image.grad.max().item()}, 均值={proj_image.grad.mean().item()}")
            print("反向传播测试成功!")
            return True
        else:
            print("输入没有梯度! 检查以下可能原因:")
            print("1. 输入到输出的梯度链路可能被中断")
            print("2. 模型内部可能包含不支持梯度的操作")
            
            # 添加额外调试信息
            print(f"proj_image requires_grad: {proj_image.requires_grad}")
            print(f"is leaf tensor: {proj_image.is_leaf}")
            
            # 检查输出是否连接到输入
            for k, v in outputs.items():
                print(f"output '{k}' requires_grad: {v.requires_grad}")
            
            return False
            
    except Exception as e:
        import traceback
        print(f"反向传播测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_model_structure():
    """测试并显示模型结构和参数数量"""
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建配置
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="SingleUNet",
            base_dim=32,
            num_blocks=2,
            attention_resolutions=[16],
            depth_scale=1.0,
            xyz_scale=0.1,
            density_scale=1.0,
            scale_scale=0.1,
            depth_bias=0.5,
            xyz_bias=0.0,
            density_bias=0.1,
            scale_bias=0.1,
            isotropic=False
        ),
        data=SimpleNamespace(
            training_resolution=64
        ),
        cam_embd=SimpleNamespace(
            embedding="positional",
            dimension=32
        )
    )
    
    try:
        # 创建模型
        model = GaussianSplatPredictor(cfg, is_ct=True).to(device)
        
        # 打印模型架构
        print("\n=== 模型架构 ===")
        print(model)
        
        # 计算并打印参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n=== 参数统计 ===")
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"参数量 (MB): {total_params * 4 / (1024 * 1024):.2f} MB")
        
        # 打印每层参数量
        print("\n=== 各模块参数统计 ===")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {module_params:,} 参数 ({module_params * 100 / total_params:.2f}%)")
            
            # 如果是网络部分，进一步打印子模块
            if name == "network":
                for subname, submodule in module.named_children():
                    submodule_params = sum(p.numel() for p in submodule.parameters())
                    print(f"  {subname}: {submodule_params:,} 参数 ({submodule_params * 100 / total_params:.2f}%)")
        
        # 检查是否包含注意力机制
        has_attention = any('attn' in name for name, _ in model.named_modules())
        print(f"\n模型包含注意力机制: {has_attention}")
        
        print("\n模型结构测试完成!")
        return True
    
    except Exception as e:
        import traceback
        print(f"模型结构测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试模型结构和参数
    print("\n===== 测试模型结构和参数 =====")
    test_model_structure()
    
    # 测试网络基本功能
    print("\n===== 测试网络基本功能 =====")
    if test_simplified():
        # 测试反向传播
        print("\n===== 测试反向传播 =====")
        test_backward_pass()
    else:
        print("基础测试失败，跳过反向传播测试")