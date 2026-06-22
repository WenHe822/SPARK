import torch
import torch.nn as nn
import yaml
import os
import sys
from collections import OrderedDict, defaultdict
from scene.gaussian_predictor import GaussianSplatPredictor, SongUNet, SingleImageSongUNetPredictor

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def dict_to_namespace(d):
    """将嵌套字典转换为嵌套命名空间"""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def count_parameters(model):
    """计算模型的参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 按模块分类统计参数
    module_params = OrderedDict()
    for name, module in model.named_modules():
        if list(module.children()):  # 跳过有子模块的模块
            continue
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            module_params[name] = params
    
    # 按大小排序
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    
    # 按层类型统计
    layer_type_params = defaultdict(int)
    for name, module in model.named_modules():
        if list(module.children()):  # 跳过有子模块的模块
            continue
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            layer_type = module.__class__.__name__
            layer_type_params[layer_type] += params
    
    # 按网络组件统计
    component_params = defaultdict(int)
    for name, params in module_params.items():
        # 提取组件名称（第一级路径）
        component = name.split('.')[0]
        component_params[component] += params
    
    return total_params, trainable_params, sorted_modules, dict(layer_type_params), dict(component_params)

def format_params(num):
    """格式化参数数量，使其更易读"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def create_default_config():
    """创建默认配置"""
    cfg = SimpleNamespace()
    
    # 模型配置
    cfg.model = SimpleNamespace()
    cfg.model.name = "SingleUNet"
    cfg.model.base_dim = 128
    cfg.model.num_blocks = 4
    cfg.model.attention_resolutions = [16]
    cfg.model.depth_scale = 1.0
    cfg.model.depth_bias = 0.0
    cfg.model.xyz_scale = 0.1
    cfg.model.xyz_bias = 0.0
    cfg.model.density_scale = 1.0
    cfg.model.density_bias = 0.0
    cfg.model.scale_scale = 0.003
    cfg.model.scale_bias = 0.02
    cfg.model.xyz_range = 0.2
    cfg.model.isotropic = False
    cfg.model.use_zrange = True
    cfg.model.z_near = 0.1
    cfg.model.z_far = 2
    
    # 相机嵌入配置
    cfg.cam_embd = SimpleNamespace()
    cfg.cam_embd.embedding = "positional"
    cfg.cam_embd.dimension = 32
    
    # 数据配置
    cfg.data = SimpleNamespace()
    cfg.data.training_resolution = 128
    
    return cfg

def main():
    try:
        # 尝试加载配置文件
        config_path = "configs/default_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 转换为命名空间
        cfg = dict_to_namespace(config)
        
        # 确保cam_embd存在
        if not hasattr(cfg, 'cam_embd'):
            print("警告: 配置中缺少cam_embd，使用默认值")
            cfg.cam_embd = SimpleNamespace()
            cfg.cam_embd.embedding = "positional"
            cfg.cam_embd.dimension = 32
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("使用默认配置")
        cfg = create_default_config()
    
    # 创建模型
    model = GaussianSplatPredictor(cfg)
    
    # 计算参数量
    total_params, trainable_params, sorted_modules, layer_type_params, component_params = count_parameters(model)
    
    # 打印结果
    print(f"总参数量: {format_params(total_params)} ({total_params:,})")
    print(f"可训练参数量: {format_params(trainable_params)} ({trainable_params:,})")
    
    print("\n按网络组件统计参数量:")
    sorted_components = sorted(component_params.items(), key=lambda x: x[1], reverse=True)
    for name, params in sorted_components:
        print(f"{name}: {format_params(params)} ({params:,}) - {params/total_params*100:.2f}%")
    
    print("\n按层类型统计参数量:")
    sorted_layer_types = sorted(layer_type_params.items(), key=lambda x: x[1], reverse=True)
    for layer_type, params in sorted_layer_types:
        print(f"{layer_type}: {format_params(params)} ({params:,}) - {params/total_params*100:.2f}%")
    
    print("\n按模块参数量排序 (前20):")
    for i, (name, params) in enumerate(sorted_modules[:20]):
        print(f"{i+1}. {name}: {format_params(params)} ({params:,}) - {params/total_params*100:.2f}%")
    
    # 打印主要网络结构
    print("\n主要网络结构:")
    print(f"GaussianSplatPredictor -> SingleImageSongUNetPredictor -> SongUNet")
    
    # 打印UNet结构的关键参数
    if hasattr(model, 'network') and isinstance(model.network, SingleImageSongUNetPredictor):
        unet = model.network.encoder
        print(f"\nUNet结构参数:")
        print(f"- 基础通道数: {cfg.model.base_dim}")
        print(f"- 块数量: {cfg.model.num_blocks}")
        print(f"- 注意力分辨率: {cfg.model.attention_resolutions}")
        if hasattr(unet, 'enc'):
            print(f"- 编码器层数: {len([k for k in unet.enc.keys() if 'block' in k])}")
        if hasattr(unet, 'dec'):
            print(f"- 解码器层数: {len([k for k in unet.dec.keys() if 'block' in k])}")

if __name__ == "__main__":
    main() 