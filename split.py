import os
import shutil
import random

def split_data(source_dir, train_dir, test_dir, train_ratio=0.8):
    # 获取所有子文件夹
    all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    # 打乱文件夹顺序
    random.shuffle(all_folders)
    
    # 计算训练和测试集的分割点
    split_index = int(len(all_folders) * train_ratio)
    
    # 划分训练集和测试集
    train_folders = all_folders[:split_index]
    test_folders = all_folders[split_index:]
    
    # 创建训练和测试目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 移动文件夹到训练和测试目录
    for folder in train_folders:
        shutil.move(os.path.join(source_dir, folder), os.path.join(train_dir, folder))
    
    for folder in test_folders:
        shutil.move(os.path.join(source_dir, folder), os.path.join(test_dir, folder))

# 定义源目录和目标目录
source_directory = '/Disk_16TB/zhouhaowei/code/network_GAS/TMI/data/LIDC_projections_processed'
train_directory = os.path.join(source_directory, 'train')
test_directory = os.path.join(source_directory, 'test')

# 执行数据划分
split_data(source_directory, train_directory, test_directory)
