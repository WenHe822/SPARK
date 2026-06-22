#!/bin/bash

# 基础目录（每个子目录就是proj_dir）
base_dir="/Disk_16TB/zhouhaowei/code/network_GAS/TMI/data/TCIA_projections_512res_600numproj/剩下的8-20的病人"
ckpt_path="/Disk_16TB/zhouhaowei/code/network_GAS/TMI/experiments_out/2025-04-25/10-24-50/logs/1/model_latest.pth"
config_path="configs/default_config.yaml"

# 遍历每个病人文件夹（每个文件夹就是proj_dir）
for proj_dir in "$base_dir"/*/; do
    # 获取病人文件夹名称（去掉末尾的/）
    patient_name=$(basename "$proj_dir")
    
    # 构造输出路径（在病人文件夹内创建init_前缀文件）
    output_path="${proj_dir}init_${patient_name}.npy"
    
    # 执行推理命令
    echo "Processing: $proj_dir"
    echo "Output will be saved to: $output_path"
    
    python infference_save_pcd.py \
        --proj_dir "$proj_dir" \
        --ckpt_path "$ckpt_path" \
        --config_path "$config_path" \
        --output_path "$output_path"
    
    # 检查命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "Successfully processed $patient_name"
    else
        echo "Failed to process $patient_name"
        # 可以选择继续或退出
        # exit 1
    fi
done

echo "All patients processed."