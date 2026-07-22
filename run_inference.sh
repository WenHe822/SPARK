#!/usr/bin/env bash
set -u

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 DATA_DIR CHECKPOINT [CONFIG]" >&2
    exit 2
fi

base_dir="$1"
ckpt_path="$2"
config_path="${3:-configs/default_config.yaml}"

if [[ ! -d "$base_dir" ]]; then
    echo "Data directory does not exist: $base_dir" >&2
    exit 2
fi

if [[ ! -f "$ckpt_path" ]]; then
    echo "Checkpoint does not exist: $ckpt_path" >&2
    exit 2
fi

# 遍历每个病人文件夹（每个文件夹就是proj_dir）
found_patient=false
for proj_dir in "$base_dir"/*/; do
    [[ -d "$proj_dir" ]] || continue
    found_patient=true

    # 获取病人文件夹名称（去掉末尾的/）
    patient_name=$(basename "$proj_dir")
    
    # 构造输出路径（在病人文件夹内创建init_前缀文件）
    output_path="${proj_dir}init_${patient_name}.npy"
    
    # 执行推理命令
    echo "Processing: $proj_dir"
    echo "Output will be saved to: $output_path"
    
    python inference_save_pcd.py \
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

if [[ "$found_patient" == false ]]; then
    echo "No patient directories found under: $base_dir" >&2
    exit 1
fi

echo "All patients processed."