#!/bin/bash

# 动态查找可用GPU
find_available_gpu() {
    # 使用nvidia-smi获取GPU使用情况，筛选出空闲的GPU
    # 这里的逻辑是：GPU内存使用率为0%且GPU利用率为0%的GPU被认为是空闲的
    available_gpus=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F',' '$2/$3*100 < 10 && $4 < 10 {print $1}' | head -1)
    
    if [ -z "$available_gpus" ]; then
        echo "No available GPU found!"
        exit 1
    fi
    
    echo $available_gpus
}

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 --dataset_root <dataset_root> --num_classes <num_classes> [other_options]"
    exit 1
fi

# 查找可用GPU
gpu_id=$(find_available_gpu)
echo "Using GPU: $gpu_id"

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$gpu_id

# 获取当前时间作为日志文件名的一部分
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="ldl_automation_${timestamp}.log"

# 后台运行ldl_automation.py，并将输出重定向到日志文件
python src/ldl_automation.py "$@" > "$log_file" 2>&1 &

# 输出进程ID和日志文件信息
echo "ldl_automation started in background with PID: $!"
echo "Log file: $log_file"
echo "To check the log: tail -f $log_file"
echo "To stop the process: kill $!"