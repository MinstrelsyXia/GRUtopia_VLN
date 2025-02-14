#!/bin/bash

echo "开始关闭所有相关进程..."

# 关闭所有 launch.py 进程
echo "关闭 launch.py 进程..."
pkill -f "python.*vln/launch.py"

# 关闭所有 isaac sim 相关进程
echo "关闭 Isaac Sim 进程..."
pkill -f "omni.isaac"

# 重置所有 GPU
echo "重置 GPU 状态..."
nvidia-smi --gpu-reset 2>/dev/null

# 确保进程被完全终止
sleep 2

# 检查是否还有残留进程
remaining_processes=$(ps aux | grep -E "launch.py|omni.isaac" | grep -v grep)
if [ ! -z "$remaining_processes" ]; then
    echo "发现残留进程，强制终止..."
    # 使用 SIGKILL 强制终止
    pkill -9 -f "python.*vln/launch.py"
    pkill -9 -f "python.*vln/src/task/sample_one_scan.py"
    pkill -9 -f "omni.isaac"
fi

# 检查 GPU 状态
echo "当前 GPU 状态："
nvidia-smi

echo "清理完成！"