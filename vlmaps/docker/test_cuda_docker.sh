#!/bin/bash

NAME="cuda_test_container"

# 运行容器
sudo docker run -it --rm \
    --name ${NAME} \
    --runtime=nvidia \
    --gpus all \
    --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ACCEPT_EULA=Y \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --network=host \
    --rm \
    nvcr.io/nvidia/isaac-sim:4.2.0 \
    /bin/bash -c '
    echo "=== 测试 CUDA 环境 ==="
    
    # 1. 检查 NVIDIA 驱动
    echo -e "\n1. NVIDIA-SMI 输出:"
    nvidia-smi
    
    # 2. 检查 CUDA 环境变量
    echo -e "\n2. CUDA 环境变量:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
    
    # 3. 安装 Python 和 PyTorch
    echo -e "\n3. 安装 Python 和 PyTorch:"
    apt-get update && apt-get install -y python3 python3-pip
    pip3 install torch
    
    # 4. 测试 PyTorch CUDA
    echo -e "\n4. PyTorch CUDA 测试:"
    python3 -c "
import torch
print('\nPyTorch 版本:', torch.__version__)
print('CUDA 是否可用:', torch.cuda.is_available())
print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
if torch.cuda.is_available():
    print('GPU 数量:', torch.cuda.device_count())
    print('当前设备:', torch.cuda.current_device())
    for i in range(torch.cuda.device_count()):
        print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\")
    # 测试 CUDA 计算
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('\nCUDA 矩阵乘法测试成功')
else:
    print('CUDA 不可用')
"
    '