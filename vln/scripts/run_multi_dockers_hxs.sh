#!/bin/bash

# 要运行的 GPU ID 列表 (例如：0,1,2)
GPU_LIST=("0" "1" "2" "3" "4" "5" "6")
# 每个 GPU 上要运行的实验数量
EXPERIMENTS_PER_GPU=2

# 容器基础名称
BASE_CONTAINER_NAME="w61_grutopia:v0.0"
# 镜像名称
IMAGE_NAME="w61_grutopia"

# 实验命令
EXPERIMENT_COMMANDS=(
    "source /root/.zshrc"
    "cd /isaac-sim"
    "conda activate isaaclab"
    "source setup_conda_env.sh"
    "export PYTHONPATH=\$(pwd):\$PYTHONPATH"
    "python src/standalone/env_example/collect-data-by-motion-planner-with-rendering.py"
)

# 创建日志目录
LOG_DIR="./logs/multi-dockers"
mkdir -p ${LOG_DIR}

# 遍历每个 GPU
for GPU in "${GPU_LIST[@]}"; do
    # 在该 GPU 上运行指定数量的实验
    for ((i=0; i<$EXPERIMENTS_PER_GPU; i++)); do
        # 为每个实验生成唯一的容器名称
        CONTAINER_NAME="${BASE_CONTAINER_NAME}-${GPU}-${i}-$(date +%s)"

        echo "Starting experiment on GPU ${GPU} with container name ${CONTAINER_NAME}"

        # 创建日志文件名
        LOG_FILE="${LOG_DIR}/${CONTAINER_NAME}.log"

        # 创建 Docker 容器并运行命令，输出重定向到日志文件
        docker run --name ${CONTAINER_NAME} \
            --entrypoint /bin/zsh \
            --runtime=nvidia \
            --gpus="device=${GPU}" \
            -e "ACCEPT_EULA=Y" \
            --rm \
            --network=bridge \
            --shm-size="32g" \
            -e "PRIVACY_CONSENT=Y" \
            -e "WEBUI_HOST=${WEBUI_HOST}" \
            -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
            -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
            -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
            -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
            -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
            -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
            -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
            -v ~/docker/isaac-sim/documents:/root/Documents:rw \
            -v ~/.cache/torch:/root/.cache/torch:rw \
            -v ~/main/Real2Sim-Real:/isaac-sim/src:rw \
            -v ~/main/resources:/ssd/hanxiaoshen/main/resources:rw \
            -v ~/main/SuGaR:/root/SuGaR:rw \
            -v ~/main/act-plus-plus/data:/ssd/hanxiaoshen/main/act-plus-plus/data:rw \
            -v ~/main/act-plus-plus:/isaac-sim/src/act_plus_plus:rw \
            -v ~/main/SuGaR:/ssd/hanxiaoshen/main/SuGaR:rw \
            -v /g0443_data/hanxiaoshen:/g0443_data/hanxiaoshen:rw \
            -v ~/main/gaussian-splatting:/root/gaussian-splatting:rw \
            ${IMAGE_NAME} \
            -c "$(printf "%s; " "${EXPERIMENT_COMMANDS[@]}")" > "${LOG_FILE}" 2>&1 &

        echo "Experiment started on GPU ${GPU} with container name ${CONTAINER_NAME}, logs at ${LOG_FILE}"
    done
done

echo "All experiments started."