#!/bin/bash

# Get lmdb_pathId_dir from the script arguments
LMDB_JSON_PATH_ID_DIR=$1
echo "LMDB_JSON_PATH_ID_DIR set to ${LMDB_JSON_PATH_ID_DIR}"
SPLIT = $2

# 要运行的 GPU ID 列表 (例如：0,1,2)
# GPU_LIST=("0" "1" "2" "3" "4" "5" "6")
GPU_LIST=("0" "1")
# 每个 GPU 上要运行的实验数量
EXPERIMENTS_PER_GPU=1

# 容器基础名称
BASE_CONTAINER_NAME="w61_grutopia:v0.0"
# 镜像名称
IMAGE_NAME="w61_grutopia"

# 实验命令
EXPERIMENT_COMMANDS=(
    "cd /isaac-sim/GRUtopia"
    "bash vln/scripts/run_sample_episodes.sh"
)

# 创建日志目录
LOG_DIR="./logs/multi-dockers"
mkdir -p ${LOG_DIR}

# Directory with JSON files
JSON_DIR=${LMDB_JSON_PATH_ID_DIR}

# 遍历每个 GPU
for GPU in "${GPU_LIST[@]}"; do
    # 在该 GPU 上运行指定数量的实验
    for ((i=0; i<$EXPERIMENTS_PER_GPU; i++)); do
        # 为每个实验生成唯一的容器名称
        CONTAINER_NAME="${BASE_CONTAINER_NAME}-${GPU}-${i}-$(date +%s)"

        # Select JSON file for this docker
        JSON_FILE="${JSON_DIR}/scan_pathId_part_${i}.json"

        if [[ -f "$JSON_FILE" ]]; then
            echo "Starting experiment on GPU ${GPU} with container name ${CONTAINER_NAME}"

            # 创建日志文件名
            LOG_FILE="${LOG_DIR}/${CONTAINER_NAME}.log"

            # 创建 Docker 容器并运行命令，输出重定向到日志文件
            docker run -d --name ${CONTAINER_NAME} -it --rm --gpus="device=${GPU}" \
                --network host \
                -e "ACCEPT_EULA=Y" \
                -e "PRIVACY_CONSENT=Y" \
                -e "WEBUI_HOST=${WEBUI_HOST}" \
                -v ${PWD}:/isaac-sim/GRUtopia \
                -v ${CACHE_ROOT}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
                -v ${CACHE_ROOT}/isaac-sim/cache/ov:/root/.cache/ov:rw \
                -v ${CACHE_ROOT}/isaac-sim/cache/pip:/root/.cache/pip:rw \
                -v ${CACHE_ROOT}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
                -v ${CACHE_ROOT}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
                -v ${CACHE_ROOT}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
                -v ${CACHE_ROOT}/isaac-sim/data:/root/.local/share/ov/data:rw \
                -v ${CACHE_ROOT}/isaac-sim/documents:/root/Documents:rw \
                -v /ssd/share/Matterport3D:/isaac-sim/Matterport3D:rw \
                -v /ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3:/isaac-sim/VLN/VLNCE/R2R_VLNCE_v1-3:rw \
                ${IMAGE_NAME} \
                -c "$(printf "%s; " "${EXPERIMENT_COMMANDS[@]}") --lmdb_pathId_dir ${JSON_FILE} --docker_id ${i} --split ${SPLIT}" > "${LOG_FILE}" 2>&1 &

            echo "Experiment started on GPU ${GPU} with container name ${CONTAINER_NAME}, logs at ${LOG_FILE}"
        else
            echo "JSON file for docker $i not found, skipping."
        fi
    done
done

echo "All experiments started."