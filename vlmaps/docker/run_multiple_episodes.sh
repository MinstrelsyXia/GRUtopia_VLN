#!/bin/bash

# 配置参数
NUM_GPUS=4
DATA_DIR="/ssd/xiaxinyuan/code/VLN/VLNCE/R2R_VLNCE_v1-3"
OUTPUT_DIR="multi_gpu_list" #/vlmaps/docker/multi_gpu_list
CACHE_ROOT="/ssd/xiaxinyuan/docker"
WEBUI_HOST="127.0.0.1"
NAME="xxy_v3.3"

# 创建日志目录
LOG_DIR="./.auto-run-docker"
mkdir -p ${LOG_DIR}

cd vlmaps/docker/
# 数据集拆分
python split_dataset.py --num_gpus $NUM_GPUS --data_dir $DATA_DIR --output_dir $OUTPUT_DIR

cd ../..
# GPU ID 列表
GPU_LIST=("0" "1" "2" "3")
EXPERIMENTS_PER_GPU=1
BASE_CONTAINER_NAME="isaac-sim-xxy"
IMAGE_NAME="xxy_new:3.4"

# 实验命令
# EXPERIMENT_COMMANDS=(
#     "cd /isaac-sim/GRUtopia"
#     "python3 vlmaps/application_my/isaac_robot_docker.py data_paths=docker vln_config.vln_cfg_file=vln/configs/vln_cfg_docker.yaml --episode_file=/vlmaps/docker/multi_gpu_list/\${GPU_IDX}.txt"
# )

# EXPERIMENT_COMMANDS=(
#     "cd /isaac-sim/GRUtopia"
#     "bash vlmaps/docker/restart_env.sh \${GPU}"
# )
# 遍历每个 GPU

EXPERIMENT_COMMANDS=(
    "source /root/.bashrc"
    "cd /isaac-sim/GRUtopia"
    "bash vlmaps/docker/restart_env_docker.sh  \${GPU}"
)

for GPU in "${GPU_LIST[@]}"; do
    for ((i=0; i<$EXPERIMENTS_PER_GPU; i++)); do
        CONTAINER_NAME="${BASE_CONTAINER_NAME}-${GPU}-${i}-$(date +%s)"
        LOG_FILE="${LOG_DIR}/${CONTAINER_NAME}.log"

        echo "Starting experiment on GPU ${GPU} with container name ${CONTAINER_NAME}"

        sudo docker run --name ${CONTAINER_NAME} \
            --entrypoint "/bin/bash" \
            --runtime=nvidia \
            --gpus="device=${GPU}" \
            -e "ACCEPT_EULA=Y" \
            -e "PRIVACY_CONSENT=Y" \
            -e "WEBUI_HOST=${WEBUI_HOST}" \
            -e "https_proxy=https://xiaxinyuan:OE6gf5X1v0JkSjKDOoUsVZhCdBbf0mdwfWO2kvWSlKj9L0Jwcfb9ff7snMkk@blsc-proxy.pjlab.org.cn:13128" \
            -e "http_proxy=https://xiaxinyuan:OE6gf5X1v0JkSjKDOoUsVZhCdBbf0mdwfWO2kvWSlKj9L0Jwcfb9ff7snMkk@blsc-proxy.pjlab.org.cn:13128" \
            --rm \
            --network=bridge \
            --shm-size="32g" \
            -v /ssd/xiaxinyuan/checkpoints/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz:/root/.cache/torch/hub/checkpoints/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz:ro \
            -v /home/xiaxinyuan/.cache/huggingface/hub/models--timm--vit_large_patch16_384.augreg_in21k_ft_in1k:/root/.cache/huggingface/hub/models--timm--vit_large_patch16_384.augreg_in21k_ft_in1k:ro \
            -v /ssd/xiaxinyuan/code/demo_e200.ckpt:/isaac-sim/GRUtopia/vlmaps/vlmaps/lseg/checkpoints/demo_e200.ckpt:ro \
            -v /home/xiaxinyuan/.cache/clip:/root/.cache/clip:ro \
            -v /ssd/share/Matterport3D:/isaac-sim/Matterport3D:ro \
            -v /ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3:/isaac-sim/VLN/VLNCE/R2R_VLNCE_v1-3:rw \
            -v /ssd/xiaxinyuan/code/w61-grutopia/logs_docker:/isaac-sim/GRUtopia/logs:rw \
            -v /ssd/xiaxinyuan/code/w61-grutopia:/isaac-sim/GRUtopia:rw \
            -v /ssd/xiaxinyuan/assets:/isaac-sim/GRUtopia/assets:ro \
            -v ${CACHE_ROOT}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
            -v ${CACHE_ROOT}/isaac-sim/cache/ov:/root/.cache/ov:rw \
            -v ${CACHE_ROOT}/isaac-sim/cache/pip:/root/.cache/pip:rw \
            -v ${CACHE_ROOT}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
            -v ${CACHE_ROOT}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
            -v ${CACHE_ROOT}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
            -v ${CACHE_ROOT}/isaac-sim/data:/root/.local/share/ov/data:rw \
            -v ${CACHE_ROOT}/isaac-sim/documents:/root/Documents:rw \
            -w /isaac-sim/GRUtopia \
            ${IMAGE_NAME} \
            -c "bash -i -c 'source /root/.bashrc && cd /isaac-sim/GRUtopia && bash vlmaps/docker/restart_env_docker.sh ${GPU}'" > "${LOG_FILE}" 2>&1 &
             


        echo "Experiment started on GPU ${GPU} with container name ${CONTAINER_NAME}, logs at ${LOG_FILE}"
    done
done

echo "All experiments started."
wait
# -c "$(printf "%s; " "${EXPERIMENT_COMMANDS[@]}")" > "${LOG_FILE}" 2>&1 &
# 
# bash -c "cd /isaac-sim/GRUtopia && bash vlmaps/docker/restart_env_docker.sh ${GPU}" 2>&1 | tee "${LOG_FILE}"
# -c "$(printf "%s; " "${EXPERIMENT_COMMANDS[@]}")" > "${LOG_FILE}" 2>&1 &