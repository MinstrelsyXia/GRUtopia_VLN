NAME="xxy_v3.2"

export CACHE_ROOT=/ssd/xiaxinyuan/docker  # 设置缓存路径
export WEBUI_HOST=127.0.0.1  # 设置 Web UI 监听地址，默认为 127.0.0.1

sudo docker run -it --network host \
  --name ${NAME} \
  --gpus '"device=0"' \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e WEBUI_HOST=127.0.0.1 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
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
  xxy_v3.2
sudo docker exec -it ${NAME} /bin/bash