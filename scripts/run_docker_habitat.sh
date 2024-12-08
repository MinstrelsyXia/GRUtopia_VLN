NAME=w61_grutopia_habitat
sudo docker run -d --name ${NAME} -it --rm --gpus='"device=0,1,2,3"' --network host \
     -e "ACCEPT_EULA=Y" \
     -e "PRIVACY_CONSENT=Y" \
     -e "WEBUI_HOST=${WEBUI_HOST}" \
     --shm-size 8G \
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
     -v /ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3_corrected:/isaac-sim/VLN/VLNCE/R2R_VLNCE_v1-3_corrected:rw \
     -v /ssd/share/Miniforge3-Linux-x86_64.sh:/root/Miniforge3-Linux-x86_64.sh:rw \
     -v /ssd/share/habitat-lab:/root/habitat-lab:rw \
     w61_grutopia_habitat:v0.0

# enter the docker
docker exec -it ${NAME} /bin/bash