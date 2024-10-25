#!/bin/bash

# 使用for循环启动多个Docker实例，每个实例传递不同的参数b
for i in {0..3}
do
    docker run --gpus "device=$i" -v $(pwd):/workspace -e HYDRA_FULL_ERROR=1 my_image_name \
    python vlmaps/application_my/isaac_robot.py episode_id=1
done
