#!/bin/bash

# 容器基础名称（与创建时的BASE_CONTAINER_NAME保持一致）
BASE_CONTAINER_NAME="isaac-sim-xxy"

# 查找并杀掉所有符合条件的容器
docker ps -a --filter "name=${BASE_CONTAINER_NAME}" --format "{{.ID}} {{.Names}}" | while read CONTAINER_ID CONTAINER_NAME; do
    # 如果容器名称包含了更多信息（例如时间戳），就杀掉它
    if [[ $CONTAINER_NAME =~ ${BASE_CONTAINER_NAME}-[0-9]+-[0-9]+- ]]; then
        echo "Killing container ${CONTAINER_NAME} with ID ${CONTAINER_ID}"
        docker kill ${CONTAINER_ID} &
    else
        echo "Skipping container ${CONTAINER_NAME}"
    fi
done

echo "All relevant containers with base name '${BASE_CONTAINER_NAME}' have been killed."