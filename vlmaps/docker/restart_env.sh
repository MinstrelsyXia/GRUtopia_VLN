if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_idx>"
    exit 1
fi

GPU_IDX=$1

# 设置工作目录
# WORK_DIR="/ssd/xiaxinyuan/code/w61-grutopia"
SCRIPT_PATH="vlmaps/application_my/isaac_robot_docker.py"
LAST_SCAN_FILE="vlmaps/docker/multi_gpu_list/last_scan_${GPU_IDX}.txt"  # 为每个GPU使用独立的last_scan文件
GPU_LIST_FILE="vlmaps/docker/multi_gpu_list/${GPU_IDX}.txt"

# cd $WORK_DIR

while :
do
    # 如果存在last_scan文件，读取上次的scan编号
    if [ -f "$LAST_SCAN_FILE" ]; then
        LAST_SCAN=$(cat $LAST_SCAN_FILE)
        echo "GPU ${GPU_IDX}: Resuming from scan: $LAST_SCAN"
        # 在后台运行 Python 脚本
        nohup python $SCRIPT_PATH config-name=vlmap_dataset_cfg resume_scan=$LAST_SCAN episode_file=$GPU_LIST_FILE last_scan_file=$LAST_SCAN_FILE > python_output_${GPU_IDX}.log 2>&1 & 
        PID=$!
    else
        echo "GPU ${GPU_IDX}: Starting fresh run"
        nohup python $SCRIPT_PATH config-name=vlmap_dataset_cfg episode_file=$GPU_LIST_FILE last_scan_file=$LAST_SCAN_FILE > python_output_${GPU_IDX}.log 2>&1 &
        PID=$!
    fi

    # 等待 Python 进程结束并获取退出状态
    wait $PID
    EXIT_STATUS=$?

    # 检查程序是否正常退出
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "GPU ${GPU_IDX}: Program completed successfully"
        
    else
        echo "GPU ${GPU_IDX}: Program exited abnormally (status: $EXIT_STATUS), restarting..."
        sleep 5  # 等待5秒后重启
    fi
done