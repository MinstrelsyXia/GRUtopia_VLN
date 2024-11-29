#!/bin/bash
echo "Debug in restart_env_docker.sh:"
echo "Number of arguments: $#"
echo "All arguments: $@"
echo "First argument: $1"


if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_idx>"
    exit 1
fi

GPU_IDX=$1
SCAN_FILE_DIR=$2

source /root/.bashrc
# 添加调试信息
echo "=== Debug Information ==="
echo "Current directory: $(pwd)"
echo "Python location: $(which python 2>/dev/null || echo 'python not found')"
echo "Python3 location: $(which python3 2>/dev/null || echo 'python3 not found')"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "======================="

# 设置工作目录
SCRIPT_PATH="vlmaps/application_my/isaac_robot_docker.py"
# SCAN_FILE_DIR="vlmaps/docker/multi_gpu_list"
LAST_SCAN_FILE="${SCAN_FILE_DIR}/last_scan_${GPU_IDX}.txt"
# GPU_LIST_FILE="vlmaps/docker/multi_gpu_list/${GPU_IDX}.txt"
GPU_LIST_FILE="${SCAN_FILE_DIR}/${GPU_IDX}.txt"
LOG_FILE="${SCAN_FILE_DIR}/python_output_${GPU_IDX}.log"

# 检查文件是否存在
echo "Checking files:"
echo "SCRIPT_PATH exists: $([ -f "$SCRIPT_PATH" ] && echo "Yes" || echo "No")"
echo "GPU_LIST_FILE exists: $([ -f "$GPU_LIST_FILE" ] && echo "Yes" || echo "No")"
echo "======================="


while :
do
    # 如果存在last_scan文件，读取上次的scan编号
    if [ -f "$LAST_SCAN_FILE" ]; then
        LAST_SCAN=$(cat $LAST_SCAN_FILE)
        echo "GPU ${GPU_IDX}: Resuming from scan: $LAST_SCAN"
        # 在后台运行 Python 脚本
        nohup python $SCRIPT_PATH resume_scan=$LAST_SCAN episode_file=$GPU_LIST_FILE last_scan_file=$LAST_SCAN_FILE > $LOG_FILE 2>&1 & 
        PID=$!
    else
        echo "GPU ${GPU_IDX}: Starting fresh run"
        nohup python $SCRIPT_PATH episode_file=$GPU_LIST_FILE last_scan_file=$LAST_SCAN_FILE > $LOG_FILE 2>&1 &
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