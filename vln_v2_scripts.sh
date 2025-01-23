#!/bin/bash

# 定义配置文件路径
CONFIG_FILE="vln/configs/v2/eval.json"

# 显示使用方法
usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --split-data    分配任务"
    echo "  --start         启动任务"
    echo "  --progress      查看任务进展"
    echo "  --stop          停止任务"
    echo "  --help          显示此帮助信息"
    echo "  --config        指定配置文件路径 (默认: $CONFIG_FILE)"
}

# 如果没有参数，显示使用方法并退出
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# 解析命令行参数
while [ "$1" != "" ]; do
    case $1 in
        --split-data )    python vln/split_data.py --cfg_file "$CONFIG_FILE"
                         ;;
        --start )        python scripts/start_docker.py --cfg_file "$CONFIG_FILE"
                         ;;
        --progress )     python scripts/task_progress.py --cfg_file "$CONFIG_FILE"
                         ;;
        --stop )         python scripts/stop_docker.py --cfg_file "$CONFIG_FILE"
                         ;;
        --config )       shift
                         CONFIG_FILE=$1
                         ;;
        --help )         usage
                         exit
                         ;;
        * )              usage
                         exit 1
    esac
    shift
done