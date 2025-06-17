# 找到restart_env.sh的PID
PIDS=$(pgrep -f restart_env_docker.sh)

# 终止每个restart_env.sh的进程及其子进程
for PID in $PIDS; do
    # 列出所有子进程（包括间接子进程）
    pkill -9 -P $PID
    # 终止主进程
    kill -9 $PID
done

# 以防万一，再检查是否有isaac_robot_docker.py进程漏网
pkill -9 -f isaac_robot_docker.py