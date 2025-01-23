import argparse
from vln import PROJECT_ROOT_PATH
import os
import sys
import json
import time
import docker

client = docker.from_env()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="cfg_file",
    )
    args = parser.parse_args()
    cfg_file = args.cfg_file
    print(f"cfg_file:{cfg_file}")

    project_path = PROJECT_ROOT_PATH
    cfg_file_path = f"{project_path}/{cfg_file}"
    if not os.path.exists(cfg_file_path):
        print(f"{cfg_file_path} not exist")
        sys.exit()
    with open(cfg_file_path, 'r') as file:
        config = json.load(file)
    print(f"config:{config}")

    time_interval = 5 * 60

    while True:
        time.sleep(time_interval)

        # 确保 container 都存活，否则结束任务

        # 检查任务进度，如果都完成，停止 container
        
        # 时间戳检查，如果有未更新，重启该 container

