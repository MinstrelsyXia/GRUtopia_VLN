import argparse
from vln import PROJECT_ROOT_PATH
import os
import sys
import json
import time
import docker
import logging

client = docker.from_env()
logger = logging.getLogger('health_check_logger')
logger.setLevel(logging.INFO)

def init_logger():
    log_dir = f"{PROJECT_ROOT_PATH}/logs/{name}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_name = "health_check.log" 
    file_handler = logging.FileHandler(f'{log_dir}/{file_name}')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def any_container_exist(name, device_map):
    for rank, _ in device_map.items():
        container_name = f"{name}_{rank:02}"
        container = client.containers.get(container_name)
        if container is not None:
            return True
    return False

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
    name = config["name"]
    print(f"name:{name}")
    init_logger(name)
    device_map = config["device_map"]
    print(f"device_map:{device_map}")

    time_interval = 5 * 60

    while True:
        time.sleep(time_interval)

        # 所有 container 都被 kill 的情况下，停止该任务
        if not any_container_exist(name,device_map):
            logger.info("未找到任何 container ,健康检查结束")
            sys.exit()

        # 检查任务进度，如果都完成，停止 container
        
        # 时间戳检查，如果有未更新，重启该 container

