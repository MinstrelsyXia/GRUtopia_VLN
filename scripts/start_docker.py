import argparse
from vln import PROJECT_ROOT_PATH
import sys
import os
import docker
import json
from vln.split_data import split_data
from docker.client import DockerClient
import traceback
import logging
import time

client = docker.from_env()
logger = logging.getLogger('health_check_logger')
logger.setLevel(logging.INFO)

def init_logger(name,logger):
    log_dir = f"{PROJECT_ROOT_PATH}/logs/{name}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_name = "health_check.log" 
    file_handler = logging.FileHandler(f'{log_dir}/{file_name}')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def get_container_by_name(name):
    try:
        return client.containers.get(name)
    except docker.errors.NotFound:
        return None

def stop_container_if_exist(
    config,
    rank
):
    rank=int(rank)
    name=config['name']
    container_name = f"{name}_{rank:02}"
    container = get_container_by_name(container_name)
    if container is None:
        return
    retry_count=3
    index=0
    while index < retry_count:
        index +=1
        try:
            container.stop()
            logger.info(f"[rank:{rank}] 容器停止成功")
            return
        except Exception as e:
            error_message = traceback.format_exc()
            logger.info(error_message)
            logger.info(f"[rank:{rank}] 容器停止失败,休眠 10 秒后重试,剩余重试次数:{retry_count - index}")
            time.sleep(10)

def start_contianer_with_retry(
    config,
    rank,
    cfg_file,
):
    retry_count=3
    index=0
    while index < retry_count:
        index +=1
        try:
            start_container(config,rank,cfg_file)
            logger.info(f"[rank:{rank}] 容器启动成功")
            return
        except Exception as e:
            error_message = traceback.format_exc()
            logger.info(error_message)
            logger.info(f"[rank:{rank}] 容器启动失败,休眠 10 秒后重试,剩余重试次数:{retry_count - index}")
            time.sleep(10)

def start_container(
    config,
    rank,
    cfg_file,
):  
    rank = int(rank)
    name_prefix = config['name']
    log_dir = f"logs/{name_prefix}/rank"
    device_map = config["device_map"]
    gpus = device_map[str(rank)]
    image=config["image"]
    container_name = f"{name_prefix}_{rank:02}"
    logger.info(f"begin to start container {container_name}")
    
    WEBUI_HOST = os.environ.get('WEBUI_HOST','')
    CACHE_ROOT = os.environ.get('CACHE_ROOT','')

    command = "source /isaac-sim/.venv/bin/activate && source /root/.bashrc"
    command += " && python -u"
    command += f' vln/launch.py'
    command +=f" --rank {rank}"
    command +=f" --cfg_file {cfg_file}"
    
    command +=f" >> {log_dir}/rank.{rank:02}.log"
    command +=" && tail -f /dev/null"

    container = client.containers.run(
        detach=True,
        name=container_name,
        tty=True,
        stdin_open=True,
        auto_remove=True,
        device_requests=[docker.types.DeviceRequest(device_ids=gpus,capabilities=[['gpu']])],
        network_mode="host",
        environment={
            "ACCEPT_EULA":"Y",
            "PRIVACY_CONSENT":"Y",
            "WEBUI_HOST":WEBUI_HOST,
        },
        shm_size="8G",
        entrypoint=[ "/bin/bash", "-l", "-c" ],
        command=[command],
        image=image,
        working_dir="/isaac-sim/GRUtopia",
        volumes=[
            f"{PROJECT_ROOT_PATH}:/isaac-sim/GRUtopia",
            f"{CACHE_ROOT}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw",
            f"{CACHE_ROOT}/isaac-sim/cache/ov:/root/.cache/ov:rw",
            f"{CACHE_ROOT}/isaac-sim/cache/pip:/root/.cache/pip:rw",
            f"{CACHE_ROOT}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw",
            f"{CACHE_ROOT}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw",
            f"{CACHE_ROOT}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw",
            f"{CACHE_ROOT}/isaac-sim/data:/root/.local/share/ov/data:rw",
            f"{CACHE_ROOT}/isaac-sim/documents:/root/Documents:rw",
            '/ssd/share/Matterport3D:/isaac-sim/Matterport3D:rw',
            '/ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3:/isaac-sim/VLN/VLNCE/R2R_VLNCE_v1-3:rw',
            '/ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3_corrected:/isaac-sim/VLN/VLNCE/R2R_VLNCE_v1-3_corrected:rw',
        ],
    )
    logger.info(f"finish start container {container_name}")
    return container

def split_data_if_needed(config):
    # 初始化日志
    name = config["name"]
    log_dir = f"logs/{name}/rank"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 如果没有分配任务，就进行分配
    lmdb_path = PROJECT_ROOT_PATH + f'/data/sample_episodes/{name}/sample_data.lmdb'
    if not os.path.exists(lmdb_path):
        split_data(config)

def start_health_check(cfg_file):
    command = f"nohup python {PROJECT_ROOT_PATH}/scripts/health_check.py --cfg_file {cfg_file} > /dev/null 2>&1 &"
    os.system(command)
    print('健康检查进程已启动')

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
    device_map = config["device_map"]
    name = config["name"]
    init_logger(name,logger)
    split_data_if_needed(config)
    for rank, _ in device_map.items():
        stop_container_if_exist(config,rank)
        start_contianer_with_retry(config, rank, cfg_file)
    start_health_check(cfg_file)