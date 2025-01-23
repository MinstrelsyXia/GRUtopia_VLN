import argparse
from vln import PROJECT_ROOT_PATH
import os
import sys
import json
import time
import docker
import logging
import lmdb
import msgpack_numpy
from vln.src.v2.util.eval import generate_eval_key

client = docker.from_env()
logger = logging.getLogger('health_check_logger')
logger.setLevel(logging.INFO)

def init_logger(name):
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

def container_exist(name, rank):
    container_name = f"{name}_{rank:02}"
    try:
        container = client.containers.get(container_name)
    except docker.errors.NotFound:
        container = None
    return container is not None

def stop_if_exist(name, rank):
    container_name = f"{name}_{rank:02}"
    try:
        container = client.containers.get(container_name)
    except docker.errors.NotFound:
        logger.info(f"[rank:{rank}] 容器[{container_name}]不存在")
        return
    if container is not None:
        container.stop()
        logger.info(f"[rank:{rank}] 容器[{container_name}]停止成功")

def any_container_exist(name, device_map):
    for rank, _ in device_map.items():
        if container_exist(name,rank):
            return True
    return False

def rank_finished(config, rank):
    task_type = config["task_type"]
    ckpt_file_name = config["ckpt_to_load"].split('/')[-1]
    ckpt_name=f"{name}_{ckpt_file_name}"
    if task_type == "eval":
        prefix="eval_rank"
    elif task_type == "sample":
        prefix="sample_rank"
    lmdb_path = PROJECT_ROOT_PATH + f'/data/sample_episodes/{name}'
    database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, readonly=True, lock=False)
    key = f"{prefix}_{rank}".encode()
    with database_read.begin() as txn:
        value = txn.get(key)
        if value is None:
            logger.info(f"[rank{rank}]获取抓取列表失败")
            return True
        value = msgpack_numpy.unpackb(value)
    for _, path_key_list in value.items():
        for path_key in path_key_list:
            if task_type == 'eval':
                info_key = generate_eval_key(ckpt_name,path_key)
            else:
                info_key = path_key.split('_')[0]
            with database_read.begin() as txn:
                info_value = txn.get(info_key.encode())
            if info_value is None:
                database_read.close()
                return False
    database_read.close()
    return True

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
    
    time_interval = 30 * 60 # 默认30分钟

    while True:
        logger.info(f"开始休眠 {time_interval} s")
        time.sleep(time_interval)

        # 所有 container 都被 kill 的情况下，停止该任务
        if not any_container_exist(name,device_map):
            logger.info("未找到任何 container ,健康检查结束")
            sys.exit()

        # 停止任务完成的 container
        for rank, _ in device_map.items():
            if not container_exist(name,rank):
                continue
            if rank_finished(config,rank):
                logger.info(f"[rank:{rank}] 任务执行完成，开始停止容器")
                stop_if_exist(name,rank)
        
        # 时间戳检查，如果有超过半小时未更新时间戳的容器，对其进行重启

