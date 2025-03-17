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
from scripts.start_docker import(
    init_logger,
    stop_container_if_exist,
    start_contianer_with_retry,
)

client = docker.from_env()
logger = logging.getLogger('health_check_logger')
logger.setLevel(logging.INFO)


def get_database(name, readonly=True):
    lmdb_path = PROJECT_ROOT_PATH + f'/data/sample_episodes/{name}'
    if readonly:
        database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, readonly=True, lock=False)
    else:
        database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
    return database

def init_timestamp(name,device_map): 
    timestamp = msgpack_numpy.packb({"timestamp":time.time()}, use_bin_type=True)
    database_write = get_database(name,readonly=False)
    for rank, _ in device_map.items():
        with database_write.begin(write=True) as txn:
            txn.put(f"timestamp_rank_{rank}".encode(), timestamp)
    database_write.close()

def container_exist(name, rank):
    rank=int(rank)
    container_name = f"{name}_{rank:02}"
    try:
        container = client.containers.get(container_name)
    except docker.errors.NotFound:
        container = None
    return container is not None

def any_container_exist(name, device_map):
    for rank, _ in device_map.items():
        if container_exist(name,int(rank)):
            return True
    return False

def rank_finished(config, rank):
    rank=int(rank)
    task_type = config["task_type"]
    name = config["name"]
    ckpt_file_name = config["ckpt_to_load"].split('/')[-1]
    ckpt_name=f"{name}_{ckpt_file_name}"
    if task_type == "eval":
        prefix="eval_rank"
    elif task_type == "sample":
        prefix="sample_rank"
    database_read = get_database(name)
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

def rank_stuck(name, rank, threshold=60 * 60):
    rank=int(rank)
    database_read = get_database(name)
    info_key = f"timestamp_rank_{rank}"
    with database_read.begin() as txn:
        info_value = txn.get(info_key.encode())
    database_read.close()
    if info_value is None:
        return True
    info_value = msgpack_numpy.unpackb(info_value)
    timestamp = info_value['timestamp']
    time_diff = time.time() - timestamp
    if time_diff > threshold:
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
    init_logger(name,logger)
    device_map = config["device_map"]
    print(f"device_map:{device_map}")
    init_timestamp(name, device_map)
    
    time_interval = 10 * 60 # 默认10分钟

    while True:
        logger.info(f"开始休眠 {time_interval} s")
        time.sleep(time_interval)

        # 停止任务完成的 container
        for rank, _ in device_map.items():
            if not container_exist(name,rank):
                continue
            if rank_finished(config,rank):
                logger.info(f"[rank:{rank}] 任务执行完成，开始停止容器")
                stop_container_if_exist(config,rank)
        
        # 时间戳检查，如果有超过半小时未更新时间戳的容器，对其进行重启
        for rank, _ in device_map.items():
            if not container_exist(name,rank):
                continue
            if rank_stuck(name,rank):
                logger.info(f"[rank:{rank}] 长时间未响应，开始重启容器")
                stop_container_if_exist(config, rank)
                start_contianer_with_retry(config, rank, cfg_file)
                logger.info(f"[rank:{rank}] 容器重启完成")
        
        # 所有 container 都被 kill 的情况下，停止该任务
        time.sleep(10)
        if not any_container_exist(name,device_map):
            logger.info("未找到任何 container ,健康检查结束")
            sys.exit()