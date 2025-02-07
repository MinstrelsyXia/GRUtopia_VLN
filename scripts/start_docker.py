import argparse
from vln import PROJECT_ROOT_PATH
import sys
import os
import docker
import json
from vln.split_data import split_data

client = docker.from_env()

def get_container_by_name(name):
    try:
        return client.containers.get(name)
    except docker.errors.NotFound:
        return None

def stop_if_exist(name):
    container = get_container_by_name(name)
    if container is not None:
        container.stop()
        print(f"stop container {name}")

def run_container(
    name_prefix="test", 
    rank=0, 
    gpus=['0'],
    image="w61_grutopia:v0.5",
    cfg_file="vln/configs/v2/eval.json",
    log_dir="logs",
):
    name = f"{name_prefix}_{rank:02}"
    stop_if_exist(name)
    print(f"begin to start container {name}")
    
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
        name=name,
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
    print(f"finish start container {name}")
    return container

def init(config):
    # 初始化日志
    name = config["name"]
    log_dir = f"logs/{name}/rank"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 如果没有分配任务，就进行分配
    lmdb_path = PROJECT_ROOT_PATH + f'/data/sample_episodes/{name}/sample_data.lmdb'
    if not os.path.exists(lmdb_path):
        split_data(config)
    return log_dir

def start_health_check(cfg_file):
    command = f"nohup python {PROJECT_ROOT_PATH}/scripts/health_check.py --cfg_file {cfg_file} &"
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
    log_dir = init(config)
    for rank, gpus in device_map.items():
        container = run_container(
            name_prefix=name, 
            rank=int(rank), 
            gpus=gpus,
            image=config["image"],
            cfg_file=cfg_file,
            log_dir=log_dir,
        )
    start_health_check(cfg_file)