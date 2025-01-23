import argparse
from vln import PROJECT_ROOT_PATH
import sys
import os
import docker
import json
client = docker.from_env()

def get_container_by_name(name):
    try:
        return client.containers.get(name)
    except docker.errors.NotFound:
        print(f"container {name} not found !")
        return None
    
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
    for rank, gpus in device_map.items():
        contianer_name = f"{name}_{int(rank):02}"
        contianer = get_container_by_name(contianer_name)
        if contianer is not None:
            contianer.stop()
            print(f"stop container {contianer_name}")