
import docker
from vln import PROJECT_ROOT_PATH
import os

client = docker.from_env()

def get_container_by_name(name):
    try:
        return client.containers.get(name)
    except docker.errors.NotFound:
        return None

def stop_if_exist(name):
    contianer = get_container_by_name(name)
    if contianer is not None:
        contianer.stop()

def run_container(
    name_prefix="test", 
    rank=0, 
    gpus=['0'],
    task_type='eval',
    image="w61_grutopia:v0.4"
):
    if task_type not in ['sample','eval']:
        print(f'task_type: {task_type} invalid!')
        return
    name = f"{name_prefix}_{rank:02}"
    stop_if_exist(name)
    
    WEBUI_HOST = os.environ.get('WEBUI_HOST','')
    CACHE_ROOT = os.environ.get('CACHE_ROOT','')

    command = "/isaac-sim/.venv/bin/python -u"
    if task_type == 'sample':
        command += f' vln/sample.py'
    elif task_type == 'eval':
        command += f' vln/eval.py'

    command +=" --rank {rank}"
    command +=f" >> rank.{rank:02}.log"
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
    return container

stop_if_exist("test_0")