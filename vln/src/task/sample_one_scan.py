import argparse
import time
import os
import threading
import sys
from vln.src.v2.envs.discrete_sample_dagger import DiscreteSampleDaggerSingleScanEnv
from vln.src.v2.dataloader.sample import SamplePathKeyDataloader
from grutopia.core.config import SimulatorConfig
from vln.src.v2.util.common import load_scene_usd
from vln.src.v2.envs.env_factory import get_env_by_config
import numpy as np
import sys
import shutil
from vln.src.v2.util.common_log_util import common_logger as log
from vln.src.v2.util import common_log_util 
from vln import PROJECT_ROOT_PATH
import json
import traceback

def check_process_stuck(env:DiscreteSampleDaggerSingleScanEnv):
    index = 0
    while True:
        index+=1
        current_time = time.time()
        duration = round(current_time - env.timestamp,2)
        if  duration > 300:
            log.info("5分钟时间戳未更新,杀死进程")
            os.kill(os.getpid(), 9) 
        else:
            if index % 60 == 0:
                log.info(f"check_process_stuck 存活[{env.timestamp}]")
        sys.stdout.flush()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="rank",
    )
    parser.add_argument(
        "--scan",
        type=str,
        required=True,
        help="scan",
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="cfg_file",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="debug",
    )
    args = parser.parse_args()
    rank = args.rank
    scan = args.scan
    cfg_file = args.cfg_file
    project_path = PROJECT_ROOT_PATH
    cfg_file_path = f"{project_path}/{cfg_file}"
    if not os.path.exists(cfg_file_path):
        log.info(f"{cfg_file_path} not exist")
        sys.exit()
    with open(cfg_file_path, 'r') as file:
        config = json.load(file)
    
    headless = True
    split_data_types = ['train']
    project_path = PROJECT_ROOT_PATH
    base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
    mp3d_data_dir = f"{project_path}/data/Matterport3D/data/v1/scans" # !!! convert rel path to abs path for allowing rel texture path inside the usd
    name = config["name"]
    robot_name = config["robot_name"] # h1 / aliengo
    if robot_name == "aliengo":
        robot_offset = np.array([0.   , 0.   , 0.50])
    elif robot_name == 'h1':
        robot_offset = np.array([0.   , 0.   , 1.05])
    elif robot_name == 'jetbot':
        robot_offset = np.array([0.   , 0.   , 0.])
    common_log_util.init(name,rank)
    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    retry_list = config["retry_list"]
    sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_{robot_name}_eval.yaml'
    ckpt_to_load = config["ckpt_to_load"]
    sim_config = SimulatorConfig(sim_cfg_file)
    scene_asset_path = load_scene_usd(mp3d_data_dir, scan)

    # copy the cfg_file to the project_path
    shutil.copy(cfg_file_path, f"{project_path}/data/sample_episodes/{name}")
    shutil.copy(sim_cfg_file, f"{project_path}/data/sample_episodes/{name}")

    # thrid person camera
    if "save_third_person_image" in config and config["save_third_person_image"]:
        for sensor in sim_config.config_dict['tasks'][0]['robots'][0]['sensor_params']:
            if sensor['name'] == 'debug_camera':
                sensor['enable'] = True
                log.info(f"[third person camera] enable debug_camera for saving third person image")
                break

    dataloader=SamplePathKeyDataloader(
        base_data_dir,
        split_data_types,
        robot_offset,
        rank,
        lmdb_path,
        scan,
        retry_list,
    )
    sample_path_key_list = dataloader.sample_path_key_list
    if len(sample_path_key_list) == 0:
        log.info(f"[scan:{scan}] has no data to sample")
        os.kill(os.getpid(), 9) 
    else:
        log.info(f"[scan:{scan}] has {len(sample_path_key_list)} data to sample")
    path_zero = dataloader.path_key_data[sample_path_key_list[0]]
    start_position = path_zero['start_position']
    start_rotation = path_zero['start_rotation']


    env = get_env_by_config(
        config,
        sim_config,
        scene_asset_path,
        start_position,
        start_rotation,
        headless,
        dataloader,
    )

    if args.debug:
        env.sample()
    
    else:
        monitor_thread = threading.Thread(target=check_process_stuck, args=(env,))
        monitor_thread.start()

        try:
            log.info("env.sample()")
            env.sample()
            env.stop()
        except Exception as e:
            error_message = traceback.format_exc()
            log.error(error_message)
        except KeyboardInterrupt:
            log.info("Program stopped by user.")
        finally:
            log.info("Program terminated.")
            os.kill(os.getpid(), 9) 