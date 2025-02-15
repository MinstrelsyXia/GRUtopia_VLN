import argparse
import time
import os
import threading
import sys
from vln.src.v2.envs.discrete_eval import DiscreteEvalSingleScanEnv
from vln.src.v2.dataloader.eval import EvalPathKeyDataloader
from grutopia.core.config import SimulatorConfig
from vln.src.v2.util.common import load_scene_usd
from vln.src.v2.envs.env_factory import get_env_by_config
import numpy as np
import sys
from vln.src.v2.util.common_log_util import common_logger as log
from vln.src.v2.util import common_log_util 
import json
from vln import PROJECT_ROOT_PATH
import traceback

def check_process_stuck(env:DiscreteEvalSingleScanEnv):
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
        action="store_true",
        default=False
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
    split_data_types = ['val_unseen','val_seen']
    base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
    # base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_FSASub' # !!! This is for MLANet
    mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
    name = config["name"]
    robot_name = config["robot_name"] # h1 / aliengo
    if robot_name == "aliengo":
        robot_offset = np.array([0.   , 0.   , 0.50])
    else:
        robot_offset = np.array([0.   , 0.   , 1.05])
    common_log_util.init(name,rank)
    ckpt_file_name = config["ckpt_to_load"].split('/')[-1]
    ckpt_name=f"{name}_{ckpt_file_name}"
    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    retry_list = config["retry_list"]
    sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_{robot_name}_eval.yaml'
    ckpt_to_load = config["ckpt_to_load"]
    sim_config = SimulatorConfig(sim_cfg_file)
    
    # !!! For MLANet
    # sim_config.config.tasks[0].robots[0].sensor_params[2].size=(224,224) # pano_camera_0
    # sim_config.config_dict['tasks'][0]['robots'][0]['sensor_params'][2]['size']=(224,224)

    scene_asset_path = load_scene_usd(mp3d_data_dir, scan)
    dataloader=EvalPathKeyDataloader(
        base_data_dir,
        split_data_types,
        robot_offset,
        rank,
        ckpt_name,
        lmdb_path,
        scan,
        retry_list,
    )
    eval_path_key_list = dataloader.eval_path_key_list
    if len(eval_path_key_list) == 0:
        log.info(f"[scan:{scan}] has no data to eval")
        os.kill(os.getpid(), 9) 
    path_zero = dataloader.path_key_data[eval_path_key_list[0]]
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
        eval_cfg_file=config["eval_cfg_file"]
    )
    
    if args.debug:
        log.info("DEBUG MODE")
        env.eval()

    else:
        monitor_thread = threading.Thread(target=check_process_stuck, args=(env,))
        monitor_thread.start()

        try:
            log.info("env.eval()")
            env.eval()
            env.stop()
        except Exception as e:
            error_message = traceback.format_exc()
            log.error(error_message)
        except KeyboardInterrupt:
            log.info("Program stopped by user.")
        finally:
            log.info("Program terminated.")
            os.kill(os.getpid(), 9) 