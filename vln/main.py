# Author: w61
# Date: 2024.7.19
''' Main file for VLN in GRUtopia
'''
import os,sys
import gzip
import json
import math
import numpy as np
import argparse
import yaml
import time

from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
from grutopia.core.util.log import log

# from grutopia_extension.utils import get_stage_prim_paths

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


from vln.data_utils import VLNDataLoader
from vln.utils import dict_to_namespace

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)


'''Init parser arguments'''
parser = argparse.ArgumentParser(description="Main function for VLN in GRUtopia")
parser.add_argument("--env", default="val_seen", type=str, help="The split of the dataset", choices=['train', 'val_seen', 'val_unseen'])
parser.add_argument("--path_id", default=-1, type=int, help="The number of path id") # 5593
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--test_verbose", action="store_true", default=False)
parser.add_argument("--wait", action="store_true", default=False)
parser.add_argument("--mode", type=str, default="vis_one_path", help="The mode of the program", choices=['vis_one_path'])
parser.add_argument("--sim_cfg_file", type=str, default="GRUtopia/vln/configs/sim_cfg.yaml")
parser.add_argument("--vln_cfg_file", type=str, default="GRUtopia/vln/configs/vln_cfg.yaml")
args = parser.parse_args()

args.root_dir = ROOT_DIR

'''Init simulation config'''
sim_config = SimulatorConfig(args.sim_cfg_file)

'''Init VLN config'''
with open(args.vln_cfg_file, 'r') as f:
    vln_config = dict_to_namespace(yaml.load(f.read(), yaml.FullLoader))
# update args into vln_config
for key, value in vars(args).items():
    setattr(vln_config, key, value)

def build_dataset(vln_config, sim_config):
    ''' Build dataset for VLN
    '''
    vln_dataset = VLNDataLoader(vln_config, 
                                sim_config=sim_config)
    return vln_dataset


def vis_one_path(args, vln_envs):
    if args.path_id == -1:
        log.error("Please specify the path id")
        return
    # get the specific path
    data_item = vln_envs.init_one_path(args.path_id)
    env = vln_envs.env
    
    paths = data_item['reference_path']
    current_point = 0
    move_interval = 500 # move along the path every 5 seconds
    
    '''start simulation'''
    i = 0
    warm_step = 500
    # actions = {'h1': {'move_with_keyboard': []}}
    actions = {'h1': {'stand_still': []}}
    while env.simulation_app.is_running():
        i += 1
        env_actions = []
        env_actions.append(actions)
        obs = env.step(actions=env_actions)
        
        if i % 100 == 0:
            # obs = env.get_observations(data_type=['rgba', 'depth', 'pointcloud'])
            # cur_obs = obs[vln_envs.task_name][vln_envs.robot_name]
            camera_list = [x.name for x in sim_config.config.tasks[0].robots[0].sensor_params]
            obs = vln_envs.get_observations(camera_list, data_types=['rgba', 'depth', 'pointcloud'], save_imgs=True)
            print(i)
            
        if i <= warm_step:
            # give me some time to adjust the view position
            continue
        
        if i % move_interval == 0:
            if current_point < len(paths) - 1:
                current_point += 1
                # Set the agent's world_pose to the current point in paths
                vln_envs.move_along_path(paths[current_point-1], paths[current_point])
                log.info(f'Moving to next point: {current_point}')


    env.simulation_app.close()

if __name__ == "__main__":
    vln_envs = build_dataset(vln_config, sim_config)
    
    if args.mode == "vis_one_path":
        vis_one_path(args, vln_envs)