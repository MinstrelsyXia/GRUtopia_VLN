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


from vln.src.dataset.data_utils import VLNDataLoader
from vln.src.utils.utils import dict_to_namespace


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)


'''Init parser arguments'''
parser = argparse.ArgumentParser(description="Main function for VLN in GRUtopia")
parser.add_argument("--env", default="val_seen", type=str, help="The split of the dataset", choices=['train', 'val_seen', 'val_unseen'])
parser.add_argument("--path_id", default=5593, type=int, help="The number of path id") # 5593
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--test_verbose", action="store_true", default=False)
parser.add_argument("--wait", action="store_true", default=False)
parser.add_argument("--mode", type=str, default="vis_one_path", help="The mode of the program")
parser.add_argument("--sim_cfg_file", type=str, default="GRUtopia/vln/configs/sim_cfg.yaml")
parser.add_argument("--vln_cfg_file", type=str, default="GRUtopia/vln/configs/vln_cfg.yaml")
parser.add_argument("--save_obs", action="store_true", default=False)
args = parser.parse_args()

args.root_dir = ROOT_DIR
args.log_dir = os.path.join(ROOT_DIR, "logs")
args.log_image_dir = os.path.join(args.log_dir, "images")

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
    camera_list = [x.name for x in sim_config.config.tasks[0].robots[0].sensor_params]
    vln_config.camera_list = camera_list
    
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
    warm_step = 100 if args.headless else 1000

    # init the action
    action_name = vln_config.settings.action
    if action_name == 'move_along_path':
        actions = {'h1': {action_name: [paths]}}
        # vln_envs.init_BEVMap()
        current_point = 0
    else:
        actions = {'h1': {action_name: []}}

    while env.simulation_app.is_running():
        i += 1
        env_actions = []
        # env_actions.append(actions)
        
        if i < warm_step:
            # give me some time to adjust the view position
            # let the robot stand still during the first warm steps.
            env_actions.append({'h1': {'stand_still': []}})
            obs = env.step(actions=env_actions)
            continue

        if i % 100 == 0:
            print(i)
            # agent_current_pose = vln_envs.get_agent_pose()[0]
            # agent_bottom_z = vln_envs.get_robot_bottom_z()
            # vln_envs.cam_occupancy_map.generate_occupancy_map(agent_current_pose, agent_bottom_z, verbose=True) # !!!
            
            if args.save_obs:
                vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)

            # init BEVMap
            if current_point == 0:
                agent_current_pose = vln_envs.agents.get_world_pose()[0]
                vln_envs.init_BEVMap(robot_init_pose=agent_current_pose)
            # update BEVMap every specific intervals
            vln_envs.update_occupancy_map(verbose=True)
            vln_envs.bev.step_time = i

            # after BEVMap's update, determine whether the robot's path is blocked
            if current_point > 0:
                agent_current_pose = vln_envs.agents.get_world_pose()[0]
                next_action_pose = exe_path[agent_action_state['current_index']+1] if len(exe_path)>agent_action_state['current_index']+1 else agent_action_state['current_point']
                if vln_envs.bev.is_collision(agent_current_pose, next_action_pose):
                    log.info("===The robot's path is blocked. Replanning now.===")
                    exe_path, _ = vln_envs.bev.navigate_p2p(agent_current_pose, paths[current_point], verbose=True)

            # move to next waypoint
            if current_point == 0:
                log.info(f"===The robot starts navigating===")
                log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                if args.save_obs:
                    vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
                # vln_envs.update_occupancy_map(verbose=True)
                exe_path, node_type = vln_envs.bev.navigate_p2p(paths[current_point], paths[current_point+1], verbose=True)
                current_point += 1
                actions = {'h1': {'move_along_path': [exe_path]}}
            else:
                if agent_action_state['finished']:
                    log.info("***The robot has finished the action.***")
                    if current_point < len(paths)-1:
                        if args.save_obs:
                            vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
                        # vln_envs.update_occupancy_map(verbose=True)
                        robot_current_pos = vln_envs.agents.get_world_pose()[0]
                        exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point+1], verbose=True)
                        current_point += 1

                        actions = {'h1': {'move_along_path': [exe_path]}} 
                        log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                    else:
                        actions = {'h1': {'stand_still': []}}
                        log.info("===The robot has achieved the target place.===")
                else:
                    log.info("===The robot has not finished the action yet.===")
        
        env_actions.append(actions)
        obs = env.step(actions=env_actions)

        # get the action state
        agent_action_state = obs[vln_envs.task_name][vln_envs.robot_name][action_name]

    env.simulation_app.close()

def keyboard_control(args, vln_envs):
    if args.path_id == -1:
        log.error("Please specify the path id")
        return
    # get the specific path
    data_item = vln_envs.init_one_path(args.path_id)
    env = vln_envs.env
    
    i = 0
    actions = {'h1': {'move_with_keyboard': []}}
    while env.simulation_app.is_running():
        i += 1
        env_actions = []
        env_actions.append(actions)
        obs = env.step(actions=env_actions)

        if i % 100 == 0:
            # obs = env._runner.get_obs()
            # obs = env.get_observations(data_type=['rgba', 'depth', 'pointcloud', "normals"])
            # cur_obs = obs[task_name][robot_name]
            # is_fall = check_fall(agent, cur_obs, adjust=True, initial_pose=start_position, initial_rotation=start_rotation)
            # get_sensor_info(i, cur_obs, verbose=args.test_verbose)
            print(i)
    

if __name__ == "__main__":
    vln_envs = build_dataset(vln_config, sim_config)
    
    if args.mode == "vis_one_path":
        vis_one_path(args, vln_envs)
    elif args.mode == "keyboard_control":
        keyboard_control(args, vln_envs)