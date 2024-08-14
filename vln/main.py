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

# enable multiple gpus
# import isaacsim
# import carb.settings
# settings = carb.settings.get_settings()
# settings.set("/renderer/multiGPU/enabled", True)


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

from parser import process_args

def build_dataset():
    ''' Build dataset for VLN
    '''
    vln_config, sim_config = process_args()
    vln_datasets = {}
    for split in vln_config.datasets.splits:
        vln_datasets[split] = VLNDataLoader(vln_config, 
                                sim_config=sim_config,
                                split=split)
    camera_list = [x.name for x in sim_config.config.tasks[0].robots[0].sensor_params if x.enable]
    if vln_config.mode == 'sample_episodes':
        data_camera_list = vln_config.settings.sample_camera_list
    else:
        data_camera_list = None
    # camera_list = ["pano_camera_0"] # !!! for debugging
    vln_config.camera_list = camera_list
    
    return vln_datasets, vln_config, sim_config, data_camera_list


def vis_one_path(args, vln_envs):
    if args.path_id == -1:
        log.error("Please specify the path id")
        return
    # get the specific path
    vln_envs = vln_envs[args.env]
    data_item = vln_envs.init_one_path(args.path_id)
    env = vln_envs.env
    
    paths = data_item['reference_path']
    current_point = 0
    move_interval = 500 # move along the path every 5 seconds
    reset_robot = False
    
    if vln_config.windows_head:
        vln_envs.cam_occupancy_map.open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
    '''start simulation'''
    i = 0
    warm_step = 50 if args.headless else 500

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
        reset_flag = False
        env_actions = []
        # env_actions.append(actions)
        
        if i < warm_step:
            # give me some time to adjust the view position
            # let the robot stand still during the first warm steps.
            env_actions.append({'h1': {'stand_still': []}})
            obs = env.step(actions=env_actions)
            agent_action_state = {'finished': True}
            continue
        
        if i % 10 == 0:
            # print(i)
            if vln_config.settings.check_and_reset_robot:
                reset_robot = vln_envs.check_and_reset_robot(cur_iter=i, update_freemap=False, verbose=vln_config.test_verbose)
                reset_flag = reset_robot
                if reset_flag:
                    # actions = {'h1': {'stand_still': []}}
                    vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                    robot_current_pos = vln_envs.agents.get_world_pose()[0]
                    exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point], verbose=vln_config.test_verbose)

                    
            if vln_config.windows_head:
                # show the topdown camera
                vln_envs.cam_occupancy_map.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0])

        if i % 100 == 0:
            print(i)
            if not reset_flag:
                # if args.save_obs:
                #     vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)

                # move to next waypoint
                if current_point == 0:
                    log.info(f"===The robot starts navigating===")
                    log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                    # init BEVMapgru
                    agent_current_pose = vln_envs.agents.get_world_pose()[0]
                    vln_envs.init_BEVMap(robot_init_pose=agent_current_pose)
                    
                    if args.save_obs:
                        vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
                    vln_envs.bev.step_time = i
                    vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                    exe_path, node_type = vln_envs.bev.navigate_p2p(paths[current_point], paths[current_point+1], verbose=vln_config.test_verbose)
                    if node_type == 2:
                        log.info("Path planning fails to find the path from the current point to the next point.")
                    current_point += 1
                    actions = {'h1': {'move_along_path': [exe_path]}}
                else:
                    if agent_action_state['finished']:
                        log.info("***The robot has finished the action.***")
                        if current_point < len(paths)-1:
                            if args.save_obs:
                                vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
                            vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                            robot_current_pos = vln_envs.agents.get_world_pose()[0]
                            exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point+1], verbose=vln_config.test_verbose)
                            current_point += 1

                            actions = {'h1': {'move_along_path': [exe_path]}} 
                            log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                        else:
                            actions = {'h1': {'stand_still': []}}
                            log.info("===The robot has achieved the target place.===")
                    else:
                        log.info("===The robot has not finished the action yet.===")
        
        
        if i % 500 == 0 and not agent_action_state['finished']:
            if args.save_obs:
                vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
        
            # update BEVMap every specific intervals
            vln_envs.bev.step_time = i
            vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
            
            # after BEVMap's update, determine whether the robot's path is blocked
            if current_point > 0:
                agent_current_pose = vln_envs.agents.get_world_pose()[0]
                next_action_pose = exe_path[agent_action_state['current_index']+1] if len(exe_path)>agent_action_state['current_index']+1 else agent_action_state['current_point']
                if vln_envs.bev.is_collision(agent_current_pose, next_action_pose):
                    log.info("===The robot's path is blocked. Replanning now.===")
                    exe_path, _ = vln_envs.bev.navigate_p2p(agent_current_pose, paths[current_point], verbose=vln_config.test_verbose)
                    actions = {'h1': {'move_along_path': [exe_path]}}
                            
        env_actions.append(actions)
        obs = env.step(actions=env_actions)

        # get the action state
        if len(obs[vln_envs.task_name]) > 0:
            agent_action_state = obs[vln_envs.task_name][vln_envs.robot_name][action_name]
        else:
            agent_action_state['finished'] = False

    env.simulation_app.close()
    if vln_config.windows_head:
        # close the topdown camera
        vln_envs.cam_occupancy_map.close_windows_head()

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

def llm_inference(args, vln_envs):
    if args.path_id == -1:
        log.error("Please specify the path id")
        return
    # get the specific path
    data_item = vln_envs.init_one_path(args.path_id)
    env = vln_envs.env
    
    paths = data_item['reference_path']
    current_point = 0
    move_interval = 500 # move along the path every 5 seconds
    reset_robot = False
    
    if vln_config.windows_head:
        vln_envs.cam_occupancy_map.open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
    '''start simulation'''
    i = 0
    warm_step = 50 if args.headless else 500

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
        reset_flag = False
        env_actions = []
        # env_actions.append(actions)
        
        if i < warm_step:
            # give me some time to adjust the view position
            # let the robot stand still during the first warm steps.
            env_actions.append({'h1': {'stand_still': []}})
            obs = env.step(actions=env_actions)
            agent_action_state = {'finished': True}
            continue
        
        if i % 10 == 0:
            # print(i)
            if vln_config.settings.check_and_reset_robot:
                reset_robot = vln_envs.check_and_reset_robot(cur_iter=i, update_freemap=False, verbose=vln_config.test_verbose)
                reset_flag = reset_robot
                if reset_flag:
                    # actions = {'h1': {'stand_still': []}}
                        vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                        robot_current_pos = vln_envs.agents.get_world_pose()[0]
                        exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point], verbose=vln_config.test_verbose)

                    
            if vln_config.windows_head:
                # show the topdown camera
                vln_envs.cam_occupancy_map.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0])

        if i % 100 == 0:
            print(i)
            if not reset_flag:
                # if args.save_obs:
                #     vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)

                # move to next waypoint
                if current_point == 0:
                    log.info(f"===The robot starts navigating===")
                    log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                    # init BEVMapgru
                    agent_current_pose = vln_envs.agents.get_world_pose()[0]
                    vln_envs.init_BEVMap(robot_init_pose=agent_current_pose)
                    
                    if args.save_obs:
                        vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
                    vln_envs.bev.step_time = i
                    vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                    exe_path, node_type = vln_envs.bev.navigate_p2p(paths[current_point], paths[current_point+1], verbose=vln_config.test_verbose)
                    if node_type == 2:
                        log.info("Path planning fails to find the path from the current point to the next point.")
                    current_point += 1
                    actions = {'h1': {'move_along_path': [exe_path]}}
                else:
                    if agent_action_state['finished']:
                        log.info("***The robot has finished the action.***")
                        if current_point < len(paths)-1:
                            if args.save_obs:
                                vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
                            vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                            robot_current_pos = vln_envs.agents.get_world_pose()[0]
                            exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point+1], verbose=vln_config.test_verbose)
                            current_point += 1

                            actions = {'h1': {'move_along_path': [exe_path]}} 
                            log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                        else:
                            actions = {'h1': {'stand_still': []}}
                            log.info("===The robot has achieved the target place.===")
                    else:
                        log.info("===The robot has not finished the action yet.===")
        
        
        if i % 500 == 0 and not agent_action_state['finished']:
            if args.save_obs:
                vln_envs.save_observations(camera_list=vln_config.camera_list, data_types=["rgba", "depth"], step_time=i)
        
            # update BEVMap every specific intervals
            vln_envs.bev.step_time = i
            vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
            
            # after BEVMap's update, determine whether the robot's path is blocked
            if current_point > 0:
                agent_current_pose = vln_envs.agents.get_world_pose()[0]
                next_action_pose = exe_path[agent_action_state['current_index']+1] if len(exe_path)>agent_action_state['current_index']+1 else agent_action_state['current_point']
                if vln_envs.bev.is_collision(agent_current_pose, next_action_pose):
                    log.info("===The robot's path is blocked. Replanning now.===")
                    exe_path, _ = vln_envs.bev.navigate_p2p(agent_current_pose, paths[current_point], verbose=vln_config.test_verbose)
                    actions = {'h1': {'move_along_path': [exe_path]}}
                            
        env_actions.append(actions)
        obs = env.step(actions=env_actions)

        # get the action state
        if len(obs[vln_envs.task_name]) > 0:
            agent_action_state = obs[vln_envs.task_name][vln_envs.robot_name][action_name]
        else:
            agent_action_state['finished'] = False

    env.simulation_app.close()
    if vln_config.windows_head:
        # close the topdown camera
        vln_envs.cam_occupancy_map.close_windows_head()

def sample_episodes(args, vln_envs, data_camera_list):
    for scan in vln_envs.data:
        for idx in range(vln_envs[scan]):
            # 1. init Omni Env or Reset robot episode
            if idx == 0:
                data_item = vln_envs.init_one_scan(scan, idx, init_omni_env=True)
            else:
                data_item = vln_envs.init_one_scan(scan, idx, init_omni_env=False)

            env = vln_envs.env
            paths = data_item['reference_path']
            current_point = 0
            # total_points = [vln_envs.agent_init_pose, vln_envs.agent_init_rotatation] # [[x,y,z],[w,x,y,z]]
            total_points = []
        
            if vln_config.windows_head:
                vln_envs.cam_occupancy_map.open_windows_head(text_info=data_item['instruction']['instruction_text'])
        
            '''start simulation'''
            i = 0
            warm_step = 100 if args.headless else 500
            actions = {}

            while env.simulation_app.is_running():
                i += 1
                reset_flag = False
                env_actions = []
                
                if i < warm_step:
                    continue
                
                if i % 10 == 0:
                    if current_point == 0:
                        log.info(f"===The robot starts navigating===")
                        log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                        # init BEVMapgru
                        agent_current_pose = vln_envs.agents.get_world_pose()[0]
                        vln_envs.init_BEVMap(robot_init_pose=agent_current_pose)
                        
                        if args.save_obs:
                            vln_envs.save_observations(camera_list=data_camera_list, data_types=["rgba", "depth"], step_time=i)
                        vln_envs.bev.step_time = i
                        vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                        exe_path, node_type = vln_envs.bev.navigate_p2p(paths[current_point], paths[current_point+1], verbose=vln_config.test_verbose) # TODO
                        
                        total_points.extend(exe_path)

                    else:
                        vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                        robot_current_pos = vln_envs.agents.get_world_pose()[0]
                        exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point], verbose=vln_config.test_verbose)
                    
                            
                    if vln_config.windows_head:
                        # show the topdown camera
                        vln_envs.cam_occupancy_map.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0])

                                    
                env_actions.append(actions)
                obs = env.step(actions=env_actions)


            env.simulation_app.close()
            if vln_config.windows_head:
                # close the topdown camera
                vln_envs.cam_occupancy_map.close_windows_head()
        

if __name__ == "__main__":
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset()
    
    if vln_config.mode == "vis_one_path":
        vis_one_path(vln_config, vln_envs)
    elif vln_config.mode == "keyboard_control":
        keyboard_control(vln_config, vln_envs)
    elif vln_config.mode == "llm_inference":
        # TODO
        llm_inference(vln_config, vln_envs)
    elif vln_config.mode == "sample_episodes":
        sample_episodes(vln_config, vln_envs, data_camera_list)