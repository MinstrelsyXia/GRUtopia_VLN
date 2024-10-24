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
import shutil
from collections import defaultdict
from PIL import Image
from copy import deepcopy
import threading
from multiprocessing import Pipe, Process

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

import matplotlib.pyplot as plt

from vln.src.dataset.data_utils import VLNDataLoader
from vln.src.dataset.data_collector import dataCollector
# from vln.src.local_nav.global_topdown_map import GlobalTopdownMap


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)

from parser import process_args

def build_dataset():
    ''' Build dataset for VLN
    '''
    vln_config, sim_config = process_args()
    vln_datasets = {}
    if vln_config.split != "":
        vln_config.datasets.splits = [vln_config.split]
    for split in vln_config.datasets.splits:
        vln_datasets[split] = VLNDataLoader(vln_config, 
                                sim_config=sim_config,
                                split=split)
    camera_list = [x.name for x in sim_config.config.tasks[0].robots[0].sensor_params if x.enable]
    if 'sample_episodes' in vln_config.settings.mode:
        data_camera_list = vln_config.settings.sample_camera_list
    else:
        data_camera_list = None
    if vln_config.debug:
        vln_config.save_camera_list = ["pano_camera_0", "h1_pano_camera_debug"] # !!! for debugging
    else:
        vln_config.save_camera_list = camera_list
    vln_config.camera_list = camera_list
    
    return vln_datasets, vln_config, sim_config, data_camera_list


def vis_one_path(args, vln_envs):
    if args.path_id == -1:
        log.error("Please specify the path id")
        return
    # get the specific path
    vln_envs = vln_envs[args.split]
    data_item = vln_envs.init_one_path(args.path_id)
    env = vln_envs.env
    
    paths = data_item['reference_path']
    current_point = 0
    move_interval = 500 # move along the path every 5 seconds
    reset_robot = False
    
    if vln_config.windows_head:
        vln_envs.cam_occupancy_map_local.open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
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
        # warm_step =50000
        if i < warm_step:
            # give me some time to adjust the view position
            # let the robot stand still during the first warm steps.
            env_actions.append({'h1': {'stand_still': []}})
            obs = env.step(actions=env_actions)
            agent_action_state = {'finished': True}
            continue
        
        if i % 10 == 0:
            print(i)
            if vln_config.settings.check_and_reset_robot:
                topdown_map = vln_envs.GlobalTopdownMap(args, data_item['scan']) 
                freemap, camera_pose = vln_envs.get_surrounding_free_map(verbose=False)
                topdown_map.update_map(freemap, camera_pose, verbose=False) 

                reset_robot = vln_envs.check_and_reset_robot(cur_iter=i, update_freemap=False, verbose=vln_config.test_verbose)
                reset_flag = reset_robot
                if reset_flag:
                    # actions = {'h1': {'stand_still': []}}
                    vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
                    robot_current_pos = vln_envs.agents.get_world_pose()[0]
                    exe_path, node_type = vln_envs.bev.navigate_p2p(robot_current_pos, paths[current_point], verbose=vln_config.test_verbose)

                    
            if vln_config.windows_head:
                # show the topdown camera
                vln_envs.cam_occupancy_map_local.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0])

        if i % 100 == 0:
            print(i)
            if not reset_flag:
                if args.save_obs:
                    vln_envs.save_observations(camera_list=vln_config.save_camera_list, data_types=["rgba", "depth"], step_time=i)

                # move to next waypoint
                if current_point == 0:
                    log.info(f"===The robot starts navigating===")
                    log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")
                    # init BEVMapgru
                    agent_current_pose = vln_envs.agents.get_world_pose()[0]
                    vln_envs.init_BEVMap(robot_init_pose=agent_current_pose)
                    
                    if args.save_obs:
                        vln_envs.save_observations(camera_list=vln_config.save_camera_list, data_types=["rgba", "depth"], step_time=i)
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
                                vln_envs.save_observations(camera_list=vln_config.save_camera_list, data_types=["rgba", "depth"], step_time=i)
                            vln_envs.bev.step_time = i
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
                vln_envs.save_observations(camera_list=vln_config.save_camera_list, data_types=["rgba", "depth"], step_time=i)
        
            # update BEVMap every specific intervals
            vln_envs.bev.step_time = i
            vln_envs.update_occupancy_map(verbose=vln_config.test_verbose)
            
            # after BEVMap's update, determine whether the robot's path is blocked
            # if current_point > 0:
            #     agent_current_pose = vln_envs.agents.get_world_pose()[0]
            #     next_action_pose = exe_path[agent_action_state['current_index']+1] if len(exe_path)>agent_action_state['current_index']+1 else agent_action_state['current_point']
            #     if vln_envs.bev.is_collision(agent_current_pose, next_action_pose):
            #         log.info("===The robot's path is blocked. Replanning now.===")
            #         exe_path, _ = vln_envs.bev.navigate_p2p(agent_current_pose, paths[current_point], verbose=vln_config.test_verbose)
            #         actions = {'h1': {'move_along_path': [exe_path]}}
                            
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
        vln_envs.cam_occupancy_map_local.close_windows_head()

def keyboard_control(args, vln_envs):
    if args.path_id == -1:
        log.error("Please specify the path id")
        return
    # get the specific path
    data_item = vln_envs.init_one_path(args.path_id)
    env = vln_envs.split
    
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

def sample_episodes(args, vln_envs_all, data_camera_list):
    is_app_up = False
    for split, vln_envs in vln_envs_all.items():
        for scan in vln_envs.data:
            env = sample_episodes_single_scan(args, vln_envs_all, data_camera_list, split=split, scan=scan, is_app_up=is_app_up)
            is_app_up = True

        # env.simulation_app.close()
        # print('finish')
        # if vln_config.windows_head:
        #     # close the topdown camera
        #     vln_envs.cam_occupancy_map_local.close_windows_head()
    env.simulation_app.close()

def sample_episodes_single_scan(args, vln_envs_all, data_camera_list, split=None, scan=None, is_app_up=False):
    # Init the variables
    action_name = vln_config.settings.action
    topdown_maps = {}
    is_app_up = is_app_up
    scan = args.scan if scan is None else scan
    split = args.split if split is None else split
    vln_envs = vln_envs_all[split] 
    scan_data = vln_envs.data[scan] 

    process_path_id = []
    # scan_has_init = False
    scan_first_init = True
    # if scan_has_init:
    #     vln_envs.reload_scene(scan)
    for idx in range(len(scan_data)):
        data_item = scan_data[idx]        
        paths = data_item['reference_path']
        path_id = data_item['trajectory_id']

        if hasattr(args.settings, 'filter_stairs') and args.settings.filter_stairs:
            if 'stair' in data_item['instruction']['instruction_text']:
                continue

            different_height = False
            for path_idx in range(len(paths)-1):
                if abs(paths[path_idx+1][2] - paths[path_idx][2]) > 0.3:
                    different_height = True
                    break

            if different_height:
                continue

        if path_id in process_path_id:
            continue
        else:
            process_path_id.append(path_id)

        # 1. init Omni Env or Reset robot episode
        if args.test_verbose:
            args.log_image_dir = os.path.join(args.log_dir, "images", split, scan, str(data_item['trajectory_id'])) # Note that this will inflence the global arg value
            if not os.path.exists(args.log_image_dir):
                os.makedirs(args.log_image_dir)

        episode_path = os.path.join(args.sample_episode_dir, split, scan, f"id_{str(path_id)}")
        args.episode_path = episode_path
        if os.path.exists(episode_path):
            if args.settings.force_sample:
                log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled. Force to sample again.")
                # remove the previous sampled data
                shutil.rmtree(episode_path)
                os.makedirs(episode_path)
            else:
                log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled.")
                continue
        else:
            os.makedirs(episode_path)

        if scan_first_init:
            # !!!
            scan = "sT4fr6TAbpF"
            idx = 9
            path_id = 116
            # !!!
            # path_id = -1
            data_item = vln_envs.init_one_scan(scan, idx, init_omni_env=True, reset_scene=is_app_up, save_log=True, path_id=path_id) #TODO: change to global init
            is_app_up = True

        else:
            data_item = vln_envs.init_one_scan(scan, idx, init_omni_env=False, save_log=True, path_id=path_id)

        env = vln_envs.env
        current_point = 0
        total_points = []
        total_points.append([np.array(vln_envs.agent_init_pose), np.array(vln_envs.agent_init_rotation)])
        is_image_stacked = False
    
        if vln_config.windows_head:
            vln_envs.cam_occupancy_map_local.open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
        '''start simulation'''
        i = 0
        is_episode_success = False
        if scan_first_init:
            warm_step = 240 if args.headless else 2000
            # warm_step = 5 # !!! debug
            scan_first_init = False
        else:
            warm_step = 5
        move_step = warm_step
        # Note that warm_step should be the times of the oracle sample interval (now is set to 20)

        # init pipe for saving images
        parent_conn, child_conn = Pipe()
        data_collector = dataCollector(args, parent_conn, child_conn, split, scan, path_id)
        save_process = Process(target=data_collector.save_episode_data, args=())
        save_process.start()
        log.info(f"Save process starts.")

        if 'oracle' in action_name:
            action_info = {
                'current_step': 0,
                'topdown_camera_local': vln_envs.cam_occupancy_map_local,
                'topdown_camera_global': vln_envs.cam_occupancy_map_global
            }

            init_actions = {'h1': {action_name: [[paths[0]], action_info]}}
        else:
            init_actions = {'h1': {action_name: [[paths[0]]]}}

        start_time = time.time()
        while env.simulation_app.is_running():
            max_step = 500 if args.debug else args.settings.max_step
            if i >= max_step:
                # if i >= 2: # !!! debug
                log.warning(f"Scan: {scan}, Path_id: {path_id}. Exceed the maximum steps: {max_step}")
                break

            i += 1
            env_actions = []
            
            if i < warm_step:
                if 'oracle' in action_name:
                    init_actions['h1'][action_name][1]['current_step'] = i
                env_actions.append(init_actions)
                obs = env.step(actions=env_actions)
                
                if i % 50 == 0:
                    vln_envs.update_cam_occupancy_map_pose()
                    if vln_config.windows_head:
                        # show the topdown camera
                        vln_envs.cam_occupancy_map_local.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0], mode=args.windows_head_type)
                
                if (not scan_first_init) and i % 2 == 0:              
                    vln_envs.update_cam_occupancy_map_pose() # adjust the camera pose
                
                continue

            elif i == warm_step:
                # first warm up finished
                if scan not in topdown_maps:
                    topdown_map = vln_envs.TopdownMapGlobal(args, scan)
                    topdown_maps[scan] = topdown_map
                    freemap, camera_pose = vln_envs.get_global_free_map(verbose=args.test_verbose)
                    topdown_map.update_map(freemap, camera_pose, verbose=args.test_verbose)
                    log.info(f"====The global freemap has been initialized for {scan}====")
                else:
                    topdown_map = topdown_maps[scan]

                agent_action_state = {}
                agent_action_state["finished"] = True
                
                vln_envs.update_cam_occupancy_map_pose() # adjust the camera pose
                env_actions.append(init_actions)
                obs = env.step(actions=env_actions)
                continue
            
            if 'oracle' not in action_name and i % 50 == 0:
                reset_robot = vln_envs.check_and_reset_robot(cur_iter=i, update_freemap=False, verbose=vln_config.test_verbose)
                if reset_robot:
                    log.error(f"Scan: {scan}, Path_id: {path_id}. The robot is reset. Break this episode.")
                    break
            
            if args.test_verbose and args.windows_head:
                vln_envs.cam_occupancy_map_local.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0], mode=args.windows_head_type)
            
            if agent_action_state['finished']:
                if current_point == 0:
                    log.info(f"===The robot starts navigating===")
                if current_point < len(paths)-1:
                    log.info(f"===The robot is navigating to the {current_point+1}-th target place.===")

                    with open(args.episode_status_info_file, 'a') as f:
                        f.write(f"Current point number: {current_point}\n")
                    
                    freemap, camera_pose = vln_envs.get_global_free_map(verbose=args.test_verbose)
                    topdown_map.update_map(freemap, camera_pose, update_map=True, verbose=args.test_verbose)
                    robot_current_position = vln_envs.get_agent_pose()[0]
                    # exe_path = topdown_map.navigate_p2p(robot_current_position, paths[current_point+1], step_time=i, verbose=args.test_verbose, all_paths=paths) # TODO: check world to map coordinate
                    
                    exe_path = topdown_map.navigate_p2p(robot_current_position, paths[current_point+1], step_time=i, verbose=(args.test_verbose or args.save_path_planning), save_dir=os.path.join(args.sample_episode_dir, split, scan, f"id_{str(path_id)}")) 
                    if exe_path is None or len(exe_path) == 0:
                        # path planning fails
                        log.error(f"Scan: {scan}, Path_id: {path_id}. Path planning fails to find the path from the current point to the next point.")
                        break

                    if 'oracle' in action_name:
                        action_info.update({'current_step': i})
                        actions = {'h1': {action_name: [exe_path, action_info]}}
                    else:
                        actions = {'h1': {action_name: [exe_path]}}
                    
                    # total_points.extend(exe_path)
                    current_point += 1

                    if vln_config.windows_head:
                        # show the topdown camera
                        vln_envs.cam_occupancy_map_local.update_windows_head(robot_pos=vln_envs.agents.get_world_pose()[0], mode=args.windows_head_type)
            
            if 'oracle' in action_name:
                actions['h1'][action_name][1]['current_step'] = i

            env_actions.append(actions)
            if i % args.sample_episodes.step_interval == 0:
                # data_type = args.settings.camera_data_type
                # input data_type to retrival the high quality image 
                add_rgb_subframes = True
            else:
                # data_type = None
                add_rgb_subframes = False

            obs = env.step(actions=env_actions, add_rgb_subframes=add_rgb_subframes)
            # if 'oracle' not in action_name:
                # vln_envs.update_cam_occupancy_map_pose() # adjust the camera pose

            if 'oracle' in action_name:
                exe_point = obs[vln_envs.task_name][vln_envs.robot_name][action_name].get('exe_point', None)
                if exe_point is not None:
                    total_points.append(exe_point)
                    is_image_stacked = False
                    move_step = deepcopy(i)
            else:
                move_step = warm_step

            # stack images and information
            if (i-move_step) != 0 and (i-move_step) % (args.sample_episodes.step_interval-1) == 0:
                # Since oracle_move_path_controller moves to the next point every 5 steps, the image is fetched every 5+3 steps

                camera_pose_dict = vln_envs.get_camera_pose()
                data_collector.collect_and_send_data(i, env, camera_list=data_camera_list, camera_pose_dict=camera_pose_dict, add_rgb_subframes=True, finish_flag=False)

                is_image_stacked = True

            if args.test_verbose and args.save_obs and (i-move_step) != 0 and (i-move_step)%(args.sample_episodes.step_interval-1) == 0:
                vln_envs.save_observations(camera_list=data_camera_list, data_types=["rgba", "depth"], add_rgb_subframes=True, step_time=i)
                freemap, camera_pose = vln_envs.get_global_free_map(verbose=args.test_verbose)
                topdown_map.update_map(freemap, camera_pose, update_map=True, verbose=args.test_verbose)

            # get the action state
            if len(obs[vln_envs.task_name]) > 0:
                agent_action_state = obs[vln_envs.task_name][vln_envs.robot_name][action_name]
                # print(i)
            else:
                agent_action_state['finished'] = False
            
            if agent_action_state['finished']:
                if current_point == len(paths)-1 and is_image_stacked:
                    log.info("===The robot has achieved the final target.===")
                    log.info("===Break this episode.===")
                    is_episode_success = True
                    break
        
        with open(args.episode_status_info_file, 'a') as f:
            f.write(f"Episode finished: {is_episode_success}\n")

        end_time = time.time()
        total_time = (end_time - start_time)/60
        log.info(f"Total time for the [scan: {scan}] and [path_id: {path_id}] episode: {total_time:.2f} minutes")

    print('finish')
    parent_conn.send({'finish_flag': True})
    save_process.join() 

    return env

    # if vln_config.windows_head:
        # close the topdown camera
        # vln_envs.cam_occupancy_map_local.close_windows_head()
    
    # env.simulation_app.close()
        
def generate_pc(args, vln_envs_all, pose, depth_map_list, split=None, scan=None, assigned_path_id=None, is_app_up=False):
    # Init the variables
    action_name = vln_config.settings.action
    topdown_maps = {}
    is_app_up = is_app_up
    scan = args.scan if scan is None else scan
    split = args.split if split is None else split
    vln_envs = vln_envs_all[split] 
    scan_data = vln_envs.data[scan] 

    process_path_id = []
    # scan_has_init = False
    scan_first_init = True
    # if scan_has_init:
    #     vln_envs.reload_scene(scan)
    find_path_id = False
    for idx in range(len(scan_data)):
        data_item = scan_data[idx]        
        paths = data_item['reference_path']
        path_id = data_item['trajectory_id']
        if assigned_path_id is not None:
            # find the target path_id
            if path_id != assigned_path_id:
                continue
            else:
                find_path_id = True
                print(f'find target path_id {assigned_path_id}')

        if hasattr(args.settings, 'filter_stairs') and args.settings.filter_stairs:
            if 'stair' in data_item['instruction']['instruction_text']:
                continue

            different_height = False
            for path_idx in range(len(paths)-1):
                if abs(paths[path_idx+1][2] - paths[path_idx][2]) > 0.3:
                    different_height = True
                    break

            if different_height:
                continue

        if path_id in process_path_id:
            continue
        else:
            process_path_id.append(path_id)

        # 1. init Omni Env or Reset robot episode
        if args.test_verbose:
            args.log_image_dir = os.path.join(args.log_dir, "images", split, scan, str(data_item['trajectory_id'])) # Note that this will inflence the global arg value
            if not os.path.exists(args.log_image_dir):
                os.makedirs(args.log_image_dir)

        episode_path = os.path.join(args.sample_episode_dir, split, scan, f"id_{str(path_id)}")
        args.episode_path = episode_path
        if os.path.exists(episode_path):
            if args.settings.force_sample:
                log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled. Force to sample again.")
                # remove the previous sampled data
                shutil.rmtree(episode_path)
                os.makedirs(episode_path)
            else:
                log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled.")
                continue
        else:
            os.makedirs(episode_path)

        if scan_first_init:
            # !!!
            # scan = "sT4fr6TAbpF"
            # idx = 9
            # path_id = 116
            # !!!
            # path_id = -1
            data_item = vln_envs.init_one_scan(scan, idx, init_omni_env=True, reset_scene=is_app_up, save_log=True, path_id=path_id) #TODO: change to global init
            is_app_up = True

        else:
            data_item = vln_envs.init_one_scan(scan, idx, init_omni_env=False, save_log=True, path_id=path_id)

        env = vln_envs.env
        current_point = 0
        total_points = []
        total_points.append([np.array(vln_envs.agent_init_pose), np.array(vln_envs.agent_init_rotation)])
        is_image_stacked = False
    
        if vln_config.windows_head:
            vln_envs.cam_occupancy_map_local.open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
        '''start simulation'''
        i = 0
        step_interval = 50
        pc_interval = 20
        depth_length = 3
        current_depth = 0
        set_step = step_interval
        # 生成像素坐标的网格
        width, height = depth_map1.shape
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        points_2d = np.vstack((xx_flat, yy_flat)).T  # (N, 2)

        actions = {'h1': {'move_with_keyboard': []}}
        global_pcd = o3d.geometry.PointCloud()
        
        camera = vln_envs.robots.sensors['pano_camera_0']._camera
        if scan_first_init:
            warm_step = 240 if args.headless else 2000
            # warm_step = 5 # !!! debug
            scan_first_init = False
        else:
            warm_step = 5
        move_step = warm_step
        # Note that warm_step should be the times of the oracle sample interval (now is set to 20)
        
        init_actions = {'h1': {action_name: [[paths[0]]]}}

        start_time = time.time()
        while env.simulation_app.is_running():
            i += 1
            if i%step_interval == 0:
                camera.set_world_pose(pose[current_depth, :3], pose[current_depth, 3:])
                set_step = i
            if i%(set_step+pc_interval) == 0:
                depth_map = depth_map_list[current_depth]
                depth_map = depth_map.flatten()
                # try:
                points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map)
                points_3d_downsampled = downsample_pc(points_3d, 100)
                pcd_global = o3d.geometry.PointCloud()
                pcd_global.points = o3d.utility.Vector3dVector(points_3d_downsampled)
                global_pcd += pcd_global
                visualize_pc(pcd_global, False, save_path=f"logs/pc/{current_depth}.jpg")
                current_depth += 1
                if current_depth == depth_length:
                    current_depth = 0
                # except Exception:
                #     print("Error!")
                #     continue

        end_time = time.time()
        total_time = (end_time - start_time)/60
        log.info(f"Total time for the [scan: {scan}] and [path_id: {path_id}] episode: {total_time:.2f} minutes")

    print('finish')

    return env

    # if vln_config.windows_head:
        # close the topdown camera
        # vln_envs.cam_occupancy_map_local.close_windows_head()
    
    # env.simulation_app.close()
        

if __name__ == "__main__":
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset()
    
    if vln_config.settings.mode == "vis_one_path":
        vis_one_path(vln_config, vln_envs)
    elif vln_config.settings.mode == "keyboard_control":    
        keyboard_control(vln_config, vln_envs)
    elif vln_config.settings.mode == "llm_inference":
        # TODO
        llm_inference(vln_config, vln_envs)
    elif vln_config.settings.mode == "sample_episodes":
        sample_episodes(vln_config, vln_envs, data_camera_list)
    elif vln_config.settings.mode == "sample_episodes_scripts":
        sample_episodes_single_scan(vln_config, vln_envs, data_camera_list)
    elif vln_config.settings.mode == "sample_episodes_generate_pc":
        import open3d as o3d
        pose = np.loadtxt("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/poses.txt")
        depth_map1 = np.load("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/pano_camera_0_depth_step_279.npy")
        depth_map2 = np.load("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/pano_camera_0_depth_step_318.npy")
        depth_map3 = np.load("/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes_R2R_20240908/train/1LXtFkjw3qL/id_67/pano_camera_0_depth_step_357.npy")

        # 点云下采样函数
        def downsample_pc(pc, depth_sample_rate):
            shuffle_mask = np.arange(pc.shape[0])
            np.random.shuffle(shuffle_mask)
            shuffle_mask = shuffle_mask[::depth_sample_rate]
            pc = pc[shuffle_mask, :]
            return pc

        # 保存点云图像函数
        def save_point_cloud_image(pcd, save_path="point_cloud.jpg"):
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path)
            vis.destroy_window()

        # 点云可视化函数
        def visualize_pc(pcd, headless, save_path='pc.jpg'):
            if headless:
                save_point_cloud_image(pcd, save_path=save_path)
            else:
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([pcd, coordinate_frame])
                o3d.io.write_point_cloud("logs/pc/point_cloud.pcd", pcd)
                o3d.io.write_triangle_mesh("logs/pc/coordinate_frame.ply", coordinate_frame)
                
        depth_map_list = [depth_map1, depth_map2, depth_map3]
        generate_pc(vln_config, vln_envs, pose, depth_map_list, split='train', scan='1LXtFkjw3qL', assigned_path_id=67)