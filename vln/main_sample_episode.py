# Author: w61
# Date: 2024.08.29
''' Main file for sample episodes in GRUtopia (Support multiple envs)
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
import matplotlib.pyplot as plt


from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
from grutopia.core.util.log import log

from vln.src.dataset.data_utils_multi_env import VLNDataLoader
from vln.src.dataset.data_collector import dataCollector

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
                                split=split,
                                filter_same_trajectory=True)
    camera_list = [x.name for x in sim_config.config.tasks[0].robots[0].sensor_params if x.enable]
    if 'sample_episodes' in vln_config.settings.mode:
        data_camera_list = vln_config.settings.sample_camera_list
    else:
        data_camera_list = None
    vln_config.camera_list = camera_list
    
    return vln_datasets, vln_config, sim_config, data_camera_list

def sample_episodes(args, vln_envs_all, data_camera_list):
    is_app_up = False
    for split, vln_envs in vln_envs_all.items():
        for scan in vln_envs.data:
            env = sample_episodes_single_scan(args, vln_envs_all, data_camera_list, split=split, scan=scan, is_app_up=is_app_up)
            is_app_up = True

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
    env_num = vln_envs.env_num

    process_path_id = []
    scan_first_init = True

    path_id_list = [None] * env_num
    end_list = [False] * env_num
    success_list = [False] * env_num

    env_data = [[] for _ in range(env_num)]

    # allocate data
    for idx, data in enumerate(scan_data):
        env_idx = idx % env_num
        env_data[env_idx].append(data)

    for idx in range(len(scan_data)):
        data_item = scan_data[idx]        
        paths = data_item['reference_path']
        path_id = data_item['trajectory_id']

        if path_id in process_path_id:
            continue
        else:
            process_path_id.append(path_id)

        # 1. init Omni Env or Reset robot episode
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
                    topdown_map = vln_envs.GlobalTopdownMap(args, scan)
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
        

if __name__ == "__main__":
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset()
    
    if vln_config.settings.mode == "sample_episodes":
        sample_episodes(vln_config, vln_envs, data_camera_list)
    elif vln_config.settings.mode == "sample_episodes_scripts":
        sample_episodes_single_scan(vln_config, vln_envs, data_camera_list)