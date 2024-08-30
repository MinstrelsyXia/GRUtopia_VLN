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

def update_env_actions(action_name, paths_list, path_idx=-1):
    env_actions = []
    env_num = len(paths_list)
    for env_idx in range(env_num):
        if path_idx != -1:
            init_path = paths_list[env_idx][path_idx]
        else:
            init_path = paths_list[env_idx]
        init_actions = {'h1': {action_name: [[init_path]]}}
        env_actions.append(init_actions)
    return env_actions

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
    is_app_up = is_app_up
    scan = args.scan if scan is None else scan
    split = args.split if split is None else split
    vln_envs = vln_envs_all[split] 
    scan_data = vln_envs.data[scan] 
    env_num = vln_envs.env_num

    process_path_id = []
    scan_first_init = True

    vln_envs.allocate_data(scan)
    stand_still_action = {'h1': {'stand_still': []}}

    while not all(vln_envs.all_episodes_end_list):
        '''1. Get the data'''
        data_item_list, path_id_list, paths_list = vln_envs.get_next_data()
        '''2. Init episode saving dir'''
        args.episode_path_list = []
        for env_idx in range(env_num):
            if vln_envs.end_list[env_idx]:
                continue
            path_id = path_id_list[env_idx]
            episode_path = os.path.join(args.sample_episode_dir, split, scan, f"id_{str(path_id)}")
            args.episode_path_list.append(episode_path)
            if os.path.exists(episode_path):
                if args.settings.force_sample:
                    log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled. Force to sample again.")
                    # remove the previous sampled data
                    shutil.rmtree(episode_path)
                    os.makedirs(episode_path)
                else:
                    log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled. Pass.")
                    continue
            else:
                os.makedirs(episode_path)

        if scan_first_init:
            data_item = vln_envs.init_multiple_episodes(scan, data_item_list, init_omni_env=True, save_log=True)
            is_app_up = True
        else:
            data_item = vln_envs.init_multiple_episodes(scan, data_item_list, init_omni_env=False, save_log=True)

        env = vln_envs.env
        env_action_finish_states = [False] * env_num
        nav_point_list = [0] * env_num
        topdown_maps = [None] * env_num
    
        if vln_config.windows_head:
            vln_envs.cam_occupancy_map_local_list[0].open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
        '''start simulation'''
        i = 0
        is_episode_success = False
        if scan_first_init:
            warm_step = 240 if args.headless else 2000
            scan_first_init = False
        else:
            warm_step = 5
        move_step = warm_step

        # init pipe for saving images
        parent_conn, child_conn = Pipe()
        data_collector = dataCollector(args, parent_conn, child_conn, split, scan, path_id_list)
        save_process = Process(target=data_collector.save_episode_data, args=())
        save_process.start()
        log.info(f"Save process starts.")

        if 'oracle' in action_name:
            # TODO
            action_info = {
                'current_step': 0,
                'topdown_camera_local': vln_envs.cam_occupancy_map_local,
                'topdown_camera_global': vln_envs.cam_occupancy_map_global
            }

            init_actions = {'h1': {action_name: [[paths[0]], action_info]}}
        else:
            env_actions = update_env_actions(action_name, paths_list, path_idx=0)
            env_actions = vln_envs.calc_env_action_offset(env_actions,action_name)

        start_time = time.time()
        
        '''(0) Enter the env loop'''
        while not all(vln_envs.end_list) and env.simulation_app.is_running():
            max_step = 500 if args.debug else args.settings.max_step
            if i >= max_step:
                for env_idx, env_finish_status in enumerate(vln_envs.end_list):
                    if not env_finish_status:
                        log.warning(f"Scan: {scan}, Path_id: {path_id_list[env_idx]}. Exceed the maximum steps: {max_step}")
                break

            i += 1
            
            '''(1) warm up process'''
            if i < warm_step:
                if 'oracle' in action_name:
                    init_actions['h1'][action_name][1]['current_step'] = i
                # env_actions = update_env_actions(action_name, paths_list, path_idx=0)
                # env_actions = vln_envs.calc_env_action_offset(env_actions,action_name)
                obs = env.step(actions=env_actions)
                
                if i % 50 == 0:
                    if vln_config.windows_head:
                        # show the topdown camera
                        vln_envs.cam_occupancy_map_local_list[0].update_windows_head(robot_pos=vln_envs.isaac_robots[0].get_world_pose()[0], mode=args.windows_head_type)
                        # log the FPS
                        current_time = time.time()
                        log.info(f"Current step: {i}. FPS: {i/(current_time-start_time):.2f}")
                
                
                continue

            elif i == warm_step:
                # first warm up finished
                for env_idx in range(env_num):
                    topdown_map = vln_envs.GlobalTopdownMap(args, scan)
                    freemap, camera_pose = vln_envs.get_global_free_map_single(env_idx, verbose=args.test_verbose)
                    topdown_map.update_map(freemap, camera_pose, verbose=args.test_verbose, env_idx=env_idx)
                    topdown_maps[env_idx] = topdown_map
                    log.info(f"====The global freemap has been initialized for Path_id {path_id_list[env_idx]}====")

                env_action_finish_states = [True] * env_num
                
                # vln_envs.update_cam_occupancy_map_pose() # adjust the camera pose
                # env_actions = update_env_actions(action_name, paths_list, path_idx=0)
                # env_actions = vln_envs.calc_env_action_offset(env_actions,action_name)
                obs = env.step(actions=env_actions)
                continue
            
            if 'oracle' not in action_name and i % 50 == 0:
                status_abnormal_list = vln_envs.check_and_reset_robot(cur_iter=i, update_freemap=False, verbose=vln_config.test_verbose)
                for status_idx, status in enumerate(status_abnormal_list):
                    if status:
                        # log.error(f"Scan: {scan}, Path_id: {path_id_list[status_idx]}: The robot is reset. Break this episode for this env.")
                        vln_envs.end_list[status_idx] = True
            
            if args.test_verbose and args.windows_head:
                # TODO
                vln_envs.cam_occupancy_map_local_list[0].update_windows_head(robot_pos=vln_envs.isaac_robots.get_world_pose()[0], mode=args.windows_head_type)
            
            '''(2) check for action finish status and update navigation'''
            for env_idx in range(env_num):
                exe_paths = [[]] * env_num
                if vln_envs.end_list[env_idx]:
                    # this episode has ended.
                    # env_actions[env_idx] = stand_still_action
                    continue

                if env_action_finish_states[env_idx]:
                    current_point = nav_point_list[env_idx]
                    paths = paths_list[env_idx]
                    topdown_map = topdown_maps[env_idx]
                    log.info(f"======Path_id: {path_id_list[env_idx]}========")
                    if current_point == 0:
                        log.info(f"The robot starts navigating")
                    if current_point < len(paths)-1:
                        log.info(f"The robot is navigating to the {current_point+1}-th target place.")

                        with open(args.episode_status_info_file_list[env_idx], 'a') as f:
                            f.write(f"Current point number: {current_point}\n")
                        
                        freemap, camera_pose = vln_envs.get_global_free_map_single(env_idx=env_idx, verbose=args.test_verbose)
                        topdown_map.update_map(freemap, camera_pose, update_map=True, verbose=args.test_verbose, env_idx=env_idx)
                        robot_current_position = vln_envs.get_robot_poses()[env_idx][0]
                        
                        exe_path = topdown_map.navigate_p2p(robot_current_position, paths[current_point+1], step_time=i, verbose=(args.test_verbose or args.save_path_planning), save_dir=args.episode_path_list[env_idx])
                        exe_paths.append(exe_path) 
                        if exe_path is None or len(exe_path) == 0:
                            # path planning fails
                            log.error(f"Scan: {scan}, Path_id: {path_id}. Path planning fails to find the path from the current point to the next point.")
                            vln_envs.end_list[env_idx] = True
                            continue

                        if 'oracle' in action_name:
                            # TODO
                            action_info.update({'current_step': i})
                            actions = {'h1': {action_name: [exe_path, action_info]}}
                        else:
                            exe_path = vln_envs.calc_single_env_action_offset(env_idx, exe_path)
                            actions = {'h1': {action_name: [exe_path]}}
                            env_actions[env_idx] = actions
                        
                        nav_point_list[env_idx] += 1

                        if vln_config.windows_head:
                            # show the topdown camera
                            vln_envs.cam_occupancy_map_local_list[0].update_windows_head(robot_pos=vln_envs.isaac_robots[0].get_world_pose()[0], mode=args.windows_head_type)
            
            if 'oracle' in action_name:
                actions['h1'][action_name][1]['current_step'] = i

            # env_actions.append(actions)
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
                    # total_points.append(exe_point)
                    is_image_stacked = False
                    move_step = deepcopy(i)
            else:
                move_step = warm_step

            # stack images and information
            if (i-move_step) != 0 and (i-move_step) % (args.sample_episodes.step_interval-1) == 0:
                # Since oracle_move_path_controller moves to the next point every 5 steps, the image is fetched every 5+3 steps
                camera_pose_dict = vln_envs.get_camera_pose()
                robot_pose_dict = vln_envs.get_robot_poses()
                data_collector.collect_and_send_data(i, env, 
                            camera_list=data_camera_list, camera_pose_dict=camera_pose_dict,
                            robot_pose_dict=robot_pose_dict,
                            end_list=vln_envs.end_list, 
                            add_rgb_subframes=True, finish_flag=False)

                is_image_stacked = True

            if args.test_verbose and args.save_obs and (i-move_step) != 0 and (i-move_step)%(args.sample_episodes.step_interval-1) == 0:
                # TODO
                vln_envs.save_observations(camera_list=data_camera_list, data_types=["rgba", "depth"], add_rgb_subframes=True, step_time=i)
                freemap, camera_pose = vln_envs.get_global_free_map(verbose=args.test_verbose)
                topdown_map.update_map(freemap, camera_pose, update_map=True, verbose=args.test_verbose)

            # get the action state
            if len(obs) > 0:
                for env_idx, (task_name, task) in enumerate(obs.items()):
                    for robot_name, robot in task.items():
                        action_state = robot[action_name]
                        if action_state['finished']:
                            env_action_finish_states[env_idx] = True
                        else:
                            env_action_finish_states[env_idx] = False

            else:
                for env_idx in range(env_num):
                    env_action_finish_states[env_idx] = False
            
            '''(3) Justify the episode finish status'''
            for env_idx, action_finish_state in enumerate(env_action_finish_states):
                if action_finish_state:
                    if nav_point_list[env_idx] == len(paths_list[env_idx])-1 and vln_envs.just_end_list[env_idx] == True:
                        log.info(f"[Success] Scan: {scan}, Path_id: {path_id_list[env_idx]}. The robot has finished this episode !!!")
                        vln_envs.end_list[env_idx] = True
                        vln_envs.success_list[env_idx] = True
                        vln_envs.just_end_list[env_idx] = False

                        with open(args.episode_status_info_file_list[env_idx], 'a') as f:
                            f.write(f"Episode finished: {vln_envs.success_list[env_idx]}\n")
        
        for status_info_file in args.episode_status_info_file_list:
            with open(status_info_file, 'a') as f:
                f.write(f"Episode finished: {vln_envs.success_list[env_idx]}\n")

        end_time = time.time()
        total_time = (end_time - start_time)/60
        log.info(f"Total time for this banch: {total_time:.2f} minutes")

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