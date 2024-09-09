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
from multiprocessing import Pipe, Process, Pool
from threading import Thread
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

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

def save_obs_images(obs, env_num, camera='pano_camera_0'):
    for i in range(env_num):
        rgb = obs[f'vln_{i}'][f'h1_{i}'][camera]['rgba']
        plt.clf()
        plt.imshow(rgb)
        plt.savefig(f'logs/images/obs_{camera}_{i}.png')
        plt.clf()

def build_dataset():
    ''' Build dataset for VLN
    '''
    vln_config, sim_config = process_args()
    if vln_config.split != "":
        vln_config.datasets.splits = [vln_config.split]
    vln_datasets = VLNDataLoader(vln_config, 
                            sim_config=sim_config,
                            splits=vln_config.datasets.splits,
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

def sample_episode_worker(args, vln_envs, data_camera_list, data_list):
    """
    Worker function to be executed in parallel.
    """
    is_app_up = False
    for split, scan in data_list:
        # try:
        scan_log_dir = os.path.join(args.sample_episode_dir, split, scan)
        if not args.settings.force_sample_scan and os.path.exists(scan_log_dir):
            log.info(f'Scan {scan} has been sampled. Pass.')
            continue
        env = sample_episodes_single_scan(args, vln_envs, data_camera_list, split=split, scan=scan, is_app_up=is_app_up)
        is_app_up = True
            # Assuming `sample_episodes_single_scan` handles its own exceptions and cleanup
        # except Exception as e:
        #     log.error(f"Error processing {scan} in {split}: {e}")
        # finally:
        #     # if hasattr(env, 'simulation_app'):
        #     env.simulation_app.close()
        #     return
    env.simulation_app.close()

def sample_episodes_multiprocess(args, num_workers, vln_envs, data_camera_list):
    '''Use multiprocess to handle different scans'''
    tasks = [[] for _ in range(num_workers)]
    scans = [[] for _ in range(num_workers)]
    
    i = 0
    # for split, vln_envs in vln_envs_all.items():
    for split in vln_envs.data.keys():
        for scan in vln_envs.data[split].keys():
            scans[i%num_workers].append((split, scan))
            i += 1

    for task_idx in range(num_workers):
        tasks[task_idx] = (args, vln_envs, data_camera_list, scans[task_idx])

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Using the executor to submit all tasks and immediately creating a list of futures
        futures = [executor.submit(sample_episode_worker, *task) for task in tasks]

        # Optionally, you can wait for all futures to complete and handle their results or exceptions
        for future in futures:
            try:
                result = future.result()  # This will block until the future is complete
                # Handle the result (if any) here
            except Exception as exc:
                # Handle exceptions
                print(f'Generated an exception: {exc}')

def sample_episodes_reset_scans(args, vln_envs, data_camera_list, assigned_split=None, assigned_scan=None):
    '''Use one app to handle different scans'''
    is_app_up = False
    if assigned_split is not None and assigned_scan is not None:
        env = sample_episodes_single_scan(args, vln_envs, data_camera_list, split=assigned_split, scan=assigned_scan, is_app_up=is_app_up)
    else:
        for split in vln_envs.data.keys():
            for scan in vln_envs.data[split].keys():
                scan_log_dir = os.path.join(args.sample_episode_dir, split, scan)
                if not args.settings.force_sample_scan and os.path.exists(scan_log_dir):
                    log.info(f'Scan {scan} has been sampled. Pass.')
                    continue
                env = sample_episodes_single_scan(args, vln_envs, data_camera_list, split=split, scan=scan, is_app_up=is_app_up)
                is_app_up = True

    env.simulation_app.close()

def sample_episodes_single_scan(args, vln_envs, data_camera_list, split=None, scan=None, is_app_up=False):
    '''1. Init the variables'''
    action_name = args.settings.action
    is_app_up = is_app_up
    scan = args.scan if scan is None else scan
    split = args.split if split is None else split 
    stand_still_action = {'h1': {'stand_still': []}}

    '''2. Init the data and env_num'''
    vln_envs.allocate_data(split, scan)

    '''3. Init the app or Reset the scene'''
    if not is_app_up:
        # First needs to start the app
        data_item = vln_envs.init_multiple_episodes(split, scan, init_omni_env=True)
    else:
        data_item = vln_envs.init_multiple_episodes(split, scan, init_omni_env=False, reset_scene=True)

    env = vln_envs.env

    if args.windows_head:
        vln_envs.cam_occupancy_map_local_list[0].open_windows_head(text_info=data_item['instruction']['instruction_text'])
    
    '''4. init pipe for saving images'''
    parent_conn, child_conn = Pipe()
    data_collector = dataCollector(args, parent_conn, child_conn, split, scan, vln_envs.path_id_list)
    # save_process = Process(target=data_collector.save_episode_data, args=())
    save_process = Thread(target=data_collector.save_episode_data, args=())
    save_process.start()
    log.info(f"Save process starts.")

    '''5. start simulation'''
    i = 0
    render = False
    warm_step = 240 if args.headless else 2000
    move_step = warm_step

    if 'oracle' in action_name:
        # TODO
        action_info = {
            'current_step': 0,
            'topdown_camera_local': vln_envs.cam_occupancy_map_local,
            'topdown_camera_global': vln_envs.cam_occupancy_map_global
        }

        init_actions = {'h1': {action_name: [[paths[0]], action_info]}}
    else:
        env_actions = update_env_actions(action_name, vln_envs.paths_list, path_idx=0)
        env_actions = vln_envs.calc_env_action_offset(env_actions,action_name)

    start_time = time.time()
    
    '''6. Enter the env flow loop'''
    while (not all(vln_envs.end_list)) and (not vln_envs.all_episode_finish) and env.simulation_app.is_running():
        ''' (0) check the maximum steps for each env'''
        max_step = 500 if args.debug else args.settings.max_step
        for env_idx in range(vln_envs.env_num):
            if (i - vln_envs.env_step_start_index[env_idx]) >= max_step:
                log.error(f"[Failed]. Scan: {scan}, Path_id: {vln_envs.path_id_list[env_idx]}. Exceed the maximum steps: {max_step}")
                vln_envs.episode_end_setting(split, scan, env_idx, reason='maximum step')

        i += 1

        if i % sim_config.config.simulator.rendering_interval == 0:
            render = True
        else:
            render = False

        # update warm up list
        for warm_up_idx in range(vln_envs.env_num):
            if vln_envs.warm_up_list[warm_up_idx] > 0:
                vln_envs.warm_up_list[warm_up_idx] -= 1
        
        '''(1) warm up process'''
        if i < warm_step:
            if 'oracle' in action_name:
                init_actions['h1'][action_name][1]['current_step'] = i
            obs = env.step(actions=env_actions)
            
            if i % 50 == 0:
                if args.windows_head:
                    # show the topdown camera
                    vln_envs.cam_occupancy_map_local_list[0].update_windows_head(robot_pos=vln_envs.isaac_robots[0].get_world_pose()[0], mode=args.windows_head_type)
                    # log the FPS
                current_time = time.time()
                log.info(f"Current step: {i}. FPS: {i/(current_time-start_time):.2f}")
            
            continue

        elif i == warm_step:
            # first warm up finished
            for env_idx in range(vln_envs.env_num):
                if vln_envs.warm_up_list[env_idx] == 0:
                    topdown_map = vln_envs.GlobalTopdownMap(args, scan)
                    freemap, camera_pose = vln_envs.get_global_free_map_single(env_idx, verbose=args.test_verbose)
                    topdown_map.update_map(freemap, camera_pose, verbose=args.test_verbose, env_idx=env_idx)
                    vln_envs.topdown_maps[env_idx] = topdown_map
                    log.info(f"====The global freemap has been initialized for Path_id {vln_envs.path_id_list[env_idx]}====")
                    vln_envs.env_action_finish_states[env_idx] = True
            
            obs = env.step(actions=env_actions, add_rgb_subframes=True, render=True)
            continue
        
        ''' (2) Check for the robot weather falls or stucks'''
        if 'oracle' not in action_name and i % 20 == 0:
            status_abnormal_list, fall_list, stuck_list = vln_envs.check_and_reset_robot(cur_iter=i, update_freemap=False, verbose=args.test_verbose)
            for status_idx, status in enumerate(status_abnormal_list):
                if vln_envs.warm_up_list[status_idx] == 0:
                    if status:
                        if fall_list[status_idx]:
                            reason = 'fall'
                        elif stuck_list[status_idx]:
                            reason = 'stuck'
                        vln_envs.episode_end_setting(split, scan, status_idx, reason)
        
        if args.test_verbose and args.windows_head:
            # TODO
            vln_envs.cam_occupancy_map_local_list[0].update_windows_head(robot_pos=vln_envs.isaac_robots.get_world_pose()[0], mode=args.windows_head_type)
        
        '''(3) check for action finish status and update navigation'''
        for env_idx in range(vln_envs.env_num):
            if vln_envs.end_list[env_idx] or vln_envs.warm_up_list[env_idx] > 0:
                continue

            if vln_envs.env_action_finish_states[env_idx]:
                current_point = vln_envs.nav_point_list[env_idx]
                paths = vln_envs.paths_list[env_idx]
                topdown_map = vln_envs.topdown_maps[env_idx]
                log.info(f"======Env {env_idx} | Path_id: {vln_envs.path_id_list[env_idx]}========")
                if current_point == 0:
                    log.info(f"The robot starts navigating")
                if current_point < len(paths)-1:
                    log.info(f"The robot is navigating to the {current_point+1}-th target place.")

                    with open(args.episode_status_info_file_list[env_idx], 'a') as f:
                        f.write(f"Current point number: {current_point}\n")
                    
                    freemap, camera_pose = vln_envs.get_global_free_map_single(env_idx=env_idx, verbose=args.test_verbose)
                    if topdown_map is None:
                        vln_envs.topdown_maps[env_idx] = vln_envs.GlobalTopdownMap(args, scan)
                        topdown_map = vln_envs.topdown_maps[env_idx]
                    topdown_map.update_map(freemap, camera_pose, update_map=True, verbose=args.test_verbose, env_idx=env_idx)
                    robot_current_position = vln_envs.get_robot_poses()[env_idx][0]
                    
                    exe_path = topdown_map.navigate_p2p(robot_current_position, 
                    paths[current_point+1], step_time=(i-vln_envs.env_step_start_index[env_idx]), verbose=(args.test_verbose or args.save_path_planning), save_dir=args.episode_path_list[env_idx]) 
                    if exe_path is None or len(exe_path) == 0:
                        # path planning fails
                        vln_envs.episode_end_setting(split, scan, env_idx, reason='path planning')
                        continue

                    if 'oracle' in action_name:
                        # TODO
                        action_info.update({'current_step': i})
                        actions = {'h1': {action_name: [exe_path, action_info]}}
                    else:
                        exe_path = vln_envs.calc_single_env_action_offset(env_idx, exe_path)
                        actions = {'h1': {action_name: [exe_path]}}
                        env_actions[env_idx] = actions
                    
                    vln_envs.nav_point_list[env_idx] += 1
                    vln_envs.env_action_finish_states[env_idx] = False

                    if args.windows_head:
                        # show the topdown camera
                        vln_envs.cam_occupancy_map_local_list[0].update_windows_head(robot_pos=vln_envs.isaac_robots[0].get_world_pose()[0], mode=args.windows_head_type)
        
        if 'oracle' in action_name:
            actions['h1'][action_name][1]['current_step'] = i

        if i % args.sample_episodes.step_interval == 0:
            # data_type = args.settings.camera_data_type
            # input data_type to retrival the high quality image 
            add_rgb_subframes = True
        else:
            # data_type = None
            add_rgb_subframes = False

        '''(4) Justify the episode finish status'''
        for env_idx, action_finish_state in enumerate(vln_envs.env_action_finish_states):
            if vln_envs.warm_up_list[env_idx] == 0 and (action_finish_state or vln_envs.end_list[env_idx]):
                if vln_envs.nav_point_list[env_idx] == len(vln_envs.paths_list[env_idx])-1 and vln_envs.just_end_list[env_idx] == True:
                    # success
                    vln_envs.episode_end_setting(split, scan, env_idx, reason='success')
                
                if args.settings.sample_env_flow:
                    # assign new path to the finished env
                    if vln_envs.end_list[env_idx]:
                        update_flag = vln_envs.update_next_single_data(env_idx, split, scan, current_step=i)
                        if update_flag:
                            log.error(f"{env_idx}-th Env: Assign new path_id: {vln_envs.path_id_list[env_idx]}. Reset this env!")
                            robot_pose = vln_envs.get_robot_poses()[env_idx][0]
                            robot_pose_offset = vln_envs.calc_single_env_action_offset(env_idx, [robot_pose])
                            env_actions[env_idx] = {'h1':{action_name: [robot_pose_offset]}}
                            render = True
                            add_rgb_subframes = True
                    
                    if vln_envs.data_idx > args.settings.max_episodes_per_scan:
                        vln_envs.all_episode_finish = True
                        log.info(f"Scan {scan} has been sampled over maximum episodes settings {args.settings.max_episodes_per_scan}.")
                        break

        '''(4) Step and get new observations'''
        obs = env.step(actions=env_actions, add_rgb_subframes=add_rgb_subframes, render=render)
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

        '''(5) Save observations'''
        if (i-move_step) != 0 and (i-move_step) % (args.sample_episodes.step_interval-1) == 0:
            # Since oracle_move_path_controller moves to the next point every 5 steps, the image is fetched every 5+3 steps
            camera_pose_dict = vln_envs.get_camera_pose()
            robot_pose_dict = vln_envs.get_robot_poses()
            data_collector.collect_and_send_data(i, env, 
                        camera_list=data_camera_list, camera_pose_dict=camera_pose_dict,
                        robot_pose_dict=robot_pose_dict,
                        end_list=vln_envs.end_list, 
                        path_id_list=vln_envs.path_id_list,
                        start_step_list=vln_envs.env_step_start_index,
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
                    vln_envs.env_action_finish_states[env_idx] = action_state['finished']

        else:
            for env_idx in range(vln_envs.env_num):
                vln_envs.env_action_finish_states[env_idx] = False
    
    '''7. Finish this scan'''
    end_time = time.time()
    total_time = (end_time - start_time)/60
    log.info(f"Total time for scan {scan}: {total_time:.2f} minutes")

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
    
    if vln_config.settings.mode == "sample_episodes_multiprocess":
        sample_episodes_multiprocess(vln_config, vln_config.settings.num_workers, vln_envs, data_camera_list)
    elif vln_config.settings.mode == "sample_episodes_reset_scans":
        # sample_episodes_reset_scans(vln_config, vln_envs, data_camera_list, assigned_split='train', assigned_scan='VzqfbhrpDEA')
        sample_episodes_reset_scans(vln_config, vln_envs, data_camera_list)