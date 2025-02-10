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
import lmdb
from collections import defaultdict
from PIL import Image
from copy import deepcopy
import threading
from multiprocessing import Pipe, Process, Pool
from threading import Thread
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp

from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
from grutopia.core.util.log import log

from vln.src.dataset.data_utils_multi_env import VLNDataLoader
from vln.src.dataset.data_collector import dataCollector, LmdbDataCollector

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)

from vln.parser import process_args

########### sixth floor ###################
import json
import torch
import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import hydra
import traceback
#! after sim setup is finished
import open3d as o3d

from grutopia.core.env import BaseEnv
from grutopia.core.config import SimulatorConfig
from grutopia.core.util.log import log
import cv2
class sixth_floor_scene:
    def __init__(self, json_path, sim_config_path, device_number):

        self.json_path = json_path
        self.device_number = device_number
        with open(self.json_path, "r") as json_file:
            lego_json_config = json.load(json_file)

        self.lego_usd_root = lego_json_config["usd_model_root"]
        self.lego_gs_root = lego_json_config["gs_model_root"]
        self.lego_name_list = lego_json_config["model_list"]
        self.lego_device_number = 0
        self.lego_editable = True

        sim_config = SimulatorConfig(sim_config_path) #! scene_usd is empty!
        self.sim_config = sim_config
        self.env = BaseEnv(sim_config, headless=True, webrtc=False)
        self.env.reset()
        
        self.lego_xform_list = [] 
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.prims import get_prim_at_path
        for lego_name in self.lego_name_list:
            create_prim(usd_path=os.path.join(self.lego_usd_root, lego_name["usd_name"]), 
                        prim_path="/World/" + lego_name["isaac_name"], 
                        position=lego_name["init_position"], 
                        orientation=lego_name["init_orientation"],
                        scale=lego_name["init_scale"])
            self.lego_xform_list.append(XFormPrim("/World/" + lego_name["isaac_name"]))
            print("Create preset usd at /World/" + lego_name["isaac_name"])
        create_prim("/World/light", "DistantLight")
        print("Create preset light at /World/light")
        self.task_name = self.env.config.tasks[0].name
        self.robot_name = self.env.config.tasks[0].robots[0].name
        self.init_agents()
    
    def init_agents(self):
        '''call after self.init_env'''
        self.agents = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].isaac_robot
        self.agent_last_pose = None
        self.agent_init_pose = self.sim_config.config.tasks[0].robots[0].position
        self.agent_init_rotation = self.sim_config.config.tasks[0].robots[0].orientation

        # self.set_agent_pose(self.agent_init_pose, self.agent_init_rotation)
    
    def set_agent_pose(self, position, rotation):
        self.agents.set_world_pose(position, rotation)
    def get_agent_pose(self):
        return self.agents.get_world_pose()

    def set_paths(self):
        json_path = "thirdparty/landmark_isaacsim_interaction/json_configs/multi_model_sixthfloor.json"
        img_path = "thirdparty/landmark_isaacsim_interaction/rendered_imgs/"
        file_path = '/ssd/xiaxinyuan/code/w61-grutopia/vln/configs/sim_cfg_path_generation.yaml'
        device_number = 0
        save_dir = os.path.join(img_path, "sixth_floor")
        rgb_save_dir = os.path.join(save_dir, "rgb")
        depth_save_dir = os.path.join(save_dir, "depth")
        pcd_save_dir = os.path.join(save_dir, "pcd")
        debug_rgb_save_dir = os.path.join(save_dir, "debug_rgb")
        pano_rgb_save_dir = os.path.join(save_dir, "pano_rgb")
        os.makedirs(rgb_save_dir, exist_ok=True)
        os.makedirs(depth_save_dir, exist_ok=True)
        os.makedirs(pcd_save_dir, exist_ok=True)
        os.makedirs(debug_rgb_save_dir, exist_ok=True)
        os.makedirs(pano_rgb_save_dir, exist_ok=True)


########### end sixth floor ###########################



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
    elif 'sixth_floor' in vln_config.settings.mode:
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

def sample_episode_worker(args, sim_config, vln_envs, data_camera_list, data_list):
    """
    Worker function to be executed in parallel.
    """
    is_app_up = False
    for split, scan in data_list:
        scan_log_dir = os.path.join(args.sample_episode_dir, split, scan)
        if not args.settings.force_sample_scan and os.path.exists(scan_log_dir):
            log.info(f'Scan {scan} has been sampled. Pass.')
            continue
        env = sample_episodes_single_scan(args, sim_config, vln_envs, data_camera_list, split=split, scan=scan, is_app_up=is_app_up)
        if env is not None:
            is_app_up = True
    env.simulation_app.close()

def process_wrapper(*task):
    try:
        sample_episode_worker(*task)
    except Exception as e:
        log.error(f"Process encountered an error: {e}")
            
def sample_episodes_multiprocess(args, sim_config, num_workers, vln_envs, data_camera_list):
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
        tasks[task_idx] = (args, sim_config, vln_envs, data_camera_list, scans[task_idx])
    
    mp.set_start_method("spawn", force=True)  # "spawn" is recommended for CUDA compatibility
    # with mp.Pool(num_workers) as pool:
        # pool.starmap(sample_episode_worker, tasks)  # Distribute tasks to worker function
    
    processes = []
    for task_idx in range(num_workers):
        tasks[task_idx] = (args, sim_config, vln_envs, data_camera_list, scans[task_idx])
        process = mp.Process(target=process_wrapper, args=tasks[task_idx])
        process.start()
        processes.append(process)

    # Join processes to ensure all complete
    for process in processes:
        process.join()
        
    log.info('Finished.')
    

def sample_episodes_reset_scans(args, sim_config, vln_envs, data_camera_list, assigned_split=None, assigned_scan=None, assigned_path_id=None):
    '''Use one app to handle different scans'''
    is_app_up = False
    if len(assigned_split) > 0 and len(assigned_scan) > 0:
        env = sample_episodes_single_scan(args, sim_config, vln_envs, data_camera_list, split=assigned_split, scan=assigned_scan, path_id=assigned_path_id, is_app_up=is_app_up)
    else:
        for split in vln_envs.data.keys():
            for scan in vln_envs.data[split].keys():
                scan_log_dir = os.path.join(args.sample_episode_dir, split, scan)
                if not args.settings.force_sample_scan and os.path.exists(scan_log_dir):
                    log.info(f'Scan {scan} has been sampled. Pass.')
                    continue
                env = sample_episodes_single_scan(args, sim_config, vln_envs, data_camera_list, split=split, scan=scan, is_app_up=is_app_up)
                if env is not None:
                    # env has not up
                    is_app_up = True

    env.simulation_app.close()

def sample_episodes_single_scan(args, sim_config, vln_envs, data_camera_list, split=None, scan=None, path_id=None, is_app_up=False):
    '''1. Init the variables'''
    action_name = args.settings.action
    is_app_up = is_app_up
    scan = args.scan if scan is None else scan
    split = args.split if split is None else split 
    stand_still_action = {'h1': {'stand_still': []}}

    '''2. Init the data and env_num'''
    allocate_flag = vln_envs.allocate_data(split, scan, path_id)
    if not allocate_flag:
        # This scan has been sampled.
        return None

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
    if args.sample_episodes.save_form == 'thread':
        # V1: Use multiple threads to save raw images and information
        parent_conn, child_conn = Pipe()
        data_collector = dataCollector(args, parent_conn, child_conn, split, scan, vln_envs.path_id_list)
        # save_process = Process(target=data_collector.save_episode_data, args=())
        save_process = Thread(target=data_collector.save_episode_data, args=())
        save_process.start()
        log.info(f"Save process starts.")
    elif args.sample_episodes.save_form == 'lmdb':
        # V2: use lmdb to save all information
        data_collector = LmdbDataCollector(args, split, scan, vln_envs.path_id_list, args.lmdb_path, sim_config.config.tasks[0].env_num)

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
        # if i == 30: # !!!
        #     vln_envs.all_episode_finish = True
        ''' (0) check the maximum steps for each env'''
        max_step = 400 if args.debug else args.settings.max_step
        for env_idx in range(vln_envs.env_num):
            if (i - vln_envs.env_step_start_index[env_idx]) >= max_step:
                log.error(f"[Failed]. Scan: {scan}, Path_id: {vln_envs.path_id_list[env_idx]}. Exceed the maximum steps: {max_step}")
                vln_envs.episode_end_setting(split, scan, env_idx, reason='maximum step')

        i += 1

        if i % sim_config.config.simulator.rendering_interval == 0:
            render = True
        else:
            render = False
        render = True

        # update warm up list
        for warm_up_idx in range(vln_envs.env_num):
            if vln_envs.warm_up_list[warm_up_idx] > 0:
                vln_envs.warm_up_list[warm_up_idx] -= 1
        
        '''(1) warm up process'''
        if i < warm_step:
            if 'oracle' in action_name:
                init_actions['h1'][action_name][1]['current_step'] = i
            obs = env.step(actions=env_actions,render=False, add_rgb_subframes=False)
            
            if i % 50 == 0:
                # break
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
        
        if args.test_verbose or args.windows_head:
            # TODO
            if i % 100 == 0:
                vln_envs.cam_occupancy_map_local_list[0].update_windows_head(robot_pos=vln_envs.isaac_robots[0].get_world_pose()[0], mode=args.windows_head_type) # For now, only use the first env to show the topdown camera
        
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
                        data_collector.save_data(env_idx,vln_envs.path_id_list[env_idx], vln_envs.success_list[env_idx], vln_envs.fail_reasons[env_idx], data_item['instruction']['instruction_text'])
                        if args.sample_episodes.docker_nums > 1:
                            # update the json file for multi docker
                            with open(args.lmdb_json_path, 'r') as f:
                                json_data = json.load(f)
                                if vln_envs.success_list[env_idx]:
                                    json_data[scan][vln_envs.path_id_list[env_idx]] = 'success'
                                else:
                                    json_data[scan][vln_envs.path_id_list[env_idx]] = vln_envs.fail_reasons[env_idx]
                            with open(args.lmdb_json_path, 'w') as f:
                                json.dump(json_data, f, indent=4)
            
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
            if args.sample_episodes.save_form == 'thread':
                data_collector.collect_and_send_data(i, env, 
                            camera_list=data_camera_list, camera_pose_dict=camera_pose_dict,
                            robot_pose_dict=robot_pose_dict,
                            end_list=vln_envs.end_list, 
                            path_id_list=vln_envs.path_id_list,
                            start_step_list=vln_envs.env_step_start_index,
                            add_rgb_subframes=True, finish_flag=False)
            elif args.sample_episodes.save_form == 'lmdb':
                progress_list = []
                for env_idx in range(vln_envs.env_num):
                    progress = vln_envs.nav_point_list[env_idx] / len(vln_envs.paths_list[env_idx])
                    progress_list.append(progress)
                data_collector.collect_data(i, env, 
                            camera_list=data_camera_list, camera_pose_dict=camera_pose_dict,
                            robot_pose_dict=robot_pose_dict,
                            end_list=vln_envs.end_list, 
                            path_id_list=vln_envs.path_id_list,
                            start_step_list=vln_envs.env_step_start_index,
                            progress_list=progress_list,
                            add_rgb_subframes=True, 
                            success_list=vln_envs.success_list,
                            fail_reasons=vln_envs.fail_reasons)

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
    if args.sample_episodes.save_form == 'thread':
        parent_conn.send({'finish_flag': True})
        save_process.join()

    return env

    # if vln_config.windows_head:
        # close the topdown camera
        # vln_envs.cam_occupancy_map_local.close_windows_head()
    
    # env.simulation_app.close()

def read_assigned_json(args, json_dir, docker_id):
    args.lmdb_json_path = os.path.join(json_dir, f"scan_pathId_part_{docker_id}.json")
    with open(args.lmdb_json_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    vln_envs, vln_config, sim_config, data_camera_list = build_dataset()
    log.info(f'Is in container: {is_in_container()}')
    
    if vln_config.settings.mode == "sample_episodes_multiprocess":
        sample_episodes_multiprocess(vln_config, sim_config, vln_config.settings.num_workers, vln_envs, data_camera_list)
    elif vln_config.settings.mode == "sample_episodes_reset_scans":
        sample_episodes_reset_scans(vln_config, sim_config, vln_envs, data_camera_list)
        # sample_episodes_reset_scans(vln_config, sim_config, vln_envs, data_camera_list, assigned_split=vln_config.split, assigned_scan=vln_config.scan, assigned_path_id=vln_config.path_id)
    elif vln_config.settings.mode == "sample_episodes_reset_scans_with_assigned_path":
        # This is for multi-docker
        data = read_assigned_json(vln_config, vln_config.lmdb_pathId_dir, vln_config.docker_id)
        
        for scan, path_id in data.items():
            log.info(f"***Start with Scan: {scan}***")
            sample_episodes_reset_scans(vln_config, sim_config, vln_envs, data_camera_list, assigned_split=vln_config.split, assigned_scan=scan)
    elif vln_config.settings.mode == "sixth_floor":
        sample_episodes_reset_scans(vln_config, sim_config, vln_envs, data_camera_list,assigned_split = 'sixth_floor', assigned_scan='0')
