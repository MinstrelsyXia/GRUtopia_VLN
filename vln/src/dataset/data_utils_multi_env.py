# Author: w61
# Date: 2024.7.19
''' Class to load dataset for VLN
'''

import os
import gzip
import json
import copy
import numpy as np
import time
from collections import defaultdict
from copy import deepcopy
import shutil

import torch
from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import importlib
# from scipy.ndimage import convolve, gaussian_filter

try:
    from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
except:
    pass

from grutopia.core.util.log import log
from grutopia.core.env import BaseEnv

from ..utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations

from ..local_nav.pointcloud import generate_pano_pointcloud_local, pc_to_local_pose
from ..local_nav.BEVmap import BEVMap


def load_data(args, split):
    ''' Load data based on VLN-CE
    '''
    dataset_root_dir = args.datasets.base_data_dir
    total_scans = []
    load_data = []
    with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for item in data["episodes"]:
            item["original_start_position"] = copy.copy(item["start_position"])
            item["original_start_rotation"] = copy.copy(item["start_rotation"])
            item["start_position"] = [item["original_start_position"][0], -item["original_start_position"][2], item["original_start_position"][1]]
            item["start_rotation"] = [-item["original_start_rotation"][3], item["original_start_rotation"][0], item["original_start_rotation"][2], -item["original_start_rotation"][1]] # [x,y,z,-w] => [w,x,y,z]
            item["scan"] = item["scene_id"].split("/")[1]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], -path[2], path[1]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data.append(item)
            total_scans.append(item["scan"])

    log.info(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data, list(set(total_scans))

def load_gather_data(args, split, filter_same_trajectory=False, filter_stairs=False):
    dataset_root_dir = args.datasets.base_data_dir
    with open(os.path.join(dataset_root_dir, "gather_data", f"{split}_gather_data.json"), 'r') as f:
        data = json.load(f)
    with open(os.path.join(dataset_root_dir, "gather_data", "env_scan.json"), 'r') as f:
        scan = json.load(f)

    new_data = defaultdict(list)
    if filter_same_trajectory or filter_stairs:
        if filter_same_trajectory:
            trajectory_list = []
        for scan, data_item in data.items():
            for item in data_item:
                if filter_same_trajectory:
                    if item['trajectory_id'] in trajectory_list:
                        continue
                    else:
                        trajectory_list.append(item['trajectory_id'])

                if filter_stairs:
                    if 'stair' in item['instruction']['instruction_text']:
                        continue

                    different_height = False
                    paths = item['reference_path']
                    for path_idx in range(len(paths)-1):
                        if abs(paths[path_idx+1][2] - paths[path_idx][2]) > 0.3:
                            different_height = True
                            break
                    if different_height:
                        continue

                new_data[scan].append(item)
        data = new_data

    return data, scan

def load_scene_usd(args, scan):
    ''' Load scene USD based on the scan
    '''
    find_flag = False
    for root, dirs, files in os.walk(os.path.join(args.datasets.mp3d_data_dir, scan)):
        for file in files:
            if file.endswith(".usd") and "non_metric" not in file and "isaacsim_" in file:
                scene_usd_path = os.path.join(root, file)
                find_flag = True
                break
        if find_flag:
            break
    if not find_flag:
        log.error("Scene USD not found for scan %s", scan)
        return None
    return scene_usd_path

def check_fall(agent, obs, pitch_threshold=45, roll_threshold=45, adjust=False, initial_pose=None, initial_rotation=None):
    '''
    Determine if the robot is falling based on its rotation quaternion.
    '''
    current_quaternion = obs['orientation']
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    roll, pitch, yaw = quat_to_euler_angles(current_quaternion)

    # Check if the pitch or roll exceeds the thresholds
    if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
        is_fall = True
        log.info(f"Robot falls down!!!")
        cur_pos, cur_rot = obs["position"], quat_to_euler_angles(obs["orientation"])
        log.info(f"Current Position: {cur_pos}, Orientation: {cur_rot}")
    else:
        is_fall = False

    if is_fall and adjust:
        # randomly adjust the pose to avoid falling
        # NOTE: This does not work since it could lead the robot be caught in the colliders
        if initial_pose is None:
            initial_pose = obs['position']
        if not isinstance(initial_pose, np.ndarray):
            initial_pose = np.array(initial_pose)
        
        if initial_rotation is None:
            initial_rotation = euler_angles_to_quat(np.array([0,0,0]))
        if not isinstance(initial_rotation, np.ndarray):
            initial_rotation = np.array(initial_rotation)
            initial_rotation_euler = quat_to_euler_angles(initial_rotation)

        # randomly sample offset
        position_offset = np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1), 0])
        rotation_y_offset = np.array([0, np.random.uniform(low=-30, high=30), 0])

        adjust_position = initial_pose + position_offset
        adjust_rotation = initial_rotation + euler_angles_to_quat(rotation_y_offset)

        log.info(f"Target adjust position: {adjust_position}, adjust rotation: {adjust_rotation}")

        agent.set_world_pose(position=adjust_position, 
                            orientation=adjust_rotation)
        # agent.set_joint_velocities(np.zeros(len(agent.dof_names)))
        # agent.set_joint_positions(np.zeros(len(agent.dof_names)))
    
    if not is_fall:
        log.info(f"Robot does not fall")
        cur_pos, cur_rot = obs["position"], quat_to_euler_angles(obs["orientation"])
        log.info(f"Current Position: {cur_pos}, Orientation: {cur_rot}")

    return is_fall

def get_sensor_info(step_time, cur_obs, verbose=False):
    # type: rgba, depth, frame
    camera_list = ['camera_front', 'camera_behind', 'camera_left', 'camera_right', 'camera_tp']
    for camera_type in camera_list:
        camera = cur_obs[camera_type]
        camera_rgb = camera['rgba'][...,:3]
        camera_depth = camera['depth']
        camera_pointcloud = camera['pointcloud']
        if verbose:
            image_save_dir = os.path.join(ROOT_DIR, "logs", "images")
            if not os.path.exists(image_save_dir):
                os.mkdir(image_save_dir)
            c_rgb_path = os.path.join(image_save_dir, f"{camera_type}_rgb_{str(step_time)}.jpg")
            try:
                plt.imsave(c_rgb_path, camera_rgb)
                plt.imsave(os.path.join(image_save_dir, f"{camera_type}_depth_{str(step_time)}.jpg"), camera_depth)
                log.info(f"Images have been saved in {c_rgb_path}.")
            except Exception as e:
                log.error(f"Error in saving camera image: {e}")

class VLNDataLoader(Dataset):
    def __init__(self, args, sim_config, split, filter_same_trajectory=False):
        self.args = args
        self.sim_config = sim_config
        self.batch_size = args.settings.batch_size
        if "sample_episodes" in args.settings.mode:
            self.data, self._scans = load_gather_data(args, split, filter_same_trajectory=filter_same_trajectory, filter_stairs=args.settings.filter_stairs)
        else:
            self.data, self._scans = load_data(args, split)
        self.robot_type = sim_config.config.tasks[0].robots[0].type
        for cand_robot in args.robots:
            if cand_robot["name"] == self.robot_type:
                self.robot_offset = np.array([0,0,cand_robot["z_offset"]])
                break
        if self.robot_offset is None:
            log.error("Robot offset not found for robot type %s", self.robot_type)
            raise ValueError("Robot offset not found for robot type")
        
        # process paths offset
        if "sample_episodes" in args.settings.mode:
            for scan, data in self.data.items():
                for item in data:
                    item["start_position"] += self.robot_offset
                    for i, path in enumerate(item["reference_path"]):
                        item["reference_path"][i] += self.robot_offset
        else:
            for item in self.data:
                item["start_position"] += self.robot_offset
                for i, path in enumerate(item["reference_path"]):
                    item["reference_path"][i] += self.robot_offset

        '''Init multiple list'''
        self.env_num = sim_config.config.tasks[0].env_num # only one task
        self.robot_name = sim_config.config.tasks[0].robots[0].name # only one robot
        self.all_episode_finish = False
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]    
        
    def init_env(self, sim_config, headless=False):
        '''init env''' 
        self.env = BaseEnv(sim_config, headless=headless, webrtc=False)
        # self.init_env_manager()
    
    def init_robots(self):
        '''call after self.init_env'''
        self.tasks = self.env._runner.current_tasks
        self.robot_names = [list(task.robots.keys())[0] for task in self.tasks.values()]
        self.task_names = list(self.tasks.keys())
        self.robots = []
        self.isaac_robots = []
        self.robot_init_poses = []
        self.robot_init_orientations = []
        for idx, (task_name, task) in enumerate(self.tasks.items()):
            self.robots.append(task.robots[self.robot_names[idx]])
            self.isaac_robots.append(task.robots[self.robot_names[idx]].isaac_robot)
            robot_pose = self.sim_config.config.tasks[idx].robots[0].position # only one robot
            robot_orientation = self.sim_config.config.tasks[idx].robots[0].orientation

            # robot_pose = self.isaac_robots[idx].get_world_pose()
            # robot_pose = task.get_robot_poses_without_offset()
            self.robot_init_poses.append(robot_pose)
            self.robot_init_orientations.append(robot_orientation)

        self.robot_last_poses = [None]*self.env_num
        self.reset_robot(self.robot_init_poses, self.robot_init_orientations)

        self.init_cam_occunpancy_map() # init camera occupancy map
        self.init_check_robot_stuck(cur_iter=0)
    
    def init_BEVMap(self, robot_init_poses):
        '''init BEV map'''
        for i in range(self.env_num):
            self.bev_list[i] = BEVMap(self.args, robot_init_pose=robot_init_poses[i])
    
    def init_cam_occunpancy_map(self):
        # some pacakages can only be imported after app.run()
        from ..local_nav.camera_occupancy_map import CamOccupancyMap
        from ..local_nav.global_topdown_map import GlobalTopdownMap
        self.GlobalTopdownMap = GlobalTopdownMap

        self.cam_occupancy_map_local_list = [None]*self.env_num
        self.cam_occupancy_map_global_list = [None]*self.env_num
        for i in range(self.env_num):
            self.cam_occupancy_map_local_list[i] = CamOccupancyMap(self.args, self.robots[i].sensors['topdown_camera_50'])
            self.cam_occupancy_map_global_list[i] = CamOccupancyMap(self.args, self.robots[i].sensors['topdown_camera_500'])
    
    def init_env_manager(self):
        '''Init env data manager'''
        self.data_idx = 0

        self.bev_list = [None]*self.env_num
        self.surrounding_freemap_connected_list = [None]*self.env_num
        self.surrounding_freemap_list = [None]*self.env_num
        self.surrounding_freemap_camera_pose_list = [None]*self.env_num

        self.path_id_list = [None] * self.env_num
        self.end_list = [False] * self.env_num
        self.just_end_list = [True] * self.env_num
        self.success_list = [False] * self.env_num
        self.env_data_list = [None] * self.env_num
        self.env_step_start_index = [0] * self.env_num
        self.warm_up_list = [10] * self.env_num # This is for warm-up after resetting

        self.path_id_list = [None] * self.env_num
        self.paths_list = [None] * self.env_num
        self.env_action_finish_states = [False] * self.env_num
        self.nav_point_list = [0] * self.env_num
        self.topdown_maps = [None] * self.env_num

        self.stuck_last_iter = [0] * self.env_num
        self.stuck_threshold = [0] * self.env_num

        self.args.episode_path_list = [None]*self.env_num
        self.args.episode_status_info_file_list = [None]*self.env_num
        
        self.all_episode_finish = False
        self.scan_success_path_id_list = []
    
    def allocate_data(self, split, scan):
        self.scan_data = self.data[scan]
        self.sim_config.config.tasks[0].env_num = self.env_num = min(len(self.scan_data), self.env_num)
        self.init_env_manager() # update env_num according to the data length

        for idx in range(self.env_num):
            self.update_next_single_data(idx, split, scan, current_step=0, reset_robot=False)

    def get_next_single_data(self):
        if self.data_idx < len(self.scan_data):
            item = self.scan_data[self.data_idx]
            self.data_idx += 1
            return item
        self.all_episode_finish = True
        return None

    def update_next_single_data(self, env_idx, split, scan, current_step=0, reset_robot=True):
        '''Get the next single data and init all settings'''
        is_data_valid = False
        while not is_data_valid:
            '''1. Get new data'''
            new_data = self.get_next_single_data()
            if new_data is None:
                return False
            
            '''2. Create log path'''
            path_id = new_data['trajectory_id']
            episode_path = os.path.join(self.args.sample_episode_dir, split, scan, f"id_{str(path_id)}")
            self.args.episode_path_list[env_idx] = episode_path
            if os.path.exists(episode_path):
                if self.args.settings.force_sample:
                    log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled. Force to sample again.")
                    # remove the previous sampled data
                    shutil.rmtree(episode_path)
                    os.makedirs(episode_path)
                    is_data_valid = True
                else:
                    log.info(f"The episode [scan: {scan}] and [path_id: {path_id}] has been sampled. Pass.")
                    is_data_valid = False
                    continue
            else:
                is_data_valid = True
                os.makedirs(episode_path)

            '''log the data'''
            status_info = []
            status_info.append(f"trajectory id {new_data['trajectory_id']}")
            status_info.append(f"Initialized scan {scan}")
            status_info.append(f"Instruction: {new_data['instruction']['instruction_text']}")
            status_info.append(f"Start Position: {new_data['start_position']}, Start Rotation: {new_data['start_rotation']}")
            status_info.append(f"GT paths length: {len(new_data['reference_path'])}, points: {new_data['reference_path']}")

            for info in status_info:
                log.info(info)

            self.args.episode_status_info_file_list[env_idx] = os.path.join(self.args.episode_path_list[env_idx], 'status_info.txt')
            with open(self.args.episode_status_info_file_list[env_idx], 'w') as f:
                for info in status_info:
                    f.write(info + '\n')   

            '''Reset env list'''
            self.path_id_list[env_idx] = path_id
            self.env_data_list[env_idx] = new_data
            self.paths_list[env_idx] = new_data['reference_path']
            self.end_list[env_idx] = False
            self.just_end_list[env_idx] = True
            self.success_list[env_idx] = False
            self.env_action_finish_states[env_idx] = False
            self.nav_point_list[env_idx] = 0
            self.topdown_maps[env_idx] = None
            self.warm_up_list[env_idx] = 10
            self.success_list[env_idx] = False
            self.env_action_finish_states[env_idx] = False
            self.env_step_start_index[env_idx] = current_step
            self.stuck_last_iter[env_idx] = 0
            self.stuck_threshold[env_idx] = 0

            '''Reset the robot'''
            if reset_robot:
                self.reset_single_robot(env_idx, new_data['start_position'], new_data['start_rotation'])

        return True
    
    def update_cam_occupancy_map_pose(self):
        '''update camera pose'''
        robot_poses = self.get_robot_poses()
        for i in range(self.env_num):
            self.cam_occupancy_map_local_list[i].set_world_pose(robot_poses[i])
            self.cam_occupancy_map_global_list[i].set_world_pose(robot_poses[i])
    
    def get_robot_bottom_z(self):
        '''get robot bottom z'''
        robot_bottom_z_list = []
        for i in range(self.env_num):
            robot_bottom_z = self.robots[i].get_ankle_height() - self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
            robot_bottom_z_list.append(robot_bottom_z)
        return robot_bottom_z_list

    def init_omni_env(self):
        rotations_utils = importlib.import_module("omni.isaac.core.utils.rotations")
        self.quat_to_euler_angles = rotations_utils.quat_to_euler_angles
        self.euler_angles_to_quat = rotations_utils.euler_angles_to_quat
        # self.rot_utils = rotations_utils.rot_utils
        # from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

    def init_one_path(self, path_id):
        # Demo for visualizing simply one path
        for item in self.data:
            if item['trajectory_id'] == path_id:
                scene_usd_path = load_scene_usd(self.args, item['scan'])
                self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
                self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
                self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
                self.init_env(self.sim_config, headless=self.args.headless)
                self.init_omni_env()
                self.init_agents()
                self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
                log.info("Initialized path id %d", path_id)
                log.info("Scan: %s", item['scan'])
                log.info("Instruction: %s", item['instruction']['instruction_text'])
                log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
                return item
        log.error("Path id %d not found in the dataset", path_id)
        return None

    def init_one_scan(self, scan, idx=0, init_omni_env=False, reset_scene=False, save_log=False, path_id=-1):
        # for extract episodes within one scan (for dataset extraction)
        if path_id != -1:
            # debug mode
            for idx, item in enumerate(self.data[scan]):
                if item['trajectory_id'] == path_id:
                    break
        item = self.data[scan][idx]
        scene_usd_path = load_scene_usd(self.args, scan)
        self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
        self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
        self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"]
        if reset_scene:
            # reset scene without restart app
            # TODO: this not works!
            # TODO: 尝试开多个task。
            start_time = time.time()
            self.env._runner._world.clear()
            self.env._runner.add_tasks(self.sim_config.config.tasks)
            log.info(f"Reset scene {scan} without restarting app for using {((time.time()-start_time)/60):.2f} minutes.")
        elif init_omni_env:
            # start app.
            # should only be called at the first time.
            self.init_env(self.sim_config, headless=self.args.headless)
            self.init_omni_env()
        self.init_agents()
        if init_omni_env:
            self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
        
        status_info = []

        status_info.append(f"trajectory id {item['trajectory_id']}")
        status_info.append(f"Initialized scan {scan}")
        status_info.append(f"Instruction: {item['instruction']['instruction_text']}")
        status_info.append(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
        status_info.append(f"GT paths length: {len(item['reference_path'])}, points: {item['reference_path']}")

        for info in status_info:
            log.info(info)

        if save_log:
            self.args.episode_status_info_file = os.path.join(self.args.episode_path, 'status_info.txt')
            with open(self.args.episode_status_info_file, 'w') as f:
                for info in status_info:
                    f.write(info + '\n')         

        return item

    def init_multiple_episodes(self, scan, path_id_list=None, init_omni_env=False, reset_scene=False):
        '''init multiple episodes'''
        if path_id_list is not None:
            idx_list = []
            for path_id in path_id_list:
                for idx, item in enumerate(self.data[scan]):
                    if item['trajectory_id'] == path_id:
                        idx_list.append(idx)
                        break

        for i, item in enumerate(self.env_data_list):
            scene_usd_path = load_scene_usd(self.args, scan)
            self.sim_config.config.tasks[i].scene_asset_path = scene_usd_path
            self.sim_config.config.tasks[i].robots[0].position = item["start_position"] # only one robot
            self.sim_config.config.tasks[i].robots[0].orientation = item["start_rotation"]  

        if reset_scene:
            # reset scene without restart app
            # TODO: 关闭world后重启task，加载不同的scene
            start_time = time.time()
            self.env._runner._world.clear()
            self.env._runner.add_tasks(self.sim_config.config.tasks)
            log.info(f"Reset scene {scan} without restarting app for using {((time.time()-start_time)/60):.2f} minutes.")
        elif init_omni_env:
            # start app.
            # should only be called at the first time.
            self.init_env(self.sim_config, headless=self.args.headless)
            self.init_omni_env()

        self.init_robots()

        return item
    
    def get_robot_poses(self):
        robot_poses = []
        for task_name, task in self.tasks.items():
            robot_poses.append(task.get_robot_poses_without_offset())
        return robot_poses

    def set_robot_poses(self, position_list, orientation_list):
        for idx in range(self.env_num):
            self.tasks[idx].set_robot_poses_without_offset(position_list, orientation_list)
    
    def move_along_path(self, prev_position, current_position):
        ''' Move the agent along the path
        '''
        # Compute the relative orientation between the two positions
        rel_orientation = compute_rel_orientations(prev_position, current_position, return_quat=True) # TODO: this orientation may be wrong
        # Set the agent's world_pose to the current position
        next_position = current_position 
        next_orientation = rel_orientation
        self.agents.set_world_pose(next_position, next_orientation)
        log.info(f"Target Position: {next_position}, Orientation: {next_orientation}")
        # self.agents.set_world_pose(current_position, rel_orientation)
    
    def get_observations(self, add_rgb_subframes=True):
        ''' GEt observations from the sensors
        '''
        return self.env.get_observations(add_rgb_subframes=add_rgb_subframes)

    def get_camera_pose(self):
        '''
        Obtain position, orientation of the camera
        Output: position, orientation
        '''
        camera_dict = self.args.camera_list
        camera_poses = {}
        for i, (task_name, task) in enumerate(self.tasks.items()):
            task_camera_pose = {}
            for camera in camera_dict:
                task_camera_pose[camera] = task.get_camera_poses_without_offset(camera)
            camera_poses[task_name] = task_camera_pose
        return camera_poses
    
    def save_observations(self, camera_list:list, data_types=[], save_image_list=None, save_imgs=True, add_rgb_subframes=False, step_time=0):
        ''' Save observations from the agent
        '''
        # TODO
        obs = self.env.get_observations(add_rgb_subframes=add_rgb_subframes)
            
        for camera in camera_list:
            cur_obs = obs[self.task_name][self.robot_name][camera]
            for data in data_types:
                if data == "rgba":
                    data_info = cur_obs[data][...,:3]
                    save_img_flag = True
                elif data == "depth":
                    data_info = cur_obs[data]
                    max_depth = 10
                    data_info[data_info > max_depth] = 0
                    save_img_flag = True
                elif data == 'pointcloud':
                    save_img_flag = False
                if save_imgs and save_img_flag:
                    save_dir = os.path.join(self.args.log_image_dir, "obs")
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_path = os.path.join(save_dir, f"{camera}_{data}_{step_time}.jpg")
                    try:
                        plt.imsave(save_path, data_info)
                        log.info(f"Images have been saved in {save_path}.")
                    except Exception as e:
                        log.error(f"Error in saving camera image: {e}")
        return obs

    def process_pointcloud(self, camera_list: list, convert_to_local=False, add_subframes=False):
        ''' Process pointcloud for combining multiple cameras
        '''
        obs = self.env.get_observations(add_subframes=add_subframes)
        camera_info = {}
        
        for task_name, task in obs:
            for robot_name, robot in task:
                camera_positions = []
                camera_orientations = []
                camera_pc_data = []
                for camera in camera_list:
                    cur_obs = obs[task_name][robot_name][camera]
                    camera_pose = robot.sensors[camera].get_world_pose()
                    camera_position, camera_orientation = camera_pose[0], camera_pose[1]
                    camera_positions.append(camera_position)
                    camera_orientations.append(camera_orientation)
                    
                    if convert_to_local:
                        pc_local = pc_to_local_pose(cur_obs)
                        camera_pc_data.append(pc_local)
                    else:
                        camera_pc_data.append(cur_obs['pointcloud']['data'])
                
                camera_info[task_name] = {
                    'camera_positions': camera_positions,
                    'camera_orientations': camera_orientations,
                    'camera_pc_data': camera_pc_data
                }

        return camera_info
    
    def get_surrounding_free_map(self, verbose=False):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        robot_poses = self.get_robot_poses()
        for idx in range(len(self.robots)):
            surrounding_freemap, surrounding_freemap_connected = self.cam_occupancy_map_local.get_surrounding_free_map(robot_pos=robot_poses[idx][0],robot_height=1.65, verbose=verbose)
            surrounding_freemap_camera_pose = self.cam_occupancy_map_local_list[idx].topdown_camera.get_world_pose()[0]
            self.surrounding_freemap_list[idx] = surrounding_freemap
            self.surrounding_freemap_connected_list[idx] = surrounding_freemap_connected
            self.surrounding_freemap_camera_pose_list[idx] = surrounding_freemap_camera_pose
    
    def get_single_surrounding_free_map(self, idx):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        robot_poses = self.get_robot_poses()[idx]
        surrounding_freemap, surrounding_freemap_connected = self.cam_occupancy_map_local_list[idx].get_surrounding_free_map(robot_pos=robot_poses[idx][0],robot_height=1.65)
        surrounding_freemap_camera_pose = self.cam_occupancy_map_local_list[idx].topdown_camera.get_world_pose()[0]
        self.surrounding_freemap_list[idx] = surrounding_freemap
        self.surrounding_freemap_connected_list[idx] = surrounding_freemap_connected
        self.surrounding_freemap_camera_pose_list[idx] = surrounding_freemap_camera_pose

    def get_global_free_map(self, verbose=False):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        self.global_freemap_list = []
        self.global_freemap_camera_pose_list = []

        robot_poses = self.get_robot_poses()
        for idx in range(self.env_num):
            global_freemap_camera_pose = self.cam_occupancy_map_global_list[idx].topdown_camera.get_world_pose()[0] - self.tasks[self.task_names[idx]]._offset
            global_freemap, _ = self.cam_occupancy_map_global_list[idx].get_global_free_map(robot_pos=robot_poses[idx][0],robot_height=1.7, update_camera_pose=False, verbose=verbose)
            self.global_freemap_list.append(global_freemap)
            self.global_freemap_camera_pose_list.append(global_freemap_camera_pose)

        return self.global_freemap_list, self.global_freemap_camera_pose_list
    
    def update_occupancy_map(self, verbose=False):
        '''Use BEVMap to update the occupancy map based on pointcloud
        '''
        robots_bottom_z = self.get_robot_bottom_z()
        robot_poses = self.get_robot_poses()
        pointclouds_info = self.process_pointcloud(self.args.camera_list)
        for idx, (task_name, task) in enumerate(self.tasks.items()):
            self.bev_list[idx].update_occupancy_map(pointclouds_info[task_name]['camera_pc_data'],
                                                    robots_bottom_z[idx], 
                                                    add_dilation=self.args.maps.add_dilation,
                                                    verbose=verbose, 
                                                    robot_coords=robot_poses[idx][0]) 

    def get_global_free_map_single(self, env_idx, verbose=False):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        if not hasattr(self, 'global_freemap_list'):
            self.global_freemap_list = [None] * self.env_num
            self.global_freemap_camera_pose_list = [None] * self.env_num

        robot_pose = self.get_robot_poses()[env_idx]
        global_freemap_camera_pose = self.cam_occupancy_map_global_list[env_idx].topdown_camera.get_world_pose()[0] - self.tasks[self.task_names[env_idx]]._offset
        global_freemap, _ = self.cam_occupancy_map_global_list[env_idx].get_global_free_map(robot_pos=robot_pose[0],robot_height=1.7, update_camera_pose=False, verbose=verbose)
        self.global_freemap_list[env_idx] = global_freemap
        self.global_freemap_camera_pose_list[env_idx] = global_freemap_camera_pose

        return global_freemap, global_freemap_camera_pose
    
    def update_occupancy_map_single(self, env_idx, verbose=False):
        '''Use BEVMap to update the occupancy map based on pointcloud
        '''
        robots_bottom_z = self.get_robot_bottom_z()
        robot_poses = self.get_robot_poses()
        pointclouds_info = self.process_pointcloud(self.args.camera_list)

        task_name = self.task_names[env_idx].keys()[0]
        self.bev_list[env_idx].update_occupancy_map(pointclouds_info[task_name]['camera_pc_data'],
                                                robots_bottom_z[env_idx], 
                                                add_dilation=self.args.maps.add_dilation,
                                                verbose=verbose, 
                                                robot_coords=robot_poses[env_idx][0])        

    def check_robot_fall(self, agent, robots_bottom_z, pitch_threshold=35, roll_threshold=15, height_threshold=0.5):
        '''
        Determine if the robot is falling based on its rotation quaternion.
        '''
        # current_quaternion = obs['orientation']
        current_position, current_quaternion = agent.get_world_pose() # orientation
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = self.quat_to_euler_angles(current_quaternion,degrees=True)

        # Check if the pitch or roll exceeds the thresholds
        if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
            is_fall = True
            log.info(f"Robot falls down!!!")
            log.info(f"Current Position: {current_position}, Orientation: {self.quat_to_euler_angles(current_quaternion)}")
        else:
            is_fall = False
        
        # Check if the height between the robot base and the robot ankle is smaller than a threshold
        robot_ankle_z = robots_bottom_z
        robot_base_z = current_position[2]
        if robot_base_z - robot_ankle_z < height_threshold:
            is_fall = True
            log.info(f"Robot falls down!!!")
            log.info(f"Current Position: {current_position}, Orientation: {self.quat_to_euler_angles(current_quaternion)}")

        return is_fall

    def init_check_robot_stuck(self, cur_iter=0):
        ''' Init the variables for checking if the robot is stuck
        '''
        self.agent_last_pose = [None]*self.env_num
        self.agent_last_rotation = [None]*self.env_num
        self.agent_last_valid_pose = [None]*self.env_num
        self.agent_last_valid_rotation = [None]*self.env_num
        self.stuck_threshold = [0]*self.env_num
        self.stuck_last_iter = [cur_iter]*self.env_num

        robot_poses = self.get_robot_poses()
        for env_idx in range(self.env_num):
            self.init_check_single_robot_stuck(robot_poses, cur_iter, idx=env_idx)
    
    def init_check_single_robot_stuck(self, robot_poses, cur_iter=0, idx=0):
        ''' Init the variables for checking if the robot is stuck
        '''
        self.agent_last_pose[idx] = robot_poses[idx][0]
        self.agent_last_rotation[idx] = robot_poses[idx][1]
        self.agent_last_valid_pose[idx] = robot_poses[idx][0]
        self.agent_last_valid_rotation[idx] = robot_poses[idx][1]
        self.stuck_threshold[idx] = 0
        self.stuck_last_iter[idx] = cur_iter
    
    def check_robot_stuck(self, idx, agent, cur_iter, max_iter=300, threshold=0.2):
        ''' Check if the robot is stuck
        '''
        is_stuck = False
        if self.stuck_last_iter[idx] == 0:
            agent_world_pose = agent.get_world_pose()
            self.stuck_last_iter[idx] = cur_iter
            self.agent_last_pose[idx] = agent_world_pose[0]
            self.agent_last_rotation[idx] = agent_world_pose[1]
            return is_stuck

        # if not hasattr(self, 'agent_last_pose'):
        #     self.init_check_robot_stuck(cur_iter)
        #     return is_stuck

        agent_world_pose = agent.get_world_pose()
        current_pose = agent_world_pose[0] - self.tasks[self.task_names[idx]]._offset
        current_rotation = agent_world_pose[1]
        diff = np.linalg.norm(current_pose[:2] - self.agent_last_pose[idx][:2])
        self.stuck_threshold[idx] = diff

        if (cur_iter - self.stuck_last_iter[idx]) >= max_iter:
            if self.stuck_threshold[idx] < threshold:
                is_stuck = True
            else:
                self.stuck_threshold[idx] = 0
                self.stuck_last_iter[idx] = cur_iter
                self.agent_last_valid_pose[idx] = current_pose
                self.agent_last_valid_rotation[idx] = current_rotation
                self.agent_last_pose[idx] = current_pose

        return is_stuck
    
    def randomly_pick_position_from_freemap(self, idx, verbose=False):
        ''' Randomly pick a position from the free map
        '''
        if self.surrounding_freemap_connected_list[idx] is None:
            self.get_single_surrounding_free_map(idx=idx)
            return None
        
        free_map = self.surrounding_freemap_connected[idx]
        camera_pose = self.surrounding_freemap_camera_pose[idx]

        freemap_center = np.array(free_map.shape) // 2
        free_indices = np.argwhere(free_map == 1)

        # Calculate distances from the center
        distances = np.linalg.norm(free_indices - freemap_center, axis=1)

        # Compute weights using a Gaussian function
        # sigma = 10 * np.std(distances)  # Standard deviation as a parameter for the Gaussian function
        # distance_weights = np.exp(-distances**2 / (2 * sigma**2))
        distance_weights = 1 / (1 + distances)

        # Compute density of free space around each free position
        # kernel = np.ones((7, 7))  # Example kernel size, adjust as needed
        # density_map = convolve(free_map.astype(float), kernel, mode='constant', cval=0.0)
        # kernel_size = 200  # Increase kernel size for smoother density map
        # density_map = gaussian_filter(free_map.astype(float), sigma=kernel_size/2)
        # density_weights = density_map[free_indices[:, 0], free_indices[:, 1]]

        # Combine distance-based weights with density-based weights
        # combined_weights = distance_weights * density_weights
        combined_weights = distance_weights

        # Normalize weights to sum to 1
        combined_weights /= combined_weights.sum()

        random_index = np.random.choice(free_indices.shape[0], p=combined_weights)
        random_position = free_indices[random_index]

        random_position = self.cam_occupancy_map_local_list[idx].pixel_to_world(random_position, camera_pose)
        # random_position = [random_position[0], random_position[1], self.agent_last_valid_pose[2]]
        random_position = [random_position[0], random_position[1], self.robot_init_poses[idx][2]]

        return random_position

    def reset_robot(self, position_list, orientation_list):
        ''' Reset the robots' pose
        '''
        for idx, (task_name, task) in enumerate(self.tasks.items()):
            task.set_robot_poses_without_offset(position_list[idx], orientation_list[idx])
            isaac_robot = self.isaac_robots[idx]
            isaac_robot.set_joint_velocities(np.zeros(len(isaac_robot.dof_names)))
            isaac_robot.set_joint_positions(np.zeros(len(isaac_robot.dof_names)))
            self.robot_last_poses[idx] = position_list[idx]
    
    def reset_single_robot(self, idx, position, orientation):
        ''' Reset a single robot's pose
        '''
        self.tasks[self.task_names[idx]].set_single_robot_poses_without_offset(position, orientation)
        self.isaac_robots[idx].set_joint_velocities(np.zeros(len(self.isaac_robots[idx].dof_names)))
        self.isaac_robots[idx].set_joint_positions(np.zeros(len(self.isaac_robots[idx].dof_names)))
        self.isaac_robots[idx].set_joint_efforts(np.zeros(len(self.isaac_robots[idx].dof_names)))
        self.robot_last_poses[idx] = position
        self.robot_init_poses[idx] = position
        self.robot_init_orientations[idx] = orientation

        # reset the camera
        robot_pos = self.get_robot_poses()[idx][0]
        self.cam_occupancy_map_global_list[idx].set_world_pose(robot_pos)
    
    def check_and_reset_robot(self, cur_iter, update_freemap=False, verbose=False, reset=False):
        is_fall_list = [None]*self.env_num
        is_stuck_list = [None]*self.env_num
        robots_bottom_z_list = self.get_robot_bottom_z()
        for idx, (isaac_robot, robots_bottom_z) in enumerate(zip(self.isaac_robots, robots_bottom_z_list)):
            if self.end_list[idx]:
                is_fall_list[idx] = True
                is_stuck_list[idx] = True
                continue
            is_fall = self.check_robot_fall(isaac_robot, robots_bottom_z)
            is_fall_list[idx] = is_fall
            is_stuck = self.check_robot_stuck(idx, isaac_robot, cur_iter=cur_iter, max_iter=1000, threshold=0.2)
            is_stuck_list[idx] = is_stuck

            if (not is_fall) and (not is_stuck):
                if update_freemap:
                    self.get_surrounding_free_map(verbose=verbose) # update the surrounding_free_map
                # return False
            else:
                if verbose:
                    if is_fall:
                        log.error(f"The {idx}-th Robot falls down.")
                    if is_stuck:
                        log.error(f"The {idx}-th Robot is stuck.")

                if reset:
                    random_position = self.randomly_pick_position_from_freemap()
                    # self.reset_robot(random_position, self.agent_last_valid_rotation)
                    self.reset_single_robot(idx, random_position, self.robot_init_orientations[idx])
                    log.info(f"Reset robot pose to {random_position}.")
                # return True
        
        status_abnormal_list = [fall or stuck for fall, stuck in zip(is_fall_list, is_stuck_list)]
        return status_abnormal_list, is_fall_list, is_stuck_list

    def calc_env_action_offset(self, env_actions, action_name):
        robot_type = self.robot_names[0].split('_')[0]
        env_actions_new = deepcopy(env_actions)
        for idx, action in enumerate(env_actions):
            env_actions_new[idx][robot_type][action_name][0][0] += np.array(self.tasks[self.task_names[idx]]._offset)
        
        return env_actions_new
    
    def calc_single_env_action_offset(self, env_idx, exe_path):
        task_name = self.task_names[env_idx]
        exe_path_new = deepcopy(exe_path)
        for idx in range(len(exe_path)):
            exe_path_new[idx] += np.array(self.tasks[task_name]._offset)
        
        return exe_path_new
    
    def episode_end_setting(self, scan, env_idx, reason):
        '''record the episode ending status
        :reason: fall / stuck / maximum step / success
        '''
        if reason == 'success':
            self.end_list[env_idx] = True
            self.success_list[env_idx] = True
            self.scan_success_path_id_list.append(self.path_id_list[env_idx])
            log.info(f"[Success] Scan: {scan}, Path_id: {self.path_id_list[env_idx]}. The robot has finished this episode !!!")
            if self.just_end_list[env_idx]:
                with open(self.args.episode_status_info_file_list[env_idx], 'a') as f:
                    f.write(f"Episode finished: {self.success_list[env_idx]}\n")
                self.just_end_list[env_idx] = False
        
        elif reason in ['fall', 'stuck', 'maximum step', 'path planning']:
            self.end_list[env_idx] = True
            self.success_list[env_idx] = False
            self.env_action_finish_states[env_idx] = True
            log.error(f"[Fail: {reason}] Scan: {scan}, Path_id: {self.path_id_list[env_idx]}.")
            if self.just_end_list[env_idx]:
                with open(self.args.episode_status_info_file_list[env_idx], 'a') as f:
                    f.write(f"Episode finished: Failed. {reason}\n")
                self.just_end_list[env_idx] = False
        else:
            raise KeyError



