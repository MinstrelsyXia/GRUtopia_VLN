# Author: w61
# Date: 2024.7.19
''' Class to load dataset for VLN
'''

import os
import gzip
import json
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from grutopia.core.util.log import log
from grutopia.core.env import BaseEnv

from ..utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations
from ..local_nav.pointcloud import generate_pano_pointcloud_local, pc_to_local_pose
from ..local_nav.BEVmap import BEVMap

def load_data(args, splits):
    ''' Load data based on VLN-CE
    '''
    dataset_root_dir = args.datasets.base_data_dir
    load_data = []
    total_scans = []
    for split in splits:
        with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
            data = json.load(f)
            for item in data["episodes"]:
                item["original_start_position"] = copy.copy(item["start_position"])
                item["original_start_rotation"] = copy.copy(item["start_rotation"])
                item["start_position"] = [item["original_start_position"][0], -item["original_start_position"][2], item["original_start_position"][1]]
                item["start_rotation"] = [-item["original_start_rotation"][3], item["original_start_rotation"][0], item["original_start_rotation"][1], item["original_start_rotation"][2]] # [x,y,z,-w] => [w,x,y,z]
                item["scan"] = item["scene_id"].split("/")[1]
                item["c_reference_path"] = []
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], -path[2], path[1]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
                load_data.append(item)
                total_scans.append(item["scan"])
    log.info("Loaded data with the length of %d", len(load_data))
    return load_data, set(total_scans)

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
    def __init__(self, args, sim_config):
        self.args = args
        self.sim_config = sim_config
        self.batch_size = args.settings.batch_size
        self.data, self._scans = load_data(args, args.datasets.splits)
        self.robot_type = sim_config.config.tasks[0].robots[0].type
        for cand_robot in args.robots:
            if cand_robot["name"] == self.robot_type:
                self.robot_offset = np.array([0,0,cand_robot["z_offset"]])
                break
        if self.robot_offset is None:
            log.error("Robot offset not found for robot type %s", self.robot_type)
        
        self.task_name = sim_config.config.tasks[0].name # only one task
        self.robot_name = sim_config.config.tasks[0].robots[0].name # only one robot type
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]    
        
    def init_env(self, sim_config, headless=False):
        '''init env''' 
        self.env = BaseEnv(sim_config, headless=headless, webrtc=False)
    
    def init_agents(self):
        '''call after self.init_env'''
        self.agents = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].isaac_robot
    
    def init_BEVMap(self):
        '''init BEV map'''
        self.bev = BEVMap(self.args)
    
    def init_one_path(self, path_id):
        # Demo for visualizing simply one path
        for item in self.data:
            if item['trajectory_id'] == path_id:
                scene_usd_path = load_scene_usd(self.args, item['scan'])
                self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
                self.sim_config.config.tasks[0].robots[0].position = item["start_position"] + self.robot_offset
                self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] # TODO: this seems not work
                self.init_env(self.sim_config, headless=self.args.headless)
                self.init_agents()
                log.info("Initialized path id %d", path_id)
                log.info("Scan: %s", item['scan'])
                log.info("Instruction: %s", item['instruction']['instruction_text'])
                return item
        log.error("Path id %d not found in the dataset", path_id)
        return None

    def set_agent_pose(self, position, rotation):
        self.agents.set_world_pose(position, rotation)
    
    def move_along_path(self, prev_position, current_position):
        ''' Move the agent along the path
        '''
        # Compute the relative orientation between the two positions
        rel_orientation = compute_rel_orientations(prev_position, current_position, return_quat=True) # TODO: this orientation may be wrong
        # Set the agent's world_pose to the current position
        next_position = current_position + self.robot_offset
        next_orientation = rel_orientation
        self.agents.set_world_pose(next_position, next_orientation)
        log.info(f"Target Position: {next_position}, Orientation: {next_orientation}")
        # self.agents.set_world_pose(current_position, rel_orientation)
    
    def get_observations(self, data_types):
        ''' GEt observations from the sensors
        '''
        return self.env.get_observations(data_type=data_types)
    
    def save_observations(self, camera_list:list, data_types:list, save_imgs=True, step_time=0):
        ''' Save observations from the agent
        '''
        obs = self.env.get_observations(data_type=data_types)
            
        for camera in camera_list:
            cur_obs = obs[self.task_name][self.robot_name][camera]
            for data in data_types:
                if data == "rgba":
                    data_info = cur_obs[data][...,:3]
                    save_img_flag = True
                elif data == "depth":
                    data_info = cur_obs[data]
                    save_img_flag = True
                elif data == 'pointcloud':
                    save_img_flag = False
                if save_imgs and save_img_flag:
                    image_save_dir = os.path.join(self.args.root_dir, "logs", "images")
                    if not os.path.exists(image_save_dir):
                        os.mkdir(image_save_dir)
                    save_path = os.path.join(image_save_dir, f"{camera}_{data}_{step_time}.jpg")
                    try:
                        plt.imsave(save_path, data_info)
                        log.info(f"Images have been saved in {save_path}.")
                    except Exception as e:
                        log.error(f"Error in saving camera image: {e}")
        return obs

    def process_pointcloud(self, camera_list: list, draw=False, convert_to_local=False):
        ''' Process pointcloud for combining multiple cameras
        '''
        obs = self.env.get_observations(data_type=['pointcloud', 'camera_params'])
        camera_positions = []
        camera_orientations = []
        camera_pc_data = []
        
        for camera in camera_list:
            cur_obs = obs[self.task_name][self.robot_name][camera]
            camera_pose = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].sensors[camera].get_world_pose()
            camera_position, camera_orientation = camera_pose[0], camera_pose[1]
            camera_positions.append(camera_position)
            camera_orientations.append(camera_orientation)
            
            if convert_to_local:
                pc_local = pc_to_local_pose(cur_obs)
                camera_pc_data.append(pc_local)
            else:
                camera_pc_data.append(cur_obs['pointcloud']['data'])
        
        if draw:
            combined_pcd = generate_pano_pointcloud_local(camera_positions, camera_orientations, camera_pc_data, draw=draw, log_dir=self.args.log_image_dir)
        return camera_pc_data, camera_positions, camera_orientations
        
        
    def get_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = self.env.__iter__()
            return next(self.data_iter)

    def reset_epoch(self):
        self.data_iter = self.env.__iter__()

    def get_vocab_size(self):
        return len(self.vocab)

    def get_tokenizer(self):
        return self.tokenizer

    def get_reward_path(self):
        return self.reward_path

    def get_vocab(self):
        return self.vocab

    def get_env(self):
        return self.env

    def get_splits(self):
        return self.split

    def get_batch_size(self):
        return self.args.batch_size

    def get_max_episode_len(self):
        return self.args.max_episode_len

    def get_max_instruction_len(self):
        return self.args.max_instruction_len

    def get_max_path_len(self):
        return self.args.max_path_len

    def get_max_actions(self):
        return self.args.max_actions

    def get_max_follower(self):
        return self.args.max_follower

    def get_max_follower_len(self):
        return self.args.max_follower_len

    def get_max_follower_actions(self):
        return self.args.max_follower_actions

    def get_max_follower_offset(self):
        return self.args.max_follower_offset

    def get_max_follower_angle(self):
        return self.args.max_follower_angle

    def get_max_follower_distance(self):
        return self.args.max_follower_distance

    def get_max_follower_rotations(self):
        return self.args.max_follower_rotations

    def get_max_follower_looks(self):
        return self.args.max_follower_looks

    def get_max_follower_exploration(self):
        return self.args.max_follower_exploration

    def get_max_follower_exploration_distance(self):
        return self.args.max_follower_exploration_distance