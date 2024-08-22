# Author: w61
# Date: 2024.7.19
''' Class to load dataset for VLN
'''

import os
import gzip
import json
import copy
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import importlib
from scipy.ndimage import convolve, gaussian_filter

try:
    from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
except:
    pass

from grutopia.core.util.log import log
from grutopia.core.env import BaseEnv

from ..utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations

from ..local_nav.pointcloud import generate_pano_pointcloud_local, pc_to_local_pose
from ..local_nav.BEVmap import BEVMap
from ..local_nav.sementic_map import BEVSemMap

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

def load_gather_data(args, split):
    dataset_root_dir = args.datasets.base_data_dir
    with open(os.path.join(dataset_root_dir, "gather_data", f"{split}_gather_data.json"), 'r') as f:
        data = json.load(f)
    with open(os.path.join(dataset_root_dir, "gather_data", "env_scan.json"), 'r') as f:
        scan = json.load(f)
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
    def __init__(self, args, sim_config, split):
        self.args = args
        self.sim_config = sim_config
        self.batch_size = args.settings.batch_size
        if args.settings.mode == "sample_episodes":
            self.data, self._scans = load_gather_data(args, split)
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
        if args.settings.mode == "sample_episodes":
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

        
        self.task_name = sim_config.config.tasks[0].name # only one task
        self.robot_name = sim_config.config.tasks[0].robots[0].name # only one robot type
        
        self.bev = None
        self.surrounding_freemap_connected = None
    
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
        self.agent_last_pose = None
        self.agent_init_pose = self.sim_config.config.tasks[0].robots[0].position
        self.agent_init_rotation = self.sim_config.config.tasks[0].robots[0].orientation
        # if 'oracle' in self.args.settings.action:
        #     from pxr import Usd, UsdPhysics
        #     robot_prim = self.agents.prim
        #     physics_schema = UsdPhysics.RigidBodyAPI(robot_prim)
        #     if physics_schema:
        #         physics_schema.CreateCollisionEnabledAttr(False)
        #         log.info("In oracle mode, set robot collision to False")
        #     else:
        #         log.warning("In oracle mode, set robot collision to False failed")
    
    def init_BEVMap(self, robot_init_pose=(0,0,0)):
        '''init BEV map'''
        self.bev = BEVMap(self.args, robot_init_pose=robot_init_pose)
        # TODO：可能在其他位置调用
        self.robot_init_pose = robot_init_pose
        self.init_BEVSemMap(robot_init_pose=robot_init_pose)
    
    def init_isaac_occupancy_map(self):
        '''init Isaac Occupancy map'''
        from ..local_nav.isaac_occupancy_map import IsaacOccupancyMap
        self.isaac_occupancy_map = IsaacOccupancyMap(self.args)
    
    def init_cam_occunpancy_map(self, robot_prim="/World/env_0/robots/World/h1",start_point=[0,0,0]):
        # some pacakages can only be imported after app.run()
        from ..local_nav.camera_occupancy_map import CamOccupancyMap
        from ..local_nav.global_topdown_map import GlobalTopdownMap
        self.GlobalTopdownMap = GlobalTopdownMap
        self.cam_occupancy_map_local = CamOccupancyMap(self.args, robot_prim, start_point, local=True)
        self.cam_occupancy_map_global = CamOccupancyMap(self.args, robot_prim, start_point, local=False)
    
    def get_robot_bottom_z(self):
        '''get robot bottom z'''
        return self.env._runner.current_tasks[self.task_name].robots[self.robot_name].get_ankle_base_z()-self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
    
    @property
    def current_task_stage(self):
        return self.env._runner.current_tasks[self.task_name]._scene.stage

    def init_omni_env(self):
        rotations_utils = importlib.import_module("omni.isaac.core.utils.rotations")
        self.quat_to_euler_angles = rotations_utils.quat_to_euler_angles
        self.euler_angles_to_quat = rotations_utils.euler_angles_to_quat
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

    def init_one_scan(self, scan, idx=0, init_omni_env=False):
        # for extract episodes within one scan (for dataset extraction)
        item = self.data[scan][idx]
        scene_usd_path = load_scene_usd(self.args, scan)
        self.sim_config.config.tasks[0].scene_asset_path = scene_usd_path
        self.sim_config.config.tasks[0].robots[0].position = item["start_position"]
        self.sim_config.config.tasks[0].robots[0].orientation = item["start_rotation"] 
        if init_omni_env:
            self.init_env(self.sim_config, headless=self.args.headless)
            self.init_omni_env()
        self.init_agents()
        self.init_cam_occunpancy_map(robot_prim=self.agents.prim_path,start_point=item["start_position"]) 
        log.info("trajectory id %d", item['trajectory_id'])
        log.info("Initialized scan %s", scan)
        log.info("Instruction: %s", item['instruction']['instruction_text'])
        log.info(f"Start Position: {self.sim_config.config.tasks[0].robots[0].position}, Start Rotation: {self.sim_config.config.tasks[0].robots[0].orientation}")
        return item
    
    def get_agent_pose(self):
        return self.agents.get_world_pose()

    def set_agent_pose(self, position, rotation):
        self.agents.set_world_pose(position, rotation)
    
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
    
    def get_observations(self, data_types):
        ''' GEt observations from the sensors
        '''
        return self.env.get_observations(data_type=data_types)

    def get_camera_pose(self):
        '''
        Obtain position, orientation of the camera
        Output: position, orientation
        '''
        camera_dict = self.args.camera_list
        camera_pose = {}
        for camera in camera_dict:
            camera_pose[camera] = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].sensors[camera].get_world_pose()
        return camera_pose
    
    def save_observations(self, camera_list:list, data_types:list, save_image_list=None, save_imgs=True, step_time=0):
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
                    max_depth = 10
                    data_info[data_info > max_depth] = 0 # automatically discard unsatisfied values
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

    def save_episode_data(self, scan, path_id, total_images, camera_list:list, data_types:list, step_time=0):
        ''' Save episode data
        '''
        # make dir
        save_dir = os.path.join(self.args.sample_episode_dir, scan, f"id_{str(path_id)}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        obs = self.env.get_observations(data_type=data_types)
        camera_pose_dict = self.get_camera_pose()

        # save camera information
        for camera in camera_list:
            cur_obs = obs[self.task_name][self.robot_name][camera]
            # save pose
            camera_pose = camera_pose_dict[camera]
            pos, quat = camera_pose[0], camera_pose[1]
            pose_save_path = os.path.join(save_dir, 'poses.txt')
            with open(pose_save_path,'a') as f:
                sep =""
                f.write(f"{sep}{pos[0]}\t{pos[1]}\t{pos[2]}\t{quat[0]}\t{quat[1]}\t{quat[2]}\t{quat[3]}")
                sep = "\n"
        
            # stack RGB into memory
            rgb_info = cur_obs['rgba'][...,:3]
            depth_info = cur_obs['depth']
            max_depth = 10
            depth_info[depth_info > max_depth] = 0 

            total_images[camera].append({
                'step_time': step_time,
                'rgb': rgb_info,
                'depth': depth_info})

            # save camera_intrinsic
            camera_params = cur_obs['camera_params']
            # 构造要保存的字典
            camera_info = {
                "camera": camera,
                "step_time": step_time,
                "intrinsic_matrix": camera_params['cameraProjection'].tolist(),
                "extrinsic_matrix": camera_params['cameraViewTransform'].tolist(),
                "cameraAperture": camera_params['cameraAperture'].tolist(),
                "cameraApertureOffset": camera_params['cameraApertureOffset'].tolist(),
                "cameraFocalLength": camera_params['cameraFocalLength'],
                "robot_init_pose": self.agent_init_pose.tolist()
            }
            # print(self.agent_init_pose)
            cam_save_path = os.path.join(save_dir, 'camera_param.jsonl')

            # 将信息追加保存到 jsonl 文件
            with open(cam_save_path, 'a') as f:
                json.dump(camera_info, f)
                f.write('\n')
        
        # save robot information
        robot_info = {
            "step_time": step_time,
            "position": obs[self.task_name][self.robot_name]['position'].tolist(),
            "orientation": obs[self.task_name][self.robot_name]['orientation'].tolist()
        }
        robot_save_path = os.path.join(save_dir, 'robot_param.jsonl')

        # 将信息追加保存到 jsonl 文件
        with open(robot_save_path, 'a') as f:
            json.dump(robot_info, f)
            f.write('\n')

        return total_images

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
            combined_pcd = generate_pano_pointcloud_local(camera_positions, camera_orientations, camera_pc_data, draw=draw, log_dir=self.args.log_image_dir) # TODO：wrong code
        return camera_pc_data, camera_positions, camera_orientations
    
    def get_surrounding_free_map(self, verbose=False):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        # agent_current_pose = self.get_agent_pose()[0]
        # agent_bottom_z = self.get_robot_bottom_z()
        self.surrounding_freemap, self.surrounding_freemap_connected = self.cam_occupancy_map_local.get_surrounding_free_map(robot_pos=self.get_agent_pose()[0],robot_height=1.65, verbose=verbose)
        self.surrounding_freemap_camera_pose = self.cam_occupancy_map_local.topdown_camera.get_world_pose()[0]

    def get_global_free_map(self, verbose=False):
        ''' Use top-down orthogonal camera to get the ground-truth surrounding free map
            This is useful to reset the robot's location when it get stuck or falling down.
        '''
        self.global_freemap_camera_pose = self.cam_occupancy_map_global.topdown_camera.get_world_pose()[0]
        self.global_freemap, _ = self.cam_occupancy_map_global.get_global_free_map(robot_pos=self.get_agent_pose()[0],robot_height=1.7, update_camera_pose=False, verbose=verbose)
        return self.global_freemap, self.global_freemap_camera_pose
    
    def update_occupancy_map(self, verbose=False):
        '''Use BEVMap to update the occupancy map based on pointcloud
        '''
        return
        pointclouds, _, _ = self.process_pointcloud(self.args.camera_list) # ! 直接拿到的pointcloud個數比depth小
        robot_ankle_z = self.get_robot_bottom_z()
        self.bev.update_occupancy_map(pointclouds, robot_ankle_z, add_dilation=self.args.maps.add_dilation, verbose=verbose, robot_coords=self.get_agent_pose()[0])
        # TODO: 可能在main里调用
        # self.update_semantic_map(verbose=verbose)

    
    def update_semantic_map(self,verbose=False):
        # single robot
        obs = self.get_observations(data_types=['rgba', 'depth', 'camera_params'])
        cur_obs = obs[self.task_name][self.robot_name]
        camera_pose = self.get_camera_pose()
        self.bev_sem.update_semantic_map(obs_tr=cur_obs,camera_dict=self.args.camera_list,camera_poses=camera_pose,verbose=verbose,robot_coords=self.get_agent_pose()[0])
        
    
    def get_camera_pose(self):
        camera_dict = self.args.camera_list
        camera_pose = {}
        for camera in camera_dict:
            camera_pose[camera] = self.env._runner.current_tasks[self.task_name].robots[self.robot_name].sensors[camera].get_world_pose()
        return camera_pose

    def init_BEVSemMap(self, robot_init_pose=(0,0,0)):
        '''init BEVSem map'''
        self.bev_sem = BEVSemMap(self.args, robot_init_pose=robot_init_pose)
        self.robot_init_pose=robot_init_pose

    def check_robot_fall(self, agent, pitch_threshold=35, roll_threshold=15, height_threshold=0.5, adjust=False, initial_pose=None, initial_rotation=None):
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
        robot_ankle_z = self.get_robot_bottom_z()
        robot_base_z = self.get_agent_pose()[0][2]
        if robot_base_z - robot_ankle_z < height_threshold:
            is_fall = True
            log.info(f"Robot falls down!!!")
            log.info(f"Current Position: {current_position}, Orientation: {self.quat_to_euler_angles(current_quaternion)}")

        return is_fall
    
    def check_robot_stuck(self, cur_iter, max_iter=300, threshold=0.2):
        ''' Check if the robot is stuck
        '''
        is_stuck = False
        if self.agent_last_pose is None:
            self.agent_last_valid_pose = self.get_agent_pose()[0]
            self.agent_last_pose, self.agent_last_rotation = self.get_agent_pose()
            self.agent_last_valid_pose = self.agent_last_pose
            self.agent_last_valid_rotation = self.agent_last_rotation
            self.stuck_threshold = 0
            self.stuck_last_iter = cur_iter
            return is_stuck

        current_pose, current_rotation = self.get_agent_pose()
        diff = np.linalg.norm(current_pose - self.agent_last_pose)
        self.stuck_threshold += diff

        if (cur_iter - self.stuck_last_iter) >= max_iter:
            if self.stuck_threshold < threshold:
                is_stuck = True
            else:
                self.stuck_threshold = 0
                self.stuck_last_iter = cur_iter
                self.agent_last_valid_pose = current_pose
                self.agent_last_valid_rotation = current_rotation

        self.agent_last_pose = current_pose

        return is_stuck
    
    def randomly_pick_position_from_freemap(self, verbose=False):
        ''' Randomly pick a position from the free map
        '''
        if self.surrounding_freemap_connected is None:
            self.get_surrounding_free_map(verbose=verbose)
            return None
        
        free_map = self.surrounding_freemap_connected
        camera_pose = self.surrounding_freemap_camera_pose

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

        random_position = self.cam_occupancy_map_local.pixel_to_world(random_position, camera_pose)
        # random_position = [random_position[0], random_position[1], self.agent_last_valid_pose[2]]
        random_position = [random_position[0], random_position[1], self.agent_init_pose[2]]

        return random_position

    def reset_robot(self, position, rotation):
        ''' Reset the robot's pose
        '''
        self.agents.set_world_pose(position, rotation)
        self.agent_last_pose = position
        self.agents.set_joint_velocities(np.zeros(len(self.agents.dof_names)))
        self.agents.set_joint_positions(np.zeros(len(self.agents.dof_names)))
    
    def check_and_reset_robot(self, cur_iter, update_freemap=False, verbose=False):
        is_fall = self.check_robot_fall(self.agents, adjust=False)
        is_stuck = self.check_robot_stuck(cur_iter=cur_iter, max_iter=300, threshold=0.2)
        if (not is_fall) and (not is_stuck):
            if update_freemap:
                self.get_surrounding_free_map(verbose=verbose) # update the surrounding_free_map
            return False
        else:
            if is_fall:
                log.info("Robot falls down. Reset robot pose.")
            if is_stuck:
                log.info("Robot is stuck. Reset robot pose.")
            random_position = self.randomly_pick_position_from_freemap()
            # self.reset_robot(random_position, self.agent_last_valid_rotation)
            self.reset_robot(random_position, self.agent_init_rotation)
            log.info(f"Reset robot pose to {random_position}.")
            return True