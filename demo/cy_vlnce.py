# Author: w61
# Date: 2024.7.17
''' Demo for loading vlnce dataset and intialize the Cylinder robot
'''
import os,sys
import gzip
import json
import math
import numpy as np
import argparse

from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
from grutopia.core.util.log import log

# from grutopia_extension.utils import get_stage_prim_paths

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)

parser = argparse.ArgumentParser(description="Demo for loading vlnce dataset and intialize the H1 robot.")
parser.add_argument("--env", default="val_seen", type=str, help="The split of the dataset", choices=['train', 'val_seen', 'val_unseen'])
parser.add_argument("--path_id", default=5593, type=int, help="The number of path id")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--test_verbose", action="store_true", default=False)
parser.add_argument("--wait", action="store_true", default=False)
parser.add_argument("--config_file", type=str, default="./GRUtopia/demo/configs/cy_vlnce.yaml")
args = parser.parse_args()

# file_path = './GRUtopia/demo/configs/cy_vlnce.yaml'
file_path = args.config_file
sim_config = SimulatorConfig(file_path)

def euler_angles_to_quat(angles):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.

    Args:
        angles (list or np.array): Euler angles [roll, pitch, yaw] in degrees.

    Returns:
        np.array: Quaternion [x, y, z, w].
    """
    r = R.from_euler('xyz', angles, degrees=True)
    return r.as_quat()

def quat_to_euler_angles(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat (list or np.array): Quaternion [x, y, z, w].

    Returns:
        np.array: Euler angles [roll, pitch, yaw] in degrees.
    """
    r = R.from_quat(quat)
    angles = r.as_euler('xyz', degrees=True)
    return angles


def load_data(file_path, path_id=None, verbose=False):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    target_item = None
    if path_id is not None:
        for item in data['episodes']:
            if item['trajectory_id'] == path_id:
                target_item = item
                break
    if target_item is None:
        print(f"Path ID {path_id} is invalid and randomly set a path id")
        target_item = data['episodes'][0]
    scan = target_item['scene_id'].split('/')[1]
    # start_position = [target_item['start_position'][0]-0.1, -target_item['start_position'][2]-0.1, target_item['start_position'][1]+1.5]
    start_position = [target_item['start_position'][0], -target_item['start_position'][2], target_item['start_position'][1]+1.2]
    start_rotation = [-target_item['start_rotation'][3], target_item['start_rotation'][0], target_item['start_rotation'][1], target_item['start_rotation'][2]] # [x,y,z,-w] => [w,x,y,z]
    if verbose: 
        log.info(f"Scan: {scan}")
        log.info(f"Initial Position: {start_position}")
        log.info(f"Initial Rotation: {start_rotation}")
    return target_item, scan, start_position, start_rotation

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
    

data_item, data_scan, start_position, start_rotation = load_data(sim_config.config_dict['datasets'][0]['base_data_dir']+f"/{args.env}/{args.env}.json.gz", 
                                                                args.path_id, verbose=args.test_verbose)

find_flag = False
for root, dirs, files in os.walk(sim_config.config_dict['datasets'][0]['mp3d_data_dir']+f"/{data_scan}"):
    for file in files:
        if file.endswith(".usd") and "non_metric" not in file and "isaacsim_" in file:
            scene_usd_path = os.path.join(root, file)
            find_flag = True
            break
    if find_flag:
        break

sim_config.config.tasks[0].scene_asset_path = scene_usd_path
sim_config.config.tasks[0].robots[0].position = start_position 
sim_config.config.tasks[0].robots[0].orientation = start_rotation

headless = args.headless
webrtc = False

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

from llm_agent.utils.utils_omni import get_camera_data, get_face_to_instance_by_2d_bbox

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name
agent = env._runner.current_tasks[task_name].robots[robot_name].isaac_robot
# camera = env._runner.current_tasks[task_name].robots[robot_name].sensors['camera']
# tp_camera = env._runner.current_tasks[task_name].robots[robot_name].sensors['tp_camera']
# agent.set_world_pose

i = 0

actions = {'h1': {'move_with_keyboard': []}}
while env.simulation_app.is_running():
    i += 1
    env_actions = []
    env_actions.append(actions)
    if not args.wait:
        obs = env.step(actions=env_actions)

    if i % 100 == 0:
        obs = env._runner.get_obs()
        # obs = env.get_observations(data_type=['rgba', 'depth', 'pointcloud'])
        cur_obs = obs[task_name][robot_name]
        # is_fall = check_fall(agent, cur_obs, adjust=True, initial_pose=start_position, initial_rotation=start_rotation)
        # get_sensor_info(i, cur_obs, verbose=args.test_verbose)
        print(i)


env.simulation_app.close()
