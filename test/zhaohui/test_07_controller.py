from vln.src.v2.dataloader.base import BasePathKeyDataloader
from vln.src.v2.envs.base import BaseSingleScanEnv
import numpy as np
from grutopia.core.config import SimulatorConfig
import math

def _get_action_state(obs, action_name):
    for env_idx, (task_name, task) in enumerate(obs.items()):
        for robot_name, robot in task.items():
            action_state = robot[action_name]
            return action_state['finished']
    return False

split_data_types = ['val_unseen'] #'val_seen'
project_path = '/ssd/zhaohui/workspace/w61_grutopia_0102'
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
robot_offset = np.array([0.   , 0.   , 1.05])
filter_same_trajectory=False
ckpt_name="ckpt.cma"
name = '20250102_ckpt_cma'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_eval.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
headless=False
scene_asset_path='/ssd/share/Matterport3D/data/v1/scans/2azQ1b91cZZ/matterport_mesh/7812e14df5e746388ff6cfe8b043950a/fixed.usd'

dataloader = BasePathKeyDataloader(
    base_data_dir,
    split_data_types,
    robot_offset,
    filter_same_trajectory,
)
path_key_data = dataloader.path_key_data
path_key_scan = dataloader.path_key_scan
path_key_split = dataloader.path_key_split

path = path_key_data['57_10']
start_position = path['start_position']
start_rotation = path['start_rotation']


env = BaseSingleScanEnv(
    sim_config,
    scene_asset_path,
    start_position,
    start_rotation,
    headless,
)
env.load_scan_and_robot()

from omni.isaac.core.utils.rotations import quat_to_euler_angles,euler_angles_to_quat

env.warm_up(100)

# 连续 10 次前进
for i in range(20):
    robot_position_s, robot_rotation_s = env.task.get_robot_poses_without_offset()
    _, _, yaw_s = quat_to_euler_angles(robot_rotation_s)
    action = [{'h1': {'move_by_descrete': [2]}}]
    finish_state = False
    while not finish_state:
        obs = env.env.step(actions=action, add_rgb_subframes=False, render=False)
        finish_state = _get_action_state(obs, 'move_by_descrete')
    
    robot_position_e, robot_rotation_e = env.task.get_robot_poses_without_offset()
    _, _, yaw_e = quat_to_euler_angles(robot_rotation_e)
    distance = np.linalg.norm(robot_position_s[:2] - robot_position_e[:2])
    distance = round(distance,2)
    yaw_diff = abs(yaw_s - yaw_e)
    angle = round(yaw_diff * (180 / math.pi), 2)
    print(f"[index:{i}] 前进距离，{distance} 米, 角度变化：{angle}°")

env.env.simulation_app.close()