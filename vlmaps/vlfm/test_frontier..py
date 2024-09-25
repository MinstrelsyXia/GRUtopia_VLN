import sys
sys.path.append('GRUtopia/grutopia_extension/agents/social_navigation_agent')
import os
import numpy as np
import yaml
import pickle
from PIL import Image
import jsonlines
from modules.mapping.obstacle_map import ObstacleMap
from GRUtopia.grutopia_extension.agents.social_navigation_agent.agent_utils.geometry_utils import get_intrinsic_matrix, extract_camera_pos_zyxrot

data_path = '/home/huangwensi/wensi/GRUtopia/images' 
with open('/home/huangwensi/isaac-sim-4.0.0/GRUtopia/grutopia_extension/agents/social_navigation_agent/memory_config.yaml', "r") as file:
    memory_config = yaml.load(file, Loader=yaml.FullLoader)
with open(os.path.join(data_path, 'camera_params.pkl'), 'rb') as f:
    camera_params = pickle.load(f)

in_matrix = get_intrinsic_matrix(camera_params)

min_depth = memory_config['obstacle_map'].pop('min_depth')
max_depth = memory_config['obstacle_map'].pop('max_depth')
obstacle_map = ObstacleMap(**memory_config['obstacle_map'])

for step_time in range(500, 1300, 100):
    depth = np.load(os.path.join(data_path, f'depth_{step_time}.npy'))
    camera_transform = np.load(os.path.join(data_path, f'camera_transform_{step_time}.npy'))
    position = np.load(os.path.join(data_path, f'robot_pos_{step_time}.npy'))
    orientation = np.load(os.path.join(data_path, f'robot_orient_{step_time}.npy'))


    camera_in = in_matrix
    topdown_fov = 2 * np.arctan(camera_in[0, 2] / camera_in[0, 0])
    camera_position, camera_rotation = extract_camera_pos_zyxrot(camera_transform)
    obstacle_map.update_map(
                depth,
                camera_in,
                camera_transform,
                min_depth,
                max_depth,
                topdown_fov,
                verbose=True
            )