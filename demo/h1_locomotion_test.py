import torch
from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container

# file_path = './GRUtopia/demo/configs/h1_locomotion.yaml'
file_path = './demo/configs/h1_locomotion_test.yaml'
sim_config = SimulatorConfig(file_path)

headless = True
webrtc = False

if is_in_container():
    headless = True
    webrtc = True

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

from grutopia.core.util import log

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name

path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
i = 0

move_action = {'move_along_path': [path]}
rotate_action = {'rotate': [euler_angles_to_quat(np.array([0, 0, np.pi]))]}

forward_action = {'move_by_speed': [1, 0, 0]}
rotate_action = {'move_by_speed': [0, 0, 1]}
mixed_action = {'move_by_speed': [1, 0, 1]}

path_finished = False
actions = {'h1': mixed_action}

distance_interval = 1.0
rotation_interval = np.pi/2
record_steps = []
record_positions = []
record_orientations = []
record_yaw = []

while env.simulation_app.is_running():
    i += 1
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)
    if i == 1:
        orientation = obs[task_name][robot_name]['orientation']
        yaw = quat_to_euler_angles(orientation)[2]  
        start_position = obs[task_name][robot_name]['position']
        start_yaw = torch.tensor(yaw)
        start_yaw = torch.atan2(torch.sin(start_yaw), torch.cos(start_yaw))
        print('start position of h1: {}'.format(start_position))
        print('start orientation of h1: {}'.format(quat_to_euler_angles(orientation)))
        print('start yaw of h1: {}'.format(start_yaw))
        record_steps.append(i)
        record_positions.append(start_position)
        record_orientations.append(orientation)
        record_yaw.append(0)
        continue
    
    # if not path_finished:
    #     path_finished = obs[task_name][robot_name]['move_along_path'].get('finished', False)
    #     if path_finished:
    #         log.info('start rotate')
    #         actions['h1'] = rotate_action
    #         start_rotate = True

    if i % 100 == 0:
        print(i)
        orientation = obs[task_name][robot_name]['orientation']
        yaw = quat_to_euler_angles(orientation)[2]
        print('available observations for h1: {}'.format(obs[task_name][robot_name].keys()))
        print('current position of h1:{}'.format(obs[task_name][robot_name]['position']))
        print('current orientation of h1: {}'.format(quat_to_euler_angles(orientation)))
        print('current yaw of h1: {}'.format(yaw))
    
    # compute the distance
    current_position = obs[task_name][robot_name]['position']
    delta_distance = np.linalg.norm(np.array(current_position) - np.array(start_position))
    orientation = obs[task_name][robot_name]['orientation']
    yaw = quat_to_euler_angles(orientation)[2]
    delta_yaw = torch.tensor(yaw - start_yaw)
    delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

    if delta_distance > distance_interval:
        print(i)
        print('distance: {}'.format(delta_distance))
        print('yaw: {}'.format(delta_yaw)) 
        record_steps.append(i)
        record_positions.append(current_position)
        record_orientations.append(orientation)
        record_yaw.append(delta_yaw)
        distance_interval += 1.0
    
    if abs(delta_yaw) > rotation_interval:
        print(i)
        print('yaw: {}'.format(delta_yaw))
        record_steps.append(i)
        record_positions.append(current_position)
        record_orientations.append(orientation)
        record_yaw.append(delta_yaw)
        start_yaw = yaw

env.simulation_app.close()
