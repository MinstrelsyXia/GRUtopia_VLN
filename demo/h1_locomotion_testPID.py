import torch
from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
import matplotlib.pyplot as plt

# file_path = './GRUtopia/demo/configs/h1_locomotion.yaml'
file_path = './demo/configs/h1_locomotion_test.yaml'
sim_config = SimulatorConfig(file_path)

headless = True
webrtc = False

if is_in_container():
    # headless = True
    # webrtc = True
    webrtc = False

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

from grutopia.core.util import log

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name

# path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
# path = [(1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
path = [(1.0, 1.0, 0.0), (2.0, 0.0, 0.0), (1,-1,0), (0.0, 0.0, 0.0)]
# path = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0), (3.0, 1.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
# path = [(-16.13786617, -0.057196334, 0.9983636), (-15.83261376, -0.47610778, 0.8799999), (-15.60473542, -0.23504831, 0.9733878)]

i = 0

move_action = {'move_along_path': [path]}
path_idx = 0
move_to_point_action = {'move_to_point': [path[path_idx]]}
move_to_point_by_pid_action = {'move_to_point_by_pid': [path[path_idx]]}
rotate_action = {'rotate': [euler_angles_to_quat(np.array([0, 0, np.pi]))]}

forward_action = {'move_by_speed': [1, 0, 0]}
rotate_action = {'move_by_speed': [0, 0, 1]}
mixed_action = {'move_by_speed': [1, 0, 1]}



path_finished = False
actions = {'h1': move_to_point_by_pid_action}

distance_interval = 1.0
rotation_interval = np.pi/2
record_steps = []
record_positions = []
record_orientations = []
record_yaw = []
record_distance_errors = []
record_yaw_errors = []
trajectory_x = []
trajectory_y = []

# Plot the errors
def plot_figure():
    plt.figure(figsize=(15, 5))

    # Plot trajectory with orientation arrows
    plt.subplot(1, 3, 1)
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, 'r--o', label='Planned Path')
    plt.plot(trajectory_x, trajectory_y, 'b-', label='Actual Trajectory')

    # Add orientation arrows (plot every nth point to avoid cluttering)
    n = 4  # Adjust this value to change arrow density
    for i in range(0, len(trajectory_x), n):
        if i < len(record_yaw):
            dx = 0.2 * np.cos(record_yaw[i])  # Arrow length in x direction
            dy = 0.2 * np.sin(record_yaw[i])  # Arrow length in y direction
            plt.arrow(trajectory_x[i], trajectory_y[i], dx, dy,
                    head_width=0.05, head_length=0.1, fc='g', ec='g', alpha=0.5)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Robot Trajectory with Orientation')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Make the plot aspect ratio 1:1

    # Plot distance error
    plt.subplot(1, 3, 2)
    plt.plot(record_steps, record_distance_errors)
    plt.xlabel('Steps')
    plt.ylabel('Distance Error (m)')
    plt.title('Distance Error vs Steps')
    plt.grid(True)

    # Plot yaw error
    plt.subplot(1, 3, 3)
    plt.plot(record_steps, record_yaw_errors)
    plt.xlabel('Steps')
    plt.ylabel('Yaw Error (rad)')
    plt.title('Yaw Error vs Steps')
    plt.grid(True)

    plt.tight_layout()
    save_path = './locomotion_errors.png'
    plt.savefig(save_path)
    plt.show()

    print(f"img has been saved to {save_path}")

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
        # record_steps.append(i)
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
    
    if not path_finished:
        path_finished = obs[task_name][robot_name]['move_to_point_by_pid'].get('finished', False)
        if path_finished:
            log.info('action move_to_point finished')
            path_idx += 1
            if path_idx < len(path):
                move_to_point_action = {'move_to_point_by_pid': [path[path_idx]]}
                actions['h1'] = move_to_point_action
            else:
                print('path finished')
                break
            path_finished = False

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

    # if delta_distance > distance_interval:
    #     print(i)
    #     print('distance: {}'.format(delta_distance))
    #     print('yaw: {}'.format(delta_yaw)) 
    #     record_steps.append(i)
    #     record_positions.append(current_position)
    #     record_orientations.append(orientation)
    #     record_yaw.append(delta_yaw)
    #     distance_interval += 1.0
    
    # if abs(delta_yaw) > rotation_interval:
    #     print(i)
    #     print('yaw: {}'.format(delta_yaw))
    #     record_steps.append(i)
    #     record_positions.append(current_position)
    #     record_orientations.append(orientation)
    #     record_yaw.append(delta_yaw)
    #     start_yaw = yaw

    # Calculate errors
    if i % 20 == 0:
        if path_idx < len(path):
            target_position = np.array(path[path_idx])
            current_position = np.array(current_position)
            distance_error = np.linalg.norm(target_position - current_position)
            
            # Calculate target yaw (angle towards goal)
            direction = target_position - current_position
            target_yaw = np.arctan2(direction[1], direction[0])
            yaw_error = torch.tensor(yaw - target_yaw)
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            
            # Record errors and trajectory
            record_steps.append(i)
            record_distance_errors.append(distance_error)
            record_yaw_errors.append(yaw_error.item())
            record_yaw.append(yaw)
            trajectory_x.append(current_position[0])
            trajectory_y.append(current_position[1])

# env.simulation_app.close()


plot_figure()