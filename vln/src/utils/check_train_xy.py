import os
import torch
import numpy as np
from vln.src.utils.read_lmdb import LmdbReader
from vln.src.utils.utils import to_local_coords, to_global_coords
import matplotlib.pyplot as plt

def compute_actions(globalgps, yaws, curr_time, len_traj_pred, waypoint_spacing, 
                   metric_waypoint_spacing=1.0, normalize=True, learn_angle=True,
                   fill_mode='constant', vis=False, save_dir=None):
    """
    Compute actions for a given trajectory at a specific time step.
    Args:
        globalgps: Global positions (N, 3)
        yaws: Global yaws (N,)
        curr_time: Current time step
        len_traj_pred: Number of future waypoints to predict
        waypoint_spacing: Spacing between waypoints
        metric_waypoint_spacing: Metric spacing for normalization
        normalize: Whether to normalize the actions
        learn_angle: Whether to include angle in actions
        fill_mode: How to fill missing waypoints ('constant' or 'zero')
        vis: Whether to visualize the waypoints
        save_dir: Directory to save visualization
    """
    start_index = curr_time
    end_index = curr_time + len_traj_pred * waypoint_spacing + 1
    yaw = yaws[start_index:end_index:waypoint_spacing]
    original_globalgps = globalgps.copy()
    globalgps = globalgps[:, [0, 1]]  # Only use x, y coordinates
    positions = globalgps[start_index:end_index:waypoint_spacing]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)
    
    if yaw.shape != (len_traj_pred + 1,):
        const_len = len_traj_pred + 1 - yaw.shape[0]
        if fill_mode == 'constant':
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.tile(positions[-1], (const_len, 1))], axis=0)
        elif fill_mode == 'zero':
            yaw = np.concatenate([yaw, np.zeros(const_len)])
            positions = np.concatenate([positions, np.zeros((const_len, 2))], axis=0)

    # Convert to local coordinates
    waypoints = to_local_coords(positions, positions[0], yaw[0])

    # Calculate relative yaw angles
    delta_yaw = yaw[1:] - yaw[0]
    # Normalize angles to [-π, π]
    delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
    
    if learn_angle:
        actions = np.concatenate([waypoints[1:], delta_yaw[:, None]], axis=-1)
    else:
        actions = waypoints[1:]
    
    if normalize:
        actions[:, :2] /= (metric_waypoint_spacing * waypoint_spacing)

    if vis:
        visualize_waypoints(waypoints, delta_yaw, curr_time, save_dir)

    return actions

def visualize_waypoints(waypoints, delta_yaw, step_idx, save_dir):
    """
    Visualize waypoints and their orientations
    """
    plt.clf()
    plt.figure(figsize=(5, 5))
    
    # Plot waypoints
    plt.scatter(waypoints[:, 0], waypoints[:, 1], 
               label='waypoints', color='red', alpha=0.5)
    
    # Plot start point
    plt.scatter(waypoints[0, 0], waypoints[0, 1], 
               color='green', alpha=1.0, label='start')
    
    # Add arrows for orientation
    arrow_length = 0.2
    start_yaw = 0
    for i in range(1, len(waypoints)):
        current_yaw = start_yaw + delta_yaw[i-1]
        dx = arrow_length * np.cos(current_yaw)
        dy = arrow_length * np.sin(current_yaw)
        
        plt.arrow(waypoints[i, 0], waypoints[i, 1], 
                 dx, dy,
                 head_width=0.05,
                 head_length=0.1,
                 fc='red',
                 ec='red',
                 alpha=0.5)
        
        plt.text(waypoints[i, 0], waypoints[i, 1],
                f'{i}', fontsize=9, color='red', ha='right')
    
    plt.title(f'Waypoints and Orientations at Step {step_idx}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Save figure
    save_path = os.path.join(save_dir, f'waypoints_step_{step_idx}.png')
    plt.savefig(save_path)
    print(f"Saved waypoints visualization to {save_path}")
    plt.close()

def main():
    # Configuration
    lmdb_path = 'data/sample_episodes/20241120_sample_episodes_full/sample_data.lmdb'
    trajectory_id = '4'  # Example trajectory ID
    save_dir = 'logs/check_waypoints'
    os.makedirs(save_dir, exist_ok=True)

    # Parameters (matching those in vlnce_dp_dataset.py)
    len_traj_pred = 9
    waypoint_spacing = 1
    metric_waypoint_spacing = 1.0
    normalize = True
    learn_angle = True

    # Read data
    reader = LmdbReader(lmdb_path)
    episode_data = reader.read_episode_data(trajectory_id)
    
    if episode_data is None:
        print(f"No data found for trajectory_id: {trajectory_id}")
        return

    # Extract positions and yaws
    positions = np.array(episode_data['episode_data']['robot_info']['position'])
    yaws = np.array(episode_data['episode_data']['robot_info']['yaw'])

    # Normalize yaws to [-π, π]
    # yaws = yaws % (2 * np.pi)
    # yaws[yaws > np.pi] -= 2 * np.pi

    # Compute actions for each time step
    total_steps = len(positions)
    all_actions = []
    
    for step in range(total_steps):
        actions = compute_actions(
            positions, yaws, step,
            len_traj_pred=len_traj_pred,
            waypoint_spacing=waypoint_spacing,
            metric_waypoint_spacing=metric_waypoint_spacing,
            normalize=normalize,
            learn_angle=learn_angle,
            vis=(step % 1 == 0),  # Visualize every 10th step
            save_dir=save_dir
        )
        all_actions.append(actions)
        
        if step % 10 == 0:
            print(f"Step {step}/{total_steps}:")
            print("Actions shape:", actions.shape)
            print("Actions:\n", actions)
            print("-" * 50)

    all_actions = np.array(all_actions)
    print("\nFinal actions shape:", all_actions.shape)

if __name__ == "__main__":
    main()
