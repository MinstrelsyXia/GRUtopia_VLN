import os,sys
import argparse
import numpy as np
import torch
import json
import gzip
import copy
# from scipy.spatial.transform import Rotation as R

from grutopia.core.util.log import log

def euler_angles_to_quat(angles, degrees=False):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.

    Args:
        angles (list or np.array): Euler angles [roll, pitch, yaw] in degrees.

    Returns:
        np.array: Quaternion [w, x, y, z].
    """
    r = R.from_euler('xyz', angles, degrees=degrees)
    quat = r.as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]

def quat_to_euler_angles(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat (list or np.array): Quaternion [w, x, y, z].

    Returns:
        np.array: Euler angles [roll, pitch, yaw] in degrees.
    """
    reordered_quat = [quat[1], quat[2], quat[3], quat[0]]
    r = R.from_quat(reordered_quat)
    angles = r.as_euler('xyz', degrees=True)
    return angles

def compute_rel_orientations(prev_position, current_position, return_quat=False):
    """
    Compute the relative orientation between two positions.

    Args:
        prev_position (np.array): Previous position [x, y, z].
        current_position (np.array): Current position [x, y, z].

    Returns:
        np.array: Relative orientation [roll, pitch, yaw] in degrees.
    """
    # Compute the relative orientation between the two positions
    current_position = np.array(current_position) if isinstance(current_position, list) else current_position
    prev_position = np.array(prev_position) if isinstance(prev_position, list) else prev_position
    diff = current_position - prev_position
    yaw = np.arctan2(diff[1], diff[0]) * 180 / np.pi
    if return_quat:
        return np.array(euler_angles_to_quat([0, 0, yaw]))
    else:
        return np.array([0, 0, yaw])

def dict_to_namespace(d):
    ns = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            value = dict_to_namespace(value)
        setattr(ns, key, value)
    return ns

def extract_best_eval_results(log_file, split):
    results = {'best_spl':-1, 'best_sr':-1, 
               'best_spl_index':0, 'best_sr_index':0, 
               'best_spl_sr': -1, 'best_spl_sr_index': 0,
               'best_spl_sr_spl': -1, 'best_spl_sr_sr': -1}

    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            if f"Best {split} SPL and SR" in line:
                parts = line.split()
                cur_best_spl_sr = float(parts[7])
                if cur_best_spl_sr > results['best_spl_sr']:
                    results['best_spl_sr'] = cur_best_spl_sr
                    results['best_spl_sr_index'] = int(parts[-7])
                    results['best_spl_sr_spl'] = float(parts[-4])
                    results['best_spl_sr_sr'] = float(parts[-1])

            elif f"Best {split} SPL" in line:
                parts = line.split()
                cur_best_spl = float(parts[-4])
                if cur_best_spl > results['best_spl']:
                    results['best_spl'] = cur_best_spl  # The value before "at"
                    results['best_spl_index'] = int(parts[-1])   # The index value at the end
            elif f"Best {split} SR" in line:
                parts = line.split()
                cur_best_sr = float(parts[-4])
                if cur_best_sr > results['best_sr']:
                    results['best_sr'] = cur_best_sr
                    results['best_sr_index'] = int(parts[-1])   
    else:
        print(f"Log file {log_file} does not exist.")

    return results

def get_delta(actions):
    if isinstance(actions, torch.Tensor):
        # Proceed with 2D case
        if len(actions.shape) == 2:
            ex_actions = torch.cat([torch.zeros((1, actions.shape[-1]), device=actions.device), actions], dim=0)
            # Regular difference for all dimensions except the last one
            delta = ex_actions[1:, :-1] - ex_actions[:-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[1:, -1] - ex_actions[:-1, -1]
            angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
            # Combine regular and angular differences
            delta = torch.cat([delta, angle_delta.unsqueeze(-1)], dim=-1)
        else:
            # For higher dimensions (batch dimension)
            ex_actions = torch.cat([torch.zeros((actions.shape[0], 1, actions.shape[-1]), device=actions.device), actions], dim=1)
            # Regular difference for all dimensions except the last one
            delta = ex_actions[:, 1:, :-1] - ex_actions[:, :-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[:, 1:, -1] - ex_actions[:, :-1, -1]
            angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
            # Combine regular and angular differences
            delta = torch.cat([delta, angle_delta.unsqueeze(-1)], dim=-1)
    elif isinstance(actions, np.ndarray):
        if len(actions.shape) == 2:
            ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
            # Regular difference for all dimensions except the last one
            delta = ex_actions[1:, :-1] - ex_actions[:-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[1:, -1] - ex_actions[:-1, -1]
            angle_delta = np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
            # Combine regular and angular differences
            delta = np.concatenate([delta, angle_delta[:, np.newaxis]], axis=-1)
        else:
            ex_actions = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1)
            # Regular difference for all dimensions except the last one
            delta = ex_actions[:, 1:, :-1] - ex_actions[:, :-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[:, 1:, -1] - ex_actions[:, :-1, -1]
            angle_delta = np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
            # Combine regular and angular differences
            delta = np.concatenate([delta, angle_delta[..., np.newaxis]], axis=-1)
    
    return delta


def map_action_to_2d(delta_actions):
    actions_2d = torch.zeros((delta_actions.shape[0], 2))
    for a_idx, action in enumerate(delta_actions):
        if action[2] > 0:
            # turn right
            actions_2d[a_idx] = [0, 1]
        elif action[2] < 0:
            # turn left
            actions_2d[a_idx] = [0, -1]
        elif action[0] == action[1] == action [2] == 0:
            # stop
            actions_2d[a_idx] = [0, 0]
        else:
            # forward
            actions_2d[a_idx] = [1,0]
    return actions_2d

def get_action(diffusion_output, action_stats, cumsum=True):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    
    ndeltas = diffusion_output
    # action_dim = diffusion_output.shape[-1]
    # ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, action_dim)
    
    # ndeltas = ndeltas.detach().cpu().numpy() # TODO: optimize
    ndeltas = unnormalize_data(ndeltas, action_stats)
    if cumsum:
        actions = torch.cumsum(ndeltas, dim=1) # This get the relative actions (not delta) from the diffusion output
    else:
        actions = ndeltas
    return actions.float()

# normalize data
def get_data_stats(data):
    data_xy = data[:,:2]
    data_xy = data_xy.reshape(-1,data_xy.shape[-1])
    stats = {
        'min': np.min(data_xy, axis=0),
        'max': np.max(data_xy, axis=0)
    }
    return stats

def normalize_data(data, stats, device=None):
    if device is not None:
        if isinstance(stats['min'], np.ndarray):
            stats['min'] = torch.from_numpy(stats['min'])
            stats['max'] = torch.from_numpy(stats['max'])
            stats['min'] = stats['min'].to(device)
            stats['max'] = stats['max'].to(device)
    
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    # else:
    #     ndata = (data[:, :2] - stats['min'][:2]) / (stats['max'][:2] - stats['min'][:2])
    #     ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    if isinstance(ndata, torch.Tensor):
        device = ndata.device

    if len(ndata.shape) == 3:
        ndata_part = (ndata + 1) / 2
    else:
        ndata_part = (ndata[:, :2] + 1) / 2
    data = ndata_part * (stats['max'].to(device) - stats['min'].to(device)) + stats['min'].to(device)
    
    # if len(ndata.shape) == 3:
    #     data = torch.cat([data, ndata[:, 2:]], dim=1)
    return data


def action_reduce(action_mask, unreduced_loss: torch.Tensor):
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
    return (unreduced_loss * action_mask).mean() / (action_mask.float().mean() + 1e-2)

def yaw_rotmat(yaw: float) -> torch.Tensor:
    return torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw), 0.0],
            [torch.sin(yaw), torch.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    
def to_local_coords(
    positions, curr_pos, curr_yaw: float
):
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    if isinstance(positions, torch.Tensor):
        return (positions - curr_pos).matmul(rotmat)
    else:
        return (positions - curr_pos).dot(rotmat)


def _compute_actions(globalgps, yaws, curr_time, fill_mode, len_traj_pred, waypoint_spacing, learn_angle, metric_waypoint_spacing, num_action_params,normalize=False):
    start_index = curr_time
    end_index = curr_time + len_traj_pred * waypoint_spacing + 1
    yaw = yaws[start_index:end_index:waypoint_spacing]
    globalgps = globalgps[:, [0, 2]]
    positions = globalgps[start_index:end_index:waypoint_spacing]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (len_traj_pred + 1,):
        const_len = len_traj_pred + 1 - yaw.shape[0]
        if fill_mode == 'constant':
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)
        elif fill_mode == 'zero':
            yaw = np.concatenate([yaw, np.zeros(const_len)])
            positions = np.concatenate([positions, np.zeros((const_len, 2))], axis=0)

    assert yaw.shape == (len_traj_pred + 1,), f"{yaw.shape} and {(len_traj_pred + 1,)} should be equal"
    assert positions.shape == (len_traj_pred + 1, 2), f"{positions.shape} and {(len_traj_pred + 1, 2)} should be equal"

    waypoints = to_local_coords(positions, positions[0], yaw[0])

    assert waypoints.shape == (len_traj_pred + 1, 2), f"{waypoints.shape} and {(len_traj_pred + 1, 2)} should be equal"

    # if learn_angle:
    yaw = yaw[1:] - yaw[0]
    actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
    # else:
    #     actions = waypoints[1:]
    
    if normalize:
        actions[:, :2] /= metric_waypoint_spacing * waypoint_spacing

    if learn_angle:
        assert actions.shape == (len_traj_pred, num_action_params), f"{actions.shape} and {(len_traj_pred, num_action_params)} should be equal"

    return actions

class FixedLengthStack:
    def __init__(self, max_size):
        self.max_size = max_size
        self.stack = []

    def push(self, item):
        if len(self.stack) >= self.max_size:
            self.stack.pop(0)  # Remove the oldest item
        self.stack.append(item)  # Add the new item

    def get_stack(self, reverse=False):
        if reverse:
            return self.reverse()
        else:
            return self.stack
    
    def reverse(self):
        return self.stack[::-1] # without modifying the original stack

def load_dataset(dataset_root_dir, split, logger=None):
    ''' Load data based on VLN-CE
    '''
    load_data = {}
    with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for item in data["episodes"]:
            item["start_position"] = [item["start_position"][0], -item["start_position"][2], item["start_position"][1]]
            item["start_rotation"] = [-item["start_rotation"][3], item["start_rotation"][0], item["start_rotation"][2], -item["start_rotation"][1]] # [x,y,z,-w] => [w,x,y,z]
            item["scan"] = item["scene_id"].split("/")[1]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], -path[2], path[1]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data[str(item['trajectory_id'])] = item
    if logger is not None:
        logger.info(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data