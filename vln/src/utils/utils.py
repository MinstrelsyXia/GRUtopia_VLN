import os,sys
import argparse
import numpy as np
import torch
import json
import gzip
import copy
import glob
import cv2

import numpy as np
import torch

from PIL import Image
from torch import Size, Tensor
from torch import nn as nn

from collections import defaultdict
# from scipy.spatial.transform import Rotation as R

from grutopia.core.util.log import log

from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from vln.src.utils.tensor_dict import TensorDict

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
    try:
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    except Exception as e:
        ndata = (data - stats.min) / (stats.max - stats.min)
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
    
    try:
        data = ndata_part * (stats['max'].to(device) - stats['min'].to(device)) + stats['min'].to(device)
    except Exception as e:
        data = ndata_part * (stats.max.to(device) - stats.min.to(device)) + stats.min.to(device)
    
    # if len(ndata.shape) == 3:
    #     data = torch.cat([data, ndata[:, 2:]], dim=1)
    return data


def action_reduce(action_mask, unreduced_loss: torch.Tensor):
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
    return (unreduced_loss * action_mask).mean() / (action_mask.float().mean() + 1e-2)

def yaw_rotmat(yaw: float):
    try:
        R = torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), 0.0],
                [torch.sin(yaw), torch.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
    except Exception as e:
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0], 
            [0.0, 0.0, 1.0]
        ])

    return R
    
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

def to_global_coords(
    local_coords, curr_pos, curr_yaw: float
):
    """
    Convert local coordinates to global coordinates

    Args:
        local_coords (np.ndarray): coordinates in local frame (dx, dy, dyaw) or (dx, dy)
        curr_pos (np.ndarray): current position in global frame
        curr_yaw (float): current yaw in global frame
    Returns:
        np.ndarray: positions in global coordinates
        float: global yaw (only if input includes dyaw)
    """
    rotmat = yaw_rotmat(-curr_yaw)  # Inverse rotation matrix (negative yaw)
    if local_coords.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
        if isinstance(local_coords, torch.Tensor):
            global_pos = local_coords.matmul(rotmat) + curr_pos
            return global_pos, None
        else:
            global_pos = local_coords.dot(rotmat) + curr_pos
            return global_pos, None
    elif local_coords.shape[-1] == 3:
        rotmat = rotmat[:2, :2]  # Only rotate x,y coordinates
        local_xy = local_coords[..., :2]
        local_yaw = local_coords[..., 2]
        
        if isinstance(local_coords, torch.Tensor):
            global_pos_xy = local_xy.matmul(rotmat) + curr_pos[:2].unsqueeze(0)  # [N,2]
            # Add z-dimension back, broadcasting to match batch size
            global_pos = torch.cat([global_pos_xy, curr_pos[2].expand(len(local_coords), 1)], dim=-1)  # [N,3]
            global_yaw = curr_yaw + local_yaw
        else:
            global_pos_xy = local_xy.dot(rotmat) + curr_pos[:2]  # [N,2]
            # Add z-dimension back, broadcasting to match batch size
            global_pos = np.concatenate([global_pos_xy, np.full((len(local_coords), 1), curr_pos[2])], axis=-1)  # [N,3]
            global_yaw = curr_yaw + local_yaw
        return global_pos, global_yaw
    else:
        raise ValueError("Input coordinates must have shape [..., 2] or [..., 3]")


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
    load_data = defaultdict(list)
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
            load_data[str(item['trajectory_id'])].append(item)
    if logger is not None:
        logger.info(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data

def get_checkpoint_id(ckpt_path: str) -> Optional[int]:
    r"""Attempts to extract the ckpt_id from the filename of a checkpoint.
    Assumes structure of ckpt.ID.path .

    Args:
        ckpt_path: the path to the ckpt file

    Returns:
        returns an int if it is able to extract the ckpt_path else None
    """
    ckpt_path = os.path.basename(ckpt_path)
    nums: List[int] = [int(s) for s in ckpt_path.split(".") if s.isdigit()]
    if len(nums) > 0:
        return nums[-1]
    return None


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int,
    start_eval_epoch=-1, first_find_start_epoch=False
):
    r"""Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    new_model_paths = []
    for path in models_paths:
        if path.endswith(".pth"):
            new_model_paths.append(path)
    models_paths = new_model_paths
    if len(models_paths) == 0:
        print('No checkpoints found in folder: ', checkpoint_folder)
        return -1
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if start_eval_epoch != -1:
        if first_find_start_epoch:
            for idx, model_path in enumerate(models_paths):
                ckpt_len = len(model_path.split('/')[-1].split('.'))
                if ckpt_len == 3:
                    ckpt_num = model_path.split('/')[-1].split('.')[1]
                elif ckpt_len == 4:
                    # save as epoch, steps
                    ckpt_num = model_path.split('/')[-1].split('.')[1] + model_path.split('/')[-1].split('.')[2]
                # ckpt_file_ind = int(model_path.split('/')[-1].split('.')[1])
                ckpt_file_ind = int(ckpt_num)
                if ckpt_file_ind >= start_eval_epoch:
                    ind = idx
                    print(f'Find the start eval epoch file for {ckpt_file_ind}-th epoch.')
                    break
            previous_ckpt_ind = ind
            return (models_paths[ind], previous_ckpt_ind)
            
    if ind < len(models_paths):
        return models_paths[ind]
    return None

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
def is_slurm_job() -> bool:
    return SLURM_JOBID is not None

def is_slurm_batch_job() -> bool:
    r"""Heuristic to determine if a slurm job is a batch job or not. Batch jobs
    will have a job name that is not a shell unless the user specifically set the job
    name to that of a shell. Interactive jobs have a shell name as their job name.
    """
    return is_slurm_job() and os.environ.get("SLURM_JOB_NAME", None) not in (
        None,
        "bash",
        "zsh",
        "fish",
        "tcsh",
        "sh",
    )

def batch_obs(
    observations,
    device: Optional[torch.device] = None,
):
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))

def save_video(VIDEO_DIR, total_rgb_list, split, ep_id, checkpoint_index, spl):
    # 保存视频
    video_path = os.path.join(VIDEO_DIR, f"{split}_episode_{ep_id}_ckpt_{checkpoint_index}_spl_{spl}.mp4")
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        10, # fps
        (total_rgb_list[0].shape[1], total_rgb_list[0].shape[0])
    )
    for frame in total_rgb_list:
        video_writer.write(frame)
    video_writer.release()
    print(f"Save video to {video_path}")