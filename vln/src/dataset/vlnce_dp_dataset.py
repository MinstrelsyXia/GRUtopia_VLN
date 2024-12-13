'''
Author: w61
Date: 2024/11/05
Function: DataLoader for training
'''
import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb
import random
import time
from collections import defaultdict
import zlib
import msgpack_numpy
import matplotlib.pyplot as plt
import copy
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, ToPILImage
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from vln.src.models.utils.feature_extract import extract_image_features, extract_instruction_tokens

from vln.src.utils.utils import (
    to_local_coords,
    to_global_coords,
    normalize_data,get_delta,map_action_to_2d
)

def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

class VLNCE_DP_Dataset(IterableDataset):
    '''For vlnce diffusion policy'''
    def __init__(
        self,
        config,
        lmdb_features_dir,
        policy,
        device,
        dataset_data: dict,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "rgbd",
        goal_type: str = "instruction",
        lmdb_map_size=1e9,
        batch_size=1,
        bert_tokenizer=None,
        inflection_weight_coef=1.0,
        is_distributed=False,
        rank = 0,
        world_size = 1,
        lmdb_save_episode_id = False,
        use_stack = True
    ):
        """
        Main VLNCE-DP dataset class
        """
        self.config = config
        self.dp_config = config.MODEL.Diffusion_Policy
        
        self.device = device
        self.camera_name = self.config.IL.camera_name
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size

        self._preload = []
        self.batch_size = batch_size
        
        self.action_stats = {}
        self.action_stats['min'] = np.array(self.config.MODEL.Diffusion_Policy.action_stats.min)
        self.action_stats['max'] = np.array(self.config.MODEL.Diffusion_Policy.action_stats.max)

        if self.config.MODEL.use_iw:
            self.use_iw = True
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.use_iw = False
            self.inflec_weights = torch.tensor([1.0, 1.0])
        
        self.lmdb_save_episode_id = lmdb_save_episode_id
        self.use_stack = use_stack
        
        self.img_mod = self.config.MODEL.IMAGE_ENCODER.RGB.img_mod
        self.is_clip_long = (self.config.MODEL.TEXT_ENCODER.type == 'clip-long')
        self.policy = policy
        
        self.use_rnn = 'noRNN' not in self.config.MODEL.policy_name
        self.preload_size = batch_size * 10 if self.use_rnn else 8

        # preprocess images
        self.to_pil = ToPILImage()
        self.image_processor = _transform(n_px=224) # copy fron clip-long
        
        self.need_extract_instr_features = False if not self.config.MODEL.TEXT_ENCODER.update_text_encoder else True # has preprocessed the instruction
        
        if self.config.IL.analysis_time:
            start_time = time.time()
        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True, 
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]
            # Obtain all keys
            with lmdb_env.begin() as txn:
                cursor = txn.cursor()
                self.lmdb_keys = []
                while cursor.next():
                    self.lmdb_keys.append(cursor.key().decode())
        if self.config.IL.analysis_time:
            end_time = time.time()
            print(f"Time taken to load LMDB: {end_time - start_time:.2f} seconds")

        self.start = 0
        self.end = self.length
        self.world_size = world_size
        if is_distributed:
            per_rank = int(np.ceil(self.length / world_size))
            self.start = per_rank * rank
            if rank == world_size-1:
                self.end = min(self.start + per_rank, self.length) # the last rank maybe out of index
            else:
                self.end = self.start + per_rank
            self.length = per_rank
        
        self.bert_tokenizer = bert_tokenizer
    
        self.dataset_data = dataset_data

        self.waypoint_spacing = self.dp_config.waypoint_spacing
        self.min_dist_cat = self.dp_config.distance.min_dist_cat
        self.max_dist_cat = self.dp_config.distance.max_dist_cat
        self.distance_categories = list(
            range(self.min_dist_cat, self.max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = self.dp_config.negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = self.dp_config.len_traj_pred
        self.learn_angle = self.config.MODEL.learn_angle
        self.metric_waypoint_spacing = self.dp_config.metric_waypoint_spacing

        self.min_action_distance = self.dp_config.action.min_dist_cat
        self.max_action_distance = self.dp_config.action.max_dist_cat

        self.context_size = self.dp_config.context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        # self._load_index()
        # self._build_caches()
        
        if self.config.MODEL.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def _create_new_data(self, data, yaws, instruction, finish_status, fail_reason):
        """Helper function to create new data entry"""
        new_data = {
            'instruction': instruction,
            'progress': data['progress'],
            'globalgps': data['robot_info']['position'],
            'global_rotation': data['robot_info']['orientation'],
            'globalyaw': yaws,
        }

        # Handle RGB and depth features/data
        if 'rgb_features' in data:
            new_data['rgb_features'] = data['rgb_features']
            if self.config.MODEL.IMAGE_ENCODER.DEPTH.update_depth_encoder:
                new_data['depth'] = np.expand_dims(data['camera_info'][self.camera_name]['depth'], axis=-1)
            else:
                new_data['depth_features'] = data['depth_features']
        else:
            new_data['rgb'] = data['camera_info'][self.camera_name]['rgb']
            new_data['depth'] = np.expand_dims(data['camera_info'][self.camera_name]['depth'], axis=-1)
        
        return new_data

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            finish_status_list = []
            fail_reasons_list = []
                
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    key = self.lmdb_keys[self.load_ordering.pop()]
                    packed_data = txn.get(key.encode())
                    # try:
                    #     data_to_load = zlib.decompress(packed_data)
                    #     data_to_load = pickle.loads(data_to_load)                 
                    # except:
                    data_to_load = msgpack_numpy.unpackb(packed_data, raw=False)
                    data = data_to_load['episode_data']
                    finish_status = data_to_load['finish_status']
                    fail_reason = data_to_load['fail_reason']
                    if self.config.IL.Filter_failure.use:
                        if finish_status != 'success':
                            if 'rgb' in data['camera_info'][self.camera_name].keys():
                                if len(data['camera_info']) == 0 or len(data['camera_info'][self.camera_name]['rgb']) < self.config.IL.Filter_failure.min_rgb_nums:
                                    continue
                            else:
                                if len(data['camera_info']) == 0 or len(data['rgb_features']) < self.config.IL.Filter_failure.min_rgb_nums:
                                    continue
                    
                    # convert yaw from [-2pi,2pi] to [-pi, pi]
                    yaws = np.array(data['robot_info']['yaw']).copy()
                    for yaw_i, yaw in enumerate(data['robot_info']['yaw']):
                        yaw = yaw%(2*np.pi)
                        if yaw > np.pi:
                            yaw -= 2*np.pi
                        yaws[yaw_i] = yaw

                    if 'instr_features' in data and not self.config.MODEL.TEXT_ENCODER.update_text_encoder:
                        instructions = data['instr_features']
                        self.need_extract_instr_features = False
                    else:
                        instructions = [
                            self.dataset_data[key][ep_idx]['instruction']['instruction_text'][:self.config.MODEL.TEXT_ENCODER.max_length]
                            for ep_idx in range(len(self.dataset_data[key]))
                        ]
                        self.need_extract_instr_features = True

                    for instruction in instructions:
                        new_data = self._create_new_data(data, yaws, instruction, finish_status, fail_reason)
                        new_preload.append(new_data)
                        finish_status_list.append(finish_status)
                        fail_reasons_list.append(fail_reason)
                        lengths.append(len(new_data))

                    if self.need_extract_instr_features:
                        # compute stack images, positions, yaw, and relative actions, time_distance for each observations
                        new_preload = extract_instruction_tokens(new_preload, self.bert_tokenizer, is_clip_long=self.is_clip_long)
            
            # process the instruction
            # copy the instruction to each step
            if self.need_extract_instr_features:
                for i in range(len(new_preload)):
                    new_preload[i]['instruction'] = np.tile(np.array(new_preload[i]['instruction']), (len(new_preload[i]['progress']),1))
            else:
                for i in range(len(new_preload)):
                    new_preload[i]['instruction'] = np.expand_dims(new_preload[i]['instruction'], axis=0)
                    new_preload[i]['instruction'] = np.tile(new_preload[i]['instruction'], (len(new_preload[i]['progress']), 1, 1))

            if self.config.IL.analysis_time:
                start_time = time.time()
            
            for item_idx in range(len(new_preload)):
                item_obs = new_preload[item_idx]
                total_steps = len(item_obs["progress"])
                '''Type-2: Remove episodes having too long steps'''
                # if total_steps > 200:
                #     continue
                
                '''Type-1: Restrict the total_steps to maximum length to avoid over-cuda memory'''
                # total_steps = min(total_steps, 200)
                # for k,v in item_obs.items():
                #     item_obs[k] = item_obs[k][:total_steps]

                # add stop_progress
                item_obs["stop_progress"] = np.arange(total_steps) / total_steps
                
                for k,v in item_obs.items():
                    item_obs[k] = torch.from_numpy(np.array(item_obs[k]))

                if self.config.MODEL.learn_angle:
                    item_obs["actions"] = torch.zeros((total_steps, self.len_traj_pred, 3))
                    item_obs["prev_actions"] = torch.zeros((total_steps, self.config.MODEL.len_traj_act, 3))
                else:
                    item_obs["actions"] = item_obs["prev_actions"] = torch.zeros((total_steps, self.len_traj_pred, 2))
                    item_obs["prev_actions"] = torch.zeros((total_steps, self.config.MODEL.len_traj_act, 2))
                item_obs["step_distance"] = torch.zeros(total_steps)

                if self.config.MODEL.STEP_ENCODER.use:
                    item_obs["steps"] = torch.arange(min(total_steps, self.config.MODEL.STEP_ENCODER.max_steps))
                
                if "rgb_features" in item_obs.keys():
                    self.extract_img_features = False
                else:
                    self.extract_img_features = True
                
                # Stack images
                img_stack_nums = 1 if not self.use_stack else self.config.MODEL.IMAGE_ENCODER.img_stack_nums
                if self.extract_img_features:
                    # extract image features from raw images
                    # process RGB images
                    process_images = []
                    for image in item_obs["rgb"]:
                        image = image.permute(2,0,1) # H,W,C -> C,H,W
                        process_images.append(self.image_processor(self.to_pil(image)))
                    item_obs["rgb"] = torch.stack(process_images) # [T, 3, 224, 224]
                    
                    img_shape = item_obs["rgb"][0].shape
                else:
                    img_shape = item_obs["rgb_features"][0].shape
                
                if "depth" in item_obs.keys():
                    depth_shape = item_obs["depth"][0].shape
                    if len(depth_shape) == 2:
                        # [256, 256] -> [256, 256, 1]
                        item_obs["depth"] = torch.unsqueeze(item_obs["depth"], dim=-1)

                if self.use_stack:
                    item_obs["stack_rgb"] = torch.zeros((total_steps, img_stack_nums, *img_shape))
                    item_obs["stack_depth"] = torch.zeros((total_steps, img_stack_nums, *depth_shape))
                    
                if self.config.MODEL.IMU_ENCODER.use:
                    item_obs["imu"] = torch.zeros((total_steps, self.config.MODEL.IMU_ENCODER.input_size))
                
                start_pos = item_obs["globalgps"][0][[0, 1]]
                start_yaw = item_obs["globalyaw"][0]
                for step_idx in range(total_steps):
                    # compute imu
                    if self.config.MODEL.IMU_ENCODER.use:
                        current_pos = item_obs["globalgps"][step_idx][[0,1]]
                        item_obs["imu"][step_idx][:2] = to_local_coords(current_pos, start_pos, start_yaw)
                        if self.config.MODEL.IMU_ENCODER.input_size == 3:
                            item_obs["imu"][step_idx][2] = item_obs["globalyaw"][step_idx] - start_yaw
                    
                    # stack multiple images and depths
                    if self.use_stack:
                        if self.extract_img_features:
                            rgb_key_name = "rgb"
                            depth_key_name = "depth"
                        else:
                            rgb_key_name = "rgb_features"
                            depth_key_name = "depth" if self.config.MODEL.IMAGE_ENCODER.DEPTH.update_depth_encoder else "depth_features" 
                            
                        if step_idx == 0:
                            item_obs["stack_rgb"][step_idx][0] = item_obs[rgb_key_name][step_idx]
                            item_obs["stack_depth"][step_idx][0] = item_obs[depth_key_name][step_idx]
                        else:
                            prev_step_idx = min(img_stack_nums, step_idx+1)     
                            # use torch.flip to make the latest image in the first token
                            flip_images = torch.flip(item_obs[rgb_key_name][step_idx+1-prev_step_idx: step_idx+1], dims=[0])
                            item_obs["stack_rgb"][step_idx][:prev_step_idx] = flip_images
                            item_obs["stack_depth"][step_idx][:prev_step_idx] = torch.flip(item_obs[depth_key_name][step_idx+1-prev_step_idx: step_idx+1], dims=[0])
                
                for step_idx in range(total_steps):
                    # compute actions
                    actions = self._compute_actions(item_obs["globalgps"], 
                                                    item_obs["globalyaw"],
                                                    step_idx,
                                                    fill_mode='constant',
                                                    vis=self.config.test_verbose, # !!! DEBUG
                                                    save_dir=self.config.LOG_DIR)
                    prev_actions = self._compute_actions(torch.flip(item_obs["globalgps"], dims=[0]),
                                                         torch.flip(item_obs["globalyaw"], dims=[0]),
                                                         total_steps-step_idx-1,
                                                         fill_mode='constant')[:self.config.MODEL.len_traj_act]
                    
                    action_deltas = get_delta(actions)
                    
                    if self.learn_angle:                         
                        item_obs["actions"][step_idx] = normalize_data(action_deltas, self.action_stats) # convert actions to [-1, 1]
                    else:
                        item_obs["actions"][step_idx] = map_action_to_2d(action_deltas) # convert to (forward, rotation) dimension
                    
                    # compute temporal step distance
                    distance = (total_steps - step_idx - 1) // self.waypoint_spacing
                    if self.config.MODEL.DISTANCE_PREDICTOR.normalize:
                        item_obs["step_distance"][step_idx] = distance / total_steps
                    else:
                        item_obs["step_distance"][step_idx] = distance
                    
                    # update prev action
                    if step_idx > 0:
                        prev_action_deltas = get_delta(prev_actions)
                        if self.learn_angle:
                            item_obs["prev_actions"][step_idx] = normalize_data(prev_action_deltas, self.action_stats)
                        else:
                            item_obs["prev_actions"][step_idx] = map_action_to_2d(prev_action_deltas)
     
                # add additional information
                # if self.lmdb_save_episode_id:
                #     new_preload[item_idx].append(episode_ids[item_idx])
                #     new_preload[item_idx].append(gt_actions[item_idx])

            if not self.use_rnn:
                # do not use rnn in model, so split the data into multiple batches
                split_new_preload = []
                for i in range(len(new_preload)):
                    total_steps = len(new_preload[i]['progress'])
                    keys = new_preload[i].keys()
                    for j in range(total_steps):
                        new_data = {}
                        for k in keys:
                            new_data[k] = new_preload[i][k][j]
                        new_data['use_rnn'] = self.use_rnn
                        split_new_preload.append(new_data)
                new_preload = split_new_preload
                lengths = [1] * len(new_preload)
                
            if self.config.IL.analysis_time:
                end_time = time.time()
                print(f"Time taken to process data in dataLoader: {end_time - start_time:.2f} seconds")

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop() # pop one item each time
    
    def __next__(self):
        obs = self._load_next()

        return (obs)

    def _compute_actions(self, globalgps, yaws, curr_time, fill_mode, vis=False, save_dir=None):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = yaws[start_index:end_index:self.waypoint_spacing]
        original_globalgps = copy.copy(globalgps)
        globalgps = globalgps[:, [0, 1]]
        positions = globalgps[start_index:end_index:self.waypoint_spacing]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        
        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            if fill_mode == 'constant':
                yaw = torch.cat([yaw, yaw[-1].repeat(const_len)])
                positions = torch.cat([positions, positions[-1].unsqueeze(0).repeat(const_len, 1)], dim=0)
            elif fill_mode == 'zero':
                yaw = torch.cat([yaw, torch.zeros(const_len)])
                positions = torch.cat([positions, torch.zeros((const_len, 2))], dim=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        # if self.learn_angle:
        # note that relative actions start from the next point
        delta_yaw = yaw[1:] - yaw[0]
        # Normalize the angles to be within [-π, π] (get the small angle between two yaws)
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))
        
        actions = torch.cat([waypoints[1:], delta_yaw[:, None]], dim=-1)
        # else:
            # actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing

        if self.learn_angle:
            assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        if vis:
            self.visualize_waypoints(waypoints, delta_yaw, curr_time, save_dir)
            # Expand waypoints from [9,2] to [9,3] by adding z dimension from original_globalgps
            cat_waypoints = torch.cat([waypoints[1:], delta_yaw.unsqueeze(-1)], dim=-1)
            return_gps, return_yaw = to_global_coords(cat_waypoints, original_globalgps[0], yaw[0])

        return actions

    def visualize_waypoints(self, waypoints, delta_yaw, step_idx, save_dir):
        """
        Visualize waypoints and their orientations
        Args:
            waypoints: shape (len_traj_pred + 1, 2) - includes start point
            delta_yaw: shape (len_traj_pred,) - relative angles from start
            step_idx: current step index
            save_dir: directory to save visualization
        """
        plt.clf()
        plt.figure(figsize=(5, 5))
        
        # Plot waypoints
        plt.scatter(waypoints[:, 0], waypoints[:, 1], 
                label='waypoints', color='red', alpha=0.5)
        
        # Plot start point in different color
        plt.scatter(waypoints[0, 0], waypoints[0, 1], 
                color='green', alpha=1.0, label='start')
        
        # Add arrows for each waypoint (except start point)
        arrow_length = 0.2
        start_yaw = 0  # Reference angle at start point
        for i in range(1, len(waypoints)):
            # Current absolute angle = start_yaw + accumulated delta
            current_yaw = start_yaw + delta_yaw[i-1]
            
            # Calculate arrow direction
            dx = arrow_length * np.cos(current_yaw)
            dy = arrow_length * np.sin(current_yaw)
            
            # Draw arrow
            plt.arrow(waypoints[i, 0], 
                    waypoints[i, 1], 
                    dx, dy,
                    head_width=0.05,
                    head_length=0.1,
                    fc='red',
                    ec='red',
                    alpha=0.5)
            
            # Add waypoint index
            plt.text(waypoints[i, 0], waypoints[i, 1],
                    f'{i}', fontsize=9, color='red', ha='right')
        
        plt.title(f'Waypoints and Orientations at Step {step_idx}')
        plt.legend()
        plt.grid(True)
        
        # Equal aspect ratio to prevent distortion
        plt.axis('equal')
        
        # Save figure
        save_path = os.path.join(save_dir, f'waypoints_step_{step_idx}.png')
        plt.savefig(save_path)
        print(f"Dataset: Saved waypoints to {save_path}")
        plt.close()


    def __len__(self) -> int:
        return self.length * 200
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self

def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
        (Optional) episode_ids,
        (Optional) gt_actions
    )
    """

    def _pad_helper(t, max_len, fill_val=0, return_masks=False):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            if return_masks:
                mask = torch.ones(max_len, dtype=torch.int)
                return t, mask
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )

        # Create the mask: 1 for original tokens, 0 for padding
        if return_masks:
            mask = torch.zeros(max_len, dtype=torch.int)
            mask[:t.size(0)] = 1  # Original tokens
            mask[t.size(0):] = 0   # Padded tokens

            return torch.cat([t, pad], dim=0), mask
        return torch.cat([t, pad], dim=0)

    # transposed = list(zip(*batch))

    # observations_batch = list(transposed[0])
    observations_batch = batch

    B = len(observations_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0].keys():
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    if 'use_rnn' in observations_batch:
        use_rnn = observations_batch['use_rnn'][0]
        del observations_batch['use_rnn']
    else:
        use_rnn = True
    if use_rnn:
        max_traj_len = max(ele.size(0) for ele in observations_batch['progress'])
        not_done_masks_batch = torch.ones(B, max_traj_len, dtype=torch.uint8)
        for bid in range(B):
            for sensor in observations_batch:
                if sensor == 'progress':
                    observations_batch[sensor][bid] = _pad_helper(
                    observations_batch[sensor][bid], max_traj_len, fill_val=1.0
                )
                else:
                    # if sensor == 'instruction':
                    #     observations_batch[sensor][bid] = observations_batch[sensor][bid].unsqueeze(0)
                    observations_batch[sensor][bid] = _pad_helper(
                        observations_batch[sensor][bid], max_traj_len, fill_val=0.0
                    )

        for sensor in observations_batch:
            observations_batch[sensor] = torch.stack(
                observations_batch[sensor], dim=1
            )
            observations_batch[sensor] = observations_batch[sensor].view(
                -1, *observations_batch[sensor].size()[2:]
            )

        observations_batch = ObservationsDict(observations_batch)
        # length 330: longest_episode_length*batch_size
        return (
            observations_batch,
            observations_batch['prev_actions'],
            not_done_masks_batch.view(-1, 1),
        )
    else:
        for sensor in observations_batch:
            if sensor == 'use_rnn':
                continue
            observations_batch[sensor] = torch.stack(
                observations_batch[sensor], dim=0
            )
        observations_batch = ObservationsDict(observations_batch)
        return (
            observations_batch,
            observations_batch['prev_actions'],
            None
        )