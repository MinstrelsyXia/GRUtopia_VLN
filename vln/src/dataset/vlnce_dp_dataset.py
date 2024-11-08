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
from collections import defaultdict
import zlib

import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.functional as TF

from vln.src.models.utils.feature_extract import extract_image_features, extract_instruction_tokens

from vln.src.utils.utils import (
    to_local_coords,
    normalize_data,get_delta,map_action_to_2d
)

def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]

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
        img_encoder=None,
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
        
        self.camera_name = self.config.IL.camera_name
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size
        
        self.action_stats = {}
        self.action_stats['min'] = np.array(self.config.MODEL.Diffusion_Policy.action_stats.min.cpu())
        self.action_stats['max'] = np.array(self.config.MODEL.Diffusion_Policy.action_stats.max.cpu())

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
                for key, _ in cursor:
                    self.lmdb_keys.append(key.decode())
        
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
        self.img_encoder = img_encoder

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
                    data_to_load = zlib.decompress(packed_data)
                    data_to_load = pickle.loads(data_to_load)                 
                    data = data_to_load['episode_data']
                    finish_status = data_to_load['finish_status']
                    fail_reason = data_to_load['fail_reason']
                    if self.config.IL.Filter_failure.use:
                        if finish_status != 'success':
                            if len(data['camera_info']) == 0 or len(data['camera_info'][self.camera_name]['rgb']) < self.config.IL.Filter_failure.min_rgb_nums:
                                continue
                        
                    instr = self.dataset_data[key]['instruction']['instruction_text'][:self.config.MODEL.TEXT_ENCODER.max_length]
                    new_data = {
                        'instruction': instr,
                        'progress': data['progress'],
                        'globalgps': data['robot_info']['position'],
                        'global_rotation': data['robot_info']['orientation'],
                        'globalyaw': data['robot_info']['yaw'],
                        'rgb': data['camera_info'][self.camera_name]['rgb'],
                        'depth': data['camera_info'][self.camera_name]['depth']
                    }
                    new_preload.append(new_data)
                    finish_status_list.append(finish_status)
                    fail_reasons_list.append(fail_reason)
                    lengths.append(len(new_preload[-1]))

            # compute stack images, positions, yaw, and relative actions, time_distance for each observations
            new_preload = extract_instruction_tokens(new_preload, self.bert_tokenizer, is_clip_long=self.is_clip_long)
            
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
                
                for k,v in item_obs.items():
                    item_obs[k] = torch.from_numpy(np.array(item_obs[k])).to(self.device)

                if self.config.MODEL.learn_angle:
                    item_obs["actions"] = np.zeros((total_steps, self.len_traj_pred, 3))
                    item_obs["prev_actions"] = np.zeros((total_steps, self.config.MODEL.len_traj_act, 3))
                else:
                    item_obs["actions"] = item_obs["prev_actions"] = np.zeros((total_steps, self.len_traj_pred, 2))
                    item_obs["prev_actions"] = np.zeros((total_steps, self.config.MODEL.len_traj_act, 2))
                item_obs["step_distance"] = np.zeros(total_steps)

                if self.config.MODEL.STEP_ENCODER.use:
                    item_obs["steps"] = np.arange(min(total_steps, self.config.MODEL.STEP_ENCODER.max_steps))
                
                if "rgb_features" in item_obs.keys():
                    self.extract_img_features = False
                else:
                    self.extract_img_features = True
                
                # Stack images
                img_stack_nums = 1 if not self.use_stack else self.config.MODEL.IMAGE_ENCODER.img_stack_nums
                if self.extract_img_features:
                    # extract image features from raw images
                    img_shape = item_obs["rgb"][0].shape
                    depth_shape = item_obs["depth"][0].shape
                    if len(depth_shape) == 2:
                        # [256, 256] -> [256, 256, 1]
                        item_obs["depth"] = np.expand_dims(item_obs["depth"], axis=-1)
                        # TODO: change 256 to 224?
                    item_obs = extract_image_features(self.policy, item_obs, 
                                                      img_mod=self.img_mod, len_traj_act=self.config.MODEL.len_traj_act,
                                                      world_size=self.world_size,
                                                      depth_encoder_type=self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck,
                                                      proj=self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj)
                    

                img_shape = item_obs["rgb_features"][0].shape
                if self.img_mod == 'cls':
                    item_obs["stack_rgb"] = np.zeros((total_steps, img_stack_nums, img_shape[-1]))
                elif self.img_mod == 'multi_patches_avg_pooling':
                    img_patch_num = item_obs["rgb_features"][0].shape[0]
                    item_obs["stack_rgb"] = np.zeros((total_steps, img_stack_nums, img_patch_num, img_shape[-1]))
                if self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'TAC':
                    item_obs["stack_depth"] = np.zeros((total_steps, img_stack_nums, img_shape[-1]))
                elif self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'resnet':
                    depth_shape = item_obs["depth_features"][0].shape
                    item_obs["stack_depth"] = np.zeros((total_steps, img_stack_nums, depth_shape[0], depth_shape[1], depth_shape[2]))
                    
                if self.config.MODEL.IMU_ENCODER.use:
                    item_obs["imu"] = np.zeros((total_steps, 2))
                
                start_pos = item_obs["globalgps"][0][[0, 1]]
                for step_idx in range(total_steps):
                    # compute imu
                    if self.config.MODEL.IMU_ENCODER.use:
                        current_pos = item_obs["globalgps"][step_idx][[0,1]]
                        item_obs["imu"][step_idx] = current_pos - start_pos
                    
                    # stack multiple images and depths
                    if self.extract_img_features and self.img_encoder is not None:
                        # TODO: adaptive to long-clip and multiple RGB patches
                        item_obs["rgb_process"][step_idx] = self.img_encoder.process_image(item_obs["rgb"][step_idx])
                        if self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'TAC':
                            item_obs["depth_process"][step_idx] = self.img_encoder.process_depth(item_obs["depth"][step_idx])
                        
                        if self.use_stack:
                            if step_idx == 0:
                                item_obs["stack_rgb"][step_idx][0] = item_obs["rgb_process"][0]
                                if self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'TAC':
                                    item_obs["stack_depth"][step_idx][0] = item_obs["depth_process"][0]
                                elif self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'resnet':
                                    item_obs["stack_depth"][step_idx][0] = item_obs["depth"][0]
                            else:
                                prev_step_idx = min(img_stack_nums, step_idx+1)
                                item_obs["stack_rgb"][step_idx][:prev_step_idx] = item_obs["rgb_process"][step_idx+1-prev_step_idx: step_idx+1]
                                if self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'TAC':
                                    item_obs["stack_depth"][step_idx][:prev_step_idx] = item_obs["depth_process"][step_idx+1-prev_step_idx: step_idx+1]
                                elif self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'resnet':
                                    item_obs["stack_depth"][step_idx][:prev_step_idx] = item_obs["depth"][step_idx+1-prev_step_idx: step_idx+1]
                        else:
                            item_obs["stack_rgb"][step_idx][0] = item_obs["rgb_process"][step_idx]
                    else:
                        if step_idx == 0:
                            if self.config.MODEL.IMAGE_ENCODER.RGB.img_mod == 'cls' and item_obs["rgb_features"][0].shape[0] == self.config.MODEL.IMAGE_ENCODER.RGB.multi_patches_num:
                                # sample data use multi patches, but only use the first cls token
                                item_obs["stack_rgb"][step_idx][0] = item_obs["rgb_features"][0][0]
                            else:
                                item_obs["stack_rgb"][step_idx][0] = item_obs["rgb_features"][0]
                            item_obs["stack_depth"][step_idx][0] = item_obs["depth_features"][0]
                        else:
                            prev_step_idx = min(img_stack_nums, step_idx+1)
                            # use np.flip to make the latest image in the first token
                            flip_images = np.flip(item_obs["rgb_features"][step_idx+1-prev_step_idx: step_idx+1], axis=0)
                            if self.config.MODEL.IMAGE_ENCODER.RGB.img_mod == 'cls' and item_obs["rgb_features"][0].shape[0] == self.config.MODEL.IMAGE_ENCODER.RGB.multi_patches_num:
                                flip_images = flip_images[:,0]
                            item_obs["stack_rgb"][step_idx][:prev_step_idx] = flip_images
                            item_obs["stack_depth"][step_idx][:prev_step_idx] = np.flip(item_obs["depth_features"][step_idx+1-prev_step_idx: step_idx+1], axis=0)
                
                if self.extract_img_features or "rgb" in item_obs.keys():
                    del item_obs["rgb"]
                    del item_obs["depth"]
                    if "rgb_process" in item_obs.keys():
                        del item_obs["rgb_process"]
                        del item_obs["depth_process"]
                
                for step_idx in range(total_steps):
                    # compute actions
                    actions = self._compute_actions(item_obs["globalgps"], 
                                                    item_obs["globalyaw"],
                                                    step_idx,
                                                    fill_mode='constant')
                    prev_actions = self._compute_actions(np.flip(item_obs["globalgps"], axis=0),
                                                         np.flip(item_obs["globalyaw"], axis=0),
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
                if self.lmdb_save_episode_id:
                    new_preload[item_idx].append(episode_ids[item_idx])
                    new_preload[item_idx].append(gt_actions[item_idx])
                    
            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop() # pop one item each time
    
    def __next__(self):
        if self.lmdb_save_episode_id:
            obs, prev_actions, oracle_actions, episode_ids, gt_actions = self._load_next()
        else:
            obs, prev_actions, oracle_actions = self._load_next()
        
        prev_actions = obs['prev_actions']

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
            episode_ids,
            gt_actions
        )

    def _compute_actions(self, globalgps, yaws, curr_time, fill_mode):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = yaws[start_index:end_index:self.waypoint_spacing]
        globalgps = globalgps[:, [0, 2]]
        positions = globalgps[start_index:end_index:self.waypoint_spacing]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            if fill_mode == 'constant':
                yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
                positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)
            elif fill_mode == 'zero':
                yaw = np.concatenate([yaw, np.zeros(const_len)])
                positions = np.concatenate([positions, np.zeros((const_len, 2))], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        # if self.learn_angle:
        # note that relative actions start from the next point
        delta_yaw = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], delta_yaw[:, None]], axis=-1)
        # else:
            # actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing

        if self.learn_angle:
            assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"
        return actions

    def __len__(self) -> int:
        return len(self.index_to_data)
    
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

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])

    
    B = len(prev_actions_batch)

    if len(transposed) == 6:
        episode_ids_batch = list(transposed[4])
        gt_actions_batch = list(transposed[5])
    else:
        episode_ids_batch = None
        gt_actions_batch = None

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    not_done_masks_batch = torch.ones(B, max_traj_len, dtype=torch.uint8)
    for bid in range(B):
        for sensor in observations_batch:
            if sensor == 'progress':
                observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )
            else:
                observations_batch[sensor][bid] = _pad_helper(
                    observations_batch[sensor][bid], max_traj_len, fill_val=0.0
                )

        # pad_outputs = _pad_helper(
        #     prev_actions_batch[bid], max_traj_len, return_masks=True
        # )
        # prev_actions_batch[bid] = pad_outputs[0]
        # not_done_masks_batch[bid] = pad_outputs[1]

        prev_actions_batch[bid], not_done_masks_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len, return_masks=True
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid][:max_traj_len], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid][:max_traj_len], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    # not_done_masks = torch.ones_like(
    #     corrected_actions_batch, dtype=torch.uint8
    # )
    # not_done_masks[0] = 0
    # not_done_masks_batch = torch.stack(not_done_masks_batch, dim=1)

    observations_batch = ObservationsDict(observations_batch)
    # length 330: longest_episode_length*batch_size
    return (
        observations_batch,
        # prev_actions_batch.view(-1, 1),
        observations_batch['prev_actions'],
        not_done_masks_batch.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
        episode_ids_batch,
        gt_actions_batch
    )