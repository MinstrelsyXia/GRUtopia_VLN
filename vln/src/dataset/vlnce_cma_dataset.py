'''
Author: w61
Date: 2024/12/23
Function: DataLoader for training CMA policy. Modified from vlnce-habitat.
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

def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class CMADataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        config,
        lmdb_features_dir,
        use_iw,
        dataset_data,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
        is_distributed=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.config = config
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.dataset_data = dataset_data
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])
        
        self.camera_name = self.config.IL.camera_name

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(float(self.lmdb_map_size)),
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

    def _create_new_data(self, data, instruction, finish_status, fail_reason):
        """Helper function to create new data entry"""
        new_data = {
            'instruction': instruction,
            'progress': data['progress'],
            'globalgps': data['robot_info']['position'],
            'global_rotation': data['robot_info']['orientation'],
            'gt_actions': data['action'],
            'prev_actions': np.concatenate([np.array([0]), data['action'][:-1]]),
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
            empty_data_nums = 0
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(float(self.lmdb_map_size)),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break
                        
                    key = self.lmdb_keys[self.load_ordering.pop()]
                    packed_data = txn.get(key.encode())
                    
                    data_to_load = msgpack_numpy.unpackb(packed_data, raw=False)
                    data = data_to_load['episode_data']
                    finish_status = data_to_load['finish_status']
                    fail_reason = data_to_load['fail_reason']
                    # Filter the empty data 
                    if len(data['camera_info']) == 0:
                        empty_data_nums += 1
                        continue
                    if self.config.IL.Filter_failure.use:
                        if finish_status != 'success':
                            if 'rgb' in data['camera_info'][self.camera_name].keys():
                                if len(data['camera_info']) == 0 or len(data['camera_info'][self.camera_name]['rgb']) < self.config.IL.Filter_failure.min_rgb_nums:
                                    continue
                            else:
                                if len(data['camera_info']) == 0 or len(data['rgb_features']) < self.config.IL.Filter_failure.min_rgb_nums:
                                    continue
                        if finish_status == 'stuck':
                            drop_last_frame_nums = 25
                            data['camera_info'][self.camera_name]['rgb'] = data['camera_info'][self.camera_name]['rgb'][:-drop_last_frame_nums]
                            data['camera_info'][self.camera_name]['depth'] = data['camera_info'][self.camera_name]['depth'][:-drop_last_frame_nums]
                            data['robot_info']['yaw'] = data['robot_info']['yaw'][:-drop_last_frame_nums]
                            data['robot_info']['position'] = data['robot_info']['position'][:-drop_last_frame_nums]
                            data['robot_info']['orientation'] = data['robot_info']['orientation'][:-drop_last_frame_nums]
                            data['progress'] = data['progress'][:-drop_last_frame_nums]
                            data['step'] = data['step'][:-drop_last_frame_nums]
                            if 'rgb_features' in data.keys():
                                data['rgb_features'] = data['rgb_features'][:-drop_last_frame_nums]
                            if 'depth_features' in data.keys():
                                data['depth_features'] = data['depth_features'][:-drop_last_frame_nums]

                        instructions = [
                            self.dataset_data[key][ep_idx]['instruction']['instruction_tokens']
                            for ep_idx in range(len(self.dataset_data[key]))
                        ]
                        for instruction in instructions:
                            new_data = self._create_new_data(data, instruction, finish_status, fail_reason)
                            new_preload.append(new_data)
                            finish_status_list.append(finish_status)
                            fail_reasons_list.append(fail_reason)
                            lengths.append(len(new_data))
                
                if empty_data_nums > 0:
                    print(f"empty data nums: {empty_data_nums}")
                    
                # process the instruction
                # copy the instruction to each step
                for i in range(len(new_preload)):
                    new_preload[i]['instruction'] = np.tile(np.array(new_preload[i]['instruction']), (len(new_preload[i]['progress']),1))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs= self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))
        
        prev_actions = torch.from_numpy(np.copy(obs['prev_actions']))
        oracle_actions = torch.from_numpy(np.copy(obs['gt_actions']))

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
        )

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


    def __len__(self) -> int:
        return self.length * 3 # each trajectory corresponds to 3 instructions
    
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
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

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
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )