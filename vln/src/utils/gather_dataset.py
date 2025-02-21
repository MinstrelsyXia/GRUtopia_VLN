import os,sys
import json
import numpy as np
import gzip
from collections import defaultdict
import copy

from vln.parser import process_args
# from vln.src.dataset.data_utils import load_data
from vln.src.utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations

def load_sixth_floor_data(args, split, dataset_root_dir = None):
    ''' Load data based on VLN-CE
    '''
    total_scans = []
    load_data = []
    with gzip.open(dataset_root_dir, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for item in data["episodes"]:
            item["original_start_position"] = copy.copy(item["start_position"])
            item["original_start_rotation"] = copy.copy(item["start_rotation"])
            item["start_position"] = [item["original_start_position"][0], item["original_start_position"][1], item["original_start_position"][2]] # unchanged
            init_orientation = item['original_start_rotation']
            init_orientation = quat_to_euler_angles(init_orientation)
            orientation = [0, 0, init_orientation[2]]
            init_orientation = euler_angles_to_quat(orientation)
            item['start_rotation'] = init_orientation
            item["scan"] = item["scene_id"]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], path[1], path[2]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data.append(item)
            total_scans.append(item["scan"])
    print(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data, list(set(total_scans))

def load_data(args, split, dataset_root_dir=None, is_fsa_dataset=False, correct_fsa_rotation=False):
    ''' Load data based on VLN-CE
    '''
    dataset_root_dir = args.datasets.base_data_dir if dataset_root_dir is None else dataset_root_dir
    total_scans = []
    load_data = []
    if is_fsa_dataset:
        # for MLANet
        dataset_file = os.path.join(dataset_root_dir, f"{split}", f"{split}_sub.json.gz")
        ori_dataset_file = os.path.join("data/datasets/R2R_VLNCE_v1-3_preprocessed", f"{split}", f"{split}.json.gz") # MLANet sub数据集里的start_rotation和v1-3_processed里的不一样
        with gzip.open(ori_dataset_file, 'rt', encoding='utf-8') as f:
            ori_data = json.load(f)
            ori_data = ori_data["episodes"]
    else:
        dataset_file = os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz")
    with gzip.open(dataset_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for idx, item in enumerate(data["episodes"]):
            item["original_start_position"] = copy.copy(item["start_position"])
            if is_fsa_dataset and correct_fsa_rotation:
                item["original_start_rotation"] = copy.copy(ori_data[idx]["start_rotation"])
            else:
                item["original_start_rotation"] = copy.copy(item["start_rotation"])
            item["start_position"] = [item["original_start_position"][0], -item["original_start_position"][2], item["original_start_position"][1]]
            item["start_rotation"] = [-item["original_start_rotation"][3], item["original_start_rotation"][0], item["original_start_rotation"][2], -item["original_start_rotation"][1]] # [x,y,z,-w] => [w,x,y,z]
            item["start_rotation"] = transform_rotation_z_90degrees(item["start_rotation"])
            item["scan"] = item["scene_id"].split("/")[1]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], -path[2], path[1]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data.append(item)
            total_scans.append(item["scan"])

    print(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data, list(set(total_scans))

def transform_rotation_z_90degrees(rotation):
    ''' 沿着z轴旋转90度
    '''
    z_rot_90 = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]  # 90 degrees = pi/2 radians
    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = z_rot_90
    revised_rotation = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ]
    return revised_rotation

def get_yaw_from_rotation(rotation):
    """从四元数计算yaw角(绕z轴的旋转)
    Args:
        rotation: 四元数 [w, x, y, z]
    Returns:
        yaw: 弧度制的偏航角
    """
    w, x, y, z = rotation
    # 计算yaw(绕z轴旋转)的弧度值
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return yaw

# 如果需要转换为角度制:
def get_yaw_degree(rotation):
    """从四元数计算yaw角并转换为角度制
    """
    yaw_rad = get_yaw_from_rotation(rotation)
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg

class datasetGather:
    def __init__(self, args, dataset_root_dir=None, is_fsa_dataset=False, correct_fsa_rotation=False):
        self.args = args
        # self.splits = ['train', 'val_seen', 'val_unseen']
        self.splits = ['train', 'val_seen']
        # self.splits = ['envdrop']
        self.data = {split: [] for split in self.splits}
        self.scan = {}
        if args.mode == 'sixth_floor':
            for split in self.splits:
                dataset_path = os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz")
                self.data[split], self.scan[split] = load_sixth_floor_data(args, split, dataset_root_dir=dataset_path)
        else:
            for split in self.splits:
                self.data[split], self.scan[split] = load_data(self.args, split, dataset_root_dir=dataset_root_dir, is_fsa_dataset=is_fsa_dataset, correct_fsa_rotation=correct_fsa_rotation)

    def gatherSameScanData(self, save_gather_data=True, save_dir='gather_data/', fix_rotation=False):
        scan2data = {split: {} for split in self.splits}
        for split in self.splits:
            for data in self.data[split]:
                scan = data['scan']
                if scan not in scan2data[split]:
                    scan2data[split][scan] = []
                if fix_rotation:
                    # no need. already turn in load_data.
                    # data['start_rotation'] = transform_rotation_z_90degrees(data['start_rotation'])
                    data['start_rotation'] = data['start_rotation']
                scan2data[split][scan].append(data)
        
        if save_gather_data:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            for split in self.splits:
                save_path = os.path.join(save_dir, f'{split}_gather_data.json')
                with open(save_path, 'w') as f:
                    json.dump(scan2data[split], f, indent=2)
                print(f'Saved gathered data for {split} to {save_path}')
            
            with open(os.path.join(save_dir, 'env_scan.json'), 'w') as f:
                json.dump(self.scan, f, indent=2)
            print(f'Saved scan data to {os.path.join(save_dir, "env_scan.json")}')
        
        return scan2data

def read_gather_data(gather_data_path):
    with open(gather_data_path, 'r') as f:
        gather_data = json.load(f)
    return gather_data


def gather_eval_data(ori_dataset, sample_dataset_file, split, save_dir='gather_data/'):
    with open(sample_dataset_file, 'r') as f:
        dataset = json.load(f)
    
    scan_data = defaultdict(list)
    if 'sixth_floor' in dataset:
        dataset = dataset['sixth_floor']
    for item in dataset:
        scan = item['scan']
        trajectory_id = item['trajectory_id']
        for ori_data in ori_dataset:
            if ori_data['trajectory_id'] == trajectory_id:
                scan_data[scan].append(ori_data)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'{split}_PReval_gather_data.json')
    with open(save_path, 'w') as f:
        json.dump(scan_data, f, indent=2)
    print(f'Saved eval data for {split} to {save_path}')
    
def load_json_gz(json_gz_path):
    with gzip.open(json_gz_path, 'rb') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    '''1. Gather standard dataset'''
    # dataset_root_dir = "data/datasets/revised/processed_corrected"
    # dataset_root_dir = "data/datasets/R2R_VLNCE_v1-3_preprocessed"
    # dataset_root_dir = "data/datasets/R2R_VLNCE_FSASub"
    # dataset_root_dir = "data/datasets/R2R_VLNCE_v1-3_corrected"
    dataset_root_dir = '/ssd/wangliuyi/code/w61_grutopia_main/data/datasets/sixth_floor/sixth_floor/'

    args, _ = process_args()
    args.mode = 'sixth_floor'
    # dataset_gather = datasetGather(args, dataset_root_dir=dataset_root_dir, is_fsa_dataset=True, correct_fsa_rotation=False)
    dataset_gather = datasetGather(args, dataset_root_dir=dataset_root_dir, is_fsa_dataset=False, correct_fsa_rotation=False)
    # scan2data = dataset_gather.gatherSameScanData(save_gather_data=True, save_dir='gather_data/', fix_rotation=False)
    
    # read_gather_data('/ssd/wangliuyi/code/w61_grutopia/data/datasets/R2R_VLNCE_FSASub/val_seen/val_seen_sub.json.gz')

    '''2. Gather eval data'''
    # val_seen_sample_dataset_file = "data/sample_episodes/20241115_sample_episodes_val_seen/analysis/success_episode_data_val_seen.json"
    # val_unseen_sample_dataset_file = "data/sample_episodes/20241115_sample_episodes_val_unseen/analysis/success_episode_data_val_unseen.json"
    
    # gather_eval_data(dataset_gather.data['val_unseen'], val_unseen_sample_dataset_file, 'val_unseen')
    # gather_eval_data(dataset_gather.data['val_seen'], val_seen_sample_dataset_file, 'val_seen')

    # train_sample_dataset_file = 'path_generation/path_generaton/output/gather_data/sixth_floor_train_gather_data.json'
    # val_seen_dataset_file = 'path_generation/path_generaton/output/gather_data/sixth_floor_val_seen_gather_data.json'
    sample_dataset_file = 'path_generation/path_generaton/output/gather_data/sixth_floor_gather_data.json'
    gather_eval_data(dataset_gather.data['train'], sample_dataset_file, 'train')
    gather_eval_data(dataset_gather.data['val_seen'], sample_dataset_file, 'val_seen')
    '''3. Check the dataset'''
    # mlanet_data = load_json_gz('/ssd/wangliuyi/code/w61_grutopia/data/datasets/R2R_VLNCE_FSASub/val_seen/val_seen_sub.json.gz')
    # mlanet_gt_data = load_json_gz('/ssd/wangliuyi/code/w61_grutopia/data/datasets/R2R_VLNCE_FSASub/val_seen/val_seen.json.gz')
    # ori_data = load_json_gz('data/datasets/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz')
    # corrected_data = load_json_gz('data/datasets/R2R_VLNCE_v1-3_corrected/val_seen/val_seen.json.gz')
    print(1)