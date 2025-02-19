import os,sys
import json
import numpy as np
import gzip
from collections import defaultdict

from vln.parser import process_args
from vln.src.dataset.data_utils import load_data
from vln.src.utils.utils import euler_angles_to_quat, quat_to_euler_angles, compute_rel_orientations

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
    def __init__(self, args, dataset_root_dir=None, is_fsa_dataset=False):
        self.args = args
        self.splits = ['train', 'val_seen', 'val_unseen']
        # self.splits = ['envdrop']
        self.data = {split: [] for split in self.splits}
        self.scan = {}
        for split in self.splits:
            self.data[split], self.scan[split] = load_data(self.args, split, dataset_root_dir=dataset_root_dir, is_fsa_dataset=is_fsa_dataset)

    def gatherSameScanData(self, save_gather_data=True, save_dir='gather_data/', fix_rotation=False):
        scan2data = {split: {} for split in self.splits}
        for split in self.splits:
            for data in self.data[split]:
                scan = data['scan']
                if scan not in scan2data[split]:
                    scan2data[split][scan] = []
                if fix_rotation:
                    # no need. already turn in load_data.
                    data['start_rotation'] = transform_rotation_z_90degrees(data['start_rotation'])
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
    dataset_root_dir = "data/datasets/R2R_VLNCE_FSASub"
    # dataset_root_dir = "data/datasets/R2R_VLNCE_v1-3_corrected"

    args, _ = process_args()
    dataset_gather = datasetGather(args, dataset_root_dir=dataset_root_dir, is_fsa_dataset=True)
    scan2data = dataset_gather.gatherSameScanData(save_gather_data=True, save_dir='gather_data/', fix_rotation=False)
    
    # read_gather_data('/ssd/wangliuyi/code/w61_grutopia/data/datasets/R2R_VLNCE_FSASub/val_seen/val_seen_sub.json.gz')

    '''2. Gather eval data'''
    # val_seen_sample_dataset_file = "data/sample_episodes/20241115_sample_episodes_val_seen/analysis/success_episode_data_val_seen.json"
    # val_unseen_sample_dataset_file = "data/sample_episodes/20241115_sample_episodes_val_unseen/analysis/success_episode_data_val_unseen.json"
    
    # gather_eval_data(dataset_gather.data['val_unseen'], val_unseen_sample_dataset_file, 'val_unseen')
    # gather_eval_data(dataset_gather.data['val_seen'], val_seen_sample_dataset_file, 'val_seen')

    '''3. Check the dataset'''
    # mlanet_data = load_json_gz('/ssd/wangliuyi/code/w61_grutopia/data/datasets/R2R_VLNCE_FSASub/val_seen/val_seen_sub.json.gz')
    # mlanet_gt_data = load_json_gz('/ssd/wangliuyi/code/w61_grutopia/data/datasets/R2R_VLNCE_FSASub/val_seen/val_seen.json.gz')
    # ori_data = load_json_gz('data/datasets/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz')
    # corrected_data = load_json_gz('data/datasets/R2R_VLNCE_v1-3_corrected/val_seen/val_seen.json.gz')
    print(1)