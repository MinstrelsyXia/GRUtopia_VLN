import os,sys
import json
from collections import defaultdict

from vln.parser import process_args
from vln.src.dataset.data_utils import load_data

class datasetGather:
    def __init__(self, args, dataset_root_dir=None):
        self.args = args
        self.splits = ['train', 'val_seen', 'val_unseen']
        # self.splits = ['envdrop']
        self.data = {split: [] for split in self.splits}
        self.scan = {}
        for split in self.splits:
            self.data[split], self.scan[split] = load_data(self.args, split, dataset_root_dir=dataset_root_dir)

    def gatherSameScanData(self, save_gather_data=True, save_dir='gather_data/'):
        scan2data = {split: {} for split in self.splits}
        for split in self.splits:
            for data in self.data[split]:
                scan = data['scan']
                if scan not in scan2data[split]:
                    scan2data[split][scan] = []
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


def gather_eval_data(dataset_file, split, save_dir='gather_data/'):
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    scan_data = defaultdict(list)
    
    for item in dataset:
        scan = item['scan']
        scan_data[scan].append(item)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'{split}_PReval_gather_data.json')
    with open(save_path, 'w') as f:
        json.dump(scan_data, f, indent=2)
    print(f'Saved eval data for {split} to {save_path}')
    

if __name__ == "__main__":
    '''1. Gather standard dataset'''
    # dataset_root_dir = "/isaac-sim/GRUtopia/data/datasets/revised/corrected"

    # args, _ = process_args()
    # dataset_gather = datasetGather(args, dataset_root_dir=dataset_root_dir)
    # scan2data = dataset_gather.gatherSameScanData(save_gather_data=True, save_dir='gather_data/')
    
    # read_gather_data('gather_data/train_gather_data.json')

    '''2. Gather eval data'''
    val_seen_dataset_file = "/ssd/wangliuyi/code/GRUtopia/data/sample_episodes/20241115_sample_episodes_val_seen/analysis/success_episode_data_val_seen.json"
    val_unseen_dataset_file = "/ssd/wangliuyi/code/GRUtopia/data/sample_episodes/20241115_sample_episodes_val_unseen/analysis/success_episode_data_val_unseen.json"

    gather_eval_data(val_unseen_dataset_file, 'val_unseen')
