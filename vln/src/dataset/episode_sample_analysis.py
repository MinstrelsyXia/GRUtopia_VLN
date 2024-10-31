''' To analyze the collection of episodes
Author: w61
Date: 2024.09.08
'''
import os,sys
import json
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOG_DIR = os.path.join(ROOT_DIR, 'logs', 'sample_episodes')
SPLITS = ['train', 'val_seen', 'val_unseen']

def analyze_episodes_per_scan(split, scan):
    success_path_id_file = os.path.join(LOG_DIR, split, scan,'success_path_id.txt')
    success_path_ids = []
    if os.path.exists(success_path_id_file):
        with open(success_path_id_file, 'r') as f:
            success_path_id = f.readlines()
        for path_id in success_path_id:
            path_id = path_id.replace('\n', '')
            success_path_ids.append(f"id_{path_id}")
    else:
        for path_dir in os.listdir(os.path.join(LOG_DIR, split, scan)):
            if os.path.exists(os.path.join(LOG_DIR, split, scan, path_dir, 'status_info.txt')):
                with open(os.path.join(LOG_DIR, split, scan, path_dir,'status_info.txt'), 'r') as f:
                    final_line = f.readlines()[-1]
                    if 'Episode finished: True' in final_line:
                        success_path_ids.append(path_dir)
    success_step_nums = []
    fail_reason_dict = {
        'fall': 0,
        'stuck': 0,
        'path planning': 0,
        'maximum step': 0
    }

    # 1. analyze the successful episode's average steps
    for path_id in success_path_ids:
        path_id = path_id.replace('\n', '')
        path_dir = os.path.join(LOG_DIR, split, scan, path_id)
        pose_file = os.path.join(path_dir, 'poses.txt')
        # count the step numbers according to the lines of the poses file
        with open(pose_file, 'r') as f:
            lines = f.readlines()
        step_num = len(lines)
        success_step_nums.append(step_num)

    # 2. analyze the failure episode's reason
    for path_dir in os.listdir(os.path.join(LOG_DIR, split, scan)):
        if path_dir in success_path_ids:
            continue
        status_info_file = os.path.join(LOG_DIR, split, scan, path_dir,'status_info.txt')
        if os.path.exists(status_info_file):
            with open(status_info_file, 'r') as f:
                # status_info is the last line of the file
                status_line = f.readlines()[-1]
                if 'Episode finished' in status_line and '.' in status_line:
                    fail_reason = status_line.split('.')[-1][1:].replace('\n','')
                    fail_reason_dict[fail_reason] += 1

    return success_path_ids, success_step_nums, fail_reason_dict

def analyze_episodes(sample_episodes_dir, save_details=False):
    split_info_dict = defaultdict(dict)
    for split in SPLITS:
        # scan is the sub dictionary of sample_episodes_dir
        for scan in os.listdir(os.path.join(sample_episodes_dir, split)):
            success_path_ids, success_step_nums, fail_reason_dict = analyze_episodes_per_scan(split, scan)
            split_info_dict[split][scan] = {
                'success_path_ids': success_path_ids,
                'success_step_nums': success_step_nums,
                'fail_reason_dict': fail_reason_dict
            }
    
    # analyze the scan number of successful episodes
    success_scan_nums = defaultdict(int)
    success_path_nums = defaultdict(int)
    failure_reason_nums = defaultdict(int)
    for split in SPLITS:
        for scan in split_info_dict[split]:
            if len(split_info_dict[split][scan]['success_path_ids']) > 0:
                success_scan_nums[split] += 1
                success_path_nums[split] += len(split_info_dict[split][scan]['success_path_ids'])
            for reason in split_info_dict[split][scan]['fail_reason_dict']:
                failure_reason_nums[reason] += split_info_dict[split][scan]['fail_reason_dict'][reason]
    
    split_info_dict['success_scan_nums'] = success_scan_nums
    split_info_dict['success_path_nums'] = success_path_nums
    split_info_dict['failure_reason_nums'] = failure_reason_nums

    if save_details:
        save_file = os.path.join(LOG_DIR, 'episode_details.txt')
        with open(save_file, 'w') as f:
            json.dump(split_info_dict, f, indent=4)
            print(f'Episode details saved to {save_file}.txt')
    print(1)


if __name__ == '__main__':
    analyze_episodes(LOG_DIR)