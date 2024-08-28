# Author: w61
# Date: 2024.08.27
''' Automatic script to sample data for VLN in isaac-sim
'''
import os,sys
import gzip
import json
import math
import numpy as np
import argparse
import yaml
import time
import shutil
from collections import defaultdict
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

# enable multiple gpus
# import isaacsim
# import carb.settings
# settings = carb.settings.get_settings()
# settings.set("/renderer/multiGPU/enabled", True)


from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
from grutopia.core.util.log import log

# from grutopia_extension.utils import get_stage_prim_paths

import matplotlib.pyplot as plt

from vln.src.dataset.data_utils import VLNDataLoader
from vln.src.utils.utils import dict_to_namespace
# from vln.src.local_nav.global_topdown_map import GlobalTopdownMap


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSAC_SIM_DIR = os.path.join(os.path.dirname(ROOT_DIR), "isaac-sim-4.0.0")
sys.path.append(ISSAC_SIM_DIR)

from parser import process_args

import subprocess
import sys

def build_dataset():
    ''' Build dataset for VLN
    '''
    vln_config, sim_config = process_args()
    vln_datasets = {}
    scan_file = os.path.join(vln_config.datasets.base_data_dir, "gather_data", "env_scan.json")
    with open(scan_file, 'r') as f:
        scan_data = json.load(f)
    
    return scan_data

def run_scans(scan_data):
    start_time = time.time()
    
    # 遍历每个数据集分割
    for key, scans in scan_data.items():
        print(f"处理数据集分割: {key}，包含 {len(scans)} 个扫描")
        
        # 使用 tqdm 显示进度条
        for scan in tqdm(scans, desc=f"Processing scans in {key}", unit="scan"):
            print(f"正在处理scan: {scan}")
            
            # 构建命令
            command = [sys.executable, 'vln/main.py', '--scan', scan, '--split', key,
                       '--headless', 
                       '--vln_cfg_file', 'vln/configs/vln_extract_data_script.yaml', 
                       '--sim_cfg_file', 'vln/configs/sample_episodes_sim_cfg.yaml',
                       '--save_path_planning']
            
            try:
                # 执行main.py，并等待其完成
                subprocess.run(command, check=True)
                print(f"Scan {scan} 处理完成")
            except subprocess.CalledProcessError as e:
                print(f"处理scan {scan} 时发生错误: {e}")
            
            print("---")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"所有扫描处理完成，总耗时: {total_time:.2f}秒")


if __name__ == "__main__":
    scan_data = build_dataset()

    run_scans(scan_data)