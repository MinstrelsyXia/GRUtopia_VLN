# Author: w61
# Date: 2024.7.11
''' Demo for loading vlnce dataset and intialize the H1 robot
'''
import os,sys
import gzip
import json
import math
import numpy as np
import argparse

from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container

parser = argparse.ArgumentParser(description="Demo for loading vlnce dataset and intialize the H1 robot.")
parser.add_argument("--env", default="val_seen", type=str, help="The split of the dataset", choices=['train', 'val_seen', 'val_unseen'])
parser.add_argument("--path_id", type=int, help="The number of path id")
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

file_path = './GRUtopia/demo/configs/h1_vlnce.yaml'
sim_config = SimulatorConfig(file_path)

def load_data(file_path, path_id=None):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    target_item = None
    if path_id is not None:
        for item in data['episodes']:
            if item['trajectory_id'] == path_id:
                target_item = item
                break
    if target_item is None:
        print(f"Path ID {path_id} is invalid and randomly set a path id")
        target_item = data['episodes'][0]
    scan = target_item['scene_id'].split('/')[1]
    start_position = [target_item['start_position'][0], -target_item['start_position'][2], target_item['start_position'][1]]
    start_rotation = [-target_item['start_rotation'][3], target_item['start_rotation'][0], target_item['start_rotation'][1], target_item['start_rotation'][2]] # [x,y,z,-w] => [w,x,y,z]
    return target_item, scan, start_position, start_rotation

data_item, data_scan, start_position, start_rotation = load_data(sim_config.config_dict['datasets'][0]['base_data_dir']+f"/{args.env}/{args.env}.json.gz", args.path_id)

find_flag = False
for root, dirs, files in os.walk(sim_config.config_dict['datasets'][0]['mp3d_data_dir']+f"/{data_scan}"):
    for file in files:
        if file.endswith(".usd") and "non_metric" not in file:
            scene_usd_path = os.path.join(root, file)
            find_flag = True
            break
    if find_flag:
        break

sim_config.config.tasks[0].scene_asset_path = scene_usd_path
sim_config.config.tasks[0].robots[0].position = start_position # TODO: lack rotation settings!!

headless = args.headless
webrtc = False

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name

i = 0

actions = {'h1': {'move_with_keyboard': []}}

while env.simulation_app.is_running():
    i += 1
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)

    if i % 100 == 0:
        print(i)

env.simulation_app.close()
