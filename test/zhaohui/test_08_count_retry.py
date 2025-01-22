import os
from dataclasses import dataclass
from vln.src.v2.dataloader.base import BasePathKeyDataloader

@dataclass(order=True)
class TrajectoryInfo:
    trajectory_id: str
    step_index: int
    duration: float
    step_count: int
    fps: float
    result:str

def read_file(file):
    data = []
    with open(file, 'r') as file:
        for line in file:
            if len(line) > 32:
                data.append(line[32:])
    return data

def parse_one_t(data):
    data = data.split(',')[0].split(' finish: ')
    step_index = int(data[0].split('][')[1][:-1].split(':')[-1])
    trajectory_id = data[1].split('][')[0][1:].split(':')[-1]
    durations = float(data[1].split('][')[1][1:].split(':')[-1][:-2])
    step_count = int(data[1].split('][')[2][1:].split(':')[-1])
    fps = float(data[1].split('][')[3][1:].split(':')[-1])
    result = data[1].split('][')[4][:-1].split(':')[-1]
    # if trajectory_id == '7179':
    #     print(trajectory_id)
    if result[-1] == ']':
        result = result[:-1]
    return TrajectoryInfo(
        trajectory_id = trajectory_id,
        step_index = step_index,
        duration = durations,
        step_count = step_count,
        fps = fps,
        result = result,
    )

def print_count(filtered_result_map, type):
    n_r = {}
    c = 0
    for _,result_list in filtered_result_map.items():
        before = result_list[0]
        after = result_list[-1]
        if before != type:
            continue
        c += 1
        if after in n_r:
            n_r[after] = n_r[after] + 1
        else:
            n_r[after] = 1
    print(f"{c} 个 {type} 重试结果如下：")
    for k,v in n_r.items():
        print(f"[result:{k}][count:{v}]")

import numpy as np

project_path = "/ssd/zhaohui/workspace/w61_grutopia_0102"
base_dir = f'{project_path}/logs/progress'
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
split_data_types = ['val_unseen'] #'val_seen'
robot_offset = np.array([0.   , 0.   , 1.05])
filter_same_trajectory=False


dataloader = BasePathKeyDataloader(
    base_data_dir,
    split_data_types,
    robot_offset,
    filter_same_trajectory,
)
path_key_data = dataloader.path_key_data

files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]

result_map = {}
for file in files:
    line_list = read_file(file)
    for line in line_list:
        if line[0] != '[':
            continue
        if line[1] == '0':
            continue
        t = parse_one_t(line)
        trajectory_id = t.trajectory_id
        result = t.result
        if trajectory_id not in path_key_data:
            continue
        if trajectory_id in result_map:
            result_map[trajectory_id].append(result)
        else:
            result_map[trajectory_id] = [result]
filtered_result_map = {}
for trajectory_id,result_list in result_map.items():
    if len(result_list) == 1:
        continue
    before = result_list[0]
    after = result_list[-1]
    filtered_result_map[trajectory_id] = [before,after]

print(f"split_data_types:{split_data_types}")
for type in ['stuck','fall']:
    print_count(filtered_result_map,type)