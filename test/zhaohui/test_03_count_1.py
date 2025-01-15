import os
import time
from dataclasses import dataclass

project_path = "/ssd/zhaohui/workspace/w61_grutopia_0107"
base_dir = f'{project_path}/20250110_dagger.log/progress/'

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

def get_total_path(datas):
    for data in datas:
        if data[:5] == 'start':
            return int(data.split(':')[-1][:-1])

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

def get_t_info_map(datas):
    t_info_map = {}
    for data in datas:
        if data[:5] == 'start':
            continue
        elif  data[:4] == 'scan':
            continue
        elif data[:3] == '[0/':
            if len(data.split('finish:')) == 1:
                continue
        t = parse_one_t(data)
        t_info_map[t.trajectory_id] = t
    return t_info_map

def get_t_info_list(datas):
    t_info_list = []
    for data in datas:
        if data[:5] == 'start':
            continue
        elif  data[:4] == 'scan':
            continue
        elif data[:3] == '[0/':
            if len(data.split('finish:')) == 1:
                continue
        t = parse_one_t(data)
        t_info_list.append(t)
    return t_info_list


files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]

total = 0
total_duration = 0
result={
    "success":0,
    "exceed_per_action_max_step":0,
    "exceed_total_max_step":0,
    "stuck":0,
    "fall":0,
    "not_reach_goal":0,
    "goal_in_obstacle":0,
    "open_set_empty":0,
    "path_planning":0,
    "fast_fall":0,
    "path planning":0,
    "max_step":0,
}

duration_map={
    "success":0,
    "exceed_per_action_max_step":0,
    "exceed_total_max_step":0,
    "stuck":0,
    "fall":0,
    "not_reach_goal":0,
    "goal_in_obstacle":0,
    "open_set_empty":0,
    "path_planning":0,
    "fast_fall":0,
    "path planning":0,
    "max_step":0,
}

scan_file_map={}
for file in files:
    scan = file.split('/')[-1].split('.')[0].split('_')[1]
    if scan not in scan_file_map:
        scan_file_map[scan]=[file]
    else:
        old = scan_file_map[scan]
        old.append(file)
        scan_file_map[scan]=old

for scan, file_list in scan_file_map.items():

    total_t_info_map = {}
    for file in file_list:
        datas = read_file(file)
        t_info_map = get_t_info_map(datas)
        for k,v in t_info_map.items():
            total_t_info_map[k] = v
    print(f"[scan:{scan}] 总共抓取数据{len(total_t_info_map)}条")
    for k,v in total_t_info_map.items():
        total = total + 1
        result[v.result] = result[v.result] + 1
        duration_map[v.result]= duration_map[v.result] + v.duration
        total_duration += v.duration
   
print(f"总共抓取数据{total}条,其中")
for type,num in result.items():
    if num == 0:
        continue
    duration = round(duration_map[type] / num,2)
    print(f"[{type}]{num} 条 ,平均每条耗时：{duration} 秒")

print(f"平均每条数据处理耗时：{round(total_duration)} / {total} = {round(total_duration / total  ,2)}")