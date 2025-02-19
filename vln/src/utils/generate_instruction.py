import gzip
import os
import json
import copy
import gzip
import os
import json
from tqdm import tqdm
import random

def load_data(dataset_file=None, change_from_habitat=True):
    ''' Load data based on VLN-CE
    '''
    total_scans = []
    load_data = []
    with gzip.open(dataset_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for idx, item in enumerate(data["episodes"]):
            item["original_start_position"] = copy.copy(item["start_position"])
            item["original_start_rotation"] = copy.copy(item["start_rotation"])
            if change_from_habitat:
                item["start_position"] = [item["original_start_position"][0], -item["original_start_position"][2], item["original_start_position"][1]]
                item["start_rotation"] = [-item["original_start_rotation"][3], item["original_start_rotation"][0], item["original_start_rotation"][2], -item["original_start_rotation"][1]] # [x,y,z,-w] => [w,x,y,z]
            else:
                item["start_position"] = item["original_start_position"]
                item["start_rotation"] = item["original_start_rotation"]
            # item["start_rotation"] = transform_rotation_z_90degrees(item["start_rotation"])
            if '/' in item["scene_id"]:
                item["scan"] = item["scene_id"].split("/")[1]
            else:
                item["scan"] = item["scene_id"]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    if change_from_habitat:
                        item["c_reference_path"].append([path[0], -path[2], path[1]])
                    else:
                        item["c_reference_path"].append(path)
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data.append(item)
            total_scans.append(item["scan"])

    print(f"Loaded data with a total of {len(load_data)} items.")
    return load_data

def generate_new_dataset(source_dataset_file, rgb_dir, output_dir, instr_dir='inst_short_gpt'):
    print(f'Use the instr_dir: {instr_dir}')
    total_instr_num = 3
    os.makedirs(output_dir, exist_ok=True)

    # 读取源数据集
    with gzip.open(source_dataset_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    trajId2instr = {}

    # 遍历output_dir下的所有文件夹
    for episode_id in tqdm(os.listdir(rgb_dir), desc="Processing episodes"):
        episode_path = os.path.join(rgb_dir, episode_id)
        
        # 确保是目录而不是文件
        if not os.path.isdir(episode_path):
            continue
            
        # 构建指令目录的路径
        instr_path = os.path.join(episode_path, instr_dir)
        
        # 初始化当前episode的指令列表
        current_instructions = []
        
        # 读取0.txt, 1.txt, 2.txt
        for i in range(total_instr_num):
            instr_file = os.path.join(instr_path, f"{i}.txt")
            if os.path.exists(instr_file):
                with open(instr_file, 'r', encoding='utf-8') as f:
                    instruction = f.read().strip()
                    current_instructions.append(instruction)
        
        # 将指令存入字典
        if current_instructions:
            trajId2instr[episode_id] = current_instructions
    
    # 更新源数据集中的指令
    new_episodes_data = []
    episodes_data = data["episodes"]
    no_instr_epids = []
    for episode in episodes_data:
        episode_id = str(episode["episode_id"])
        if episode_id in trajId2instr:
            # 更新指令
            for instr_id, instr in enumerate(trajId2instr[episode_id]):
                new_item = copy.deepcopy(episode)
                new_item["trajectory_id"] = new_item["episode_id"]
                new_item["episode_id"] = new_item["episode_id"]*total_instr_num + int(instr_id)
                new_item["instruction"]["instruction_text"] = instr
                new_episodes_data.append(new_item)
        else:
            no_instr_epids.append(episode_id)
    
    print(f"There are {len(no_instr_epids)} episodes without instructions")
    
    # 创建新的数据集
    new_data = {
        "episodes": new_episodes_data
    }
    
    # 保存新数据集
    output_file = os.path.join(output_dir, "sixth_floor_with_instr.json.gz")
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Generated new dataset with {len(new_episodes_data)} episodes")
    return new_data

def split_dataset(dataset_file, output_dir, train_ratio=0.8):
    with gzip.open(dataset_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机分配数据到训练集和验证集
    episodes = data["episodes"]
    filter_episodes = []
    for ep in episodes:
        if len(ep["instruction"]["instruction_text"]) == 0:
            filter_episodes.append(ep)
    episodes = filter_episodes  

    total_length = len(episodes)
    train_size = int(total_length * train_ratio)
    
    train_data = []
    val_seen_data = []

    random.shuffle(episodes)
    
    train_data = episodes[:train_size]
    val_seen_data = episodes[train_size:]
    
    # 创建新的数据集字典
    train_dataset = {"episodes": train_data}
    val_seen_dataset = {"episodes": val_seen_data}
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集和验证集
    train_output = os.path.join(output_dir, "train.json.gz")
    val_seen_output = os.path.join(output_dir, "val_seen.json.gz")
    
    with gzip.open(train_output, 'wt', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    
    with gzip.open(val_seen_output, 'wt', encoding='utf-8') as f:
        json.dump(val_seen_dataset, f, indent=2)
    
    print(f"Dataset split complete:")
    print(f"Total episodes: {total_length}")
    print(f"Training episodes: {len(train_data)}")
    print(f"Validation episodes: {len(val_seen_data)}")
    
    return train_dataset, val_seen_dataset

if __name__ == "__main__":
    six_floor_dataset_file = "/ailab/user/wangliuyi/code/w61_grutopia/data/datasets/sixth_floor/sixth_floor_with_instr.json.gz"

    # six_floor_data = load_data(six_floor_dataset_file, change_from_habitat=False)
    # vlnce_data = load_data(vlnce_dataset_file, change_from_habitat=True)
    # print(six_floor_data)
    # print(vlnce_data)

    # generate_new_dataset(six_floor_dataset_file, "data/six_floor_rgbs", "data/outputs")
    split_dataset(six_floor_dataset_file, "data/datasets/sixth_floor", train_ratio=0.8)

