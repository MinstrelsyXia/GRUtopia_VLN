import json
import os
import gzip
import copy
import argparse
import sys

# 添加正确的导入路径
from vln.src.v2.dataloader.data_reviser import skip_list

def load_json_file(json_file_path, data_split='val_unseen', target_dir='vlmaps/docker/valid_paths'):
    """
    直接加载JSON文件并处理数据
    
    Args:
        json_file_path: JSON文件的路径
        data_split: 数据集分割名称
        target_dir: 输出目录
    
    Returns:
        episode_ids列表
    """
    # 确保目标目录存在
    main_dir = os.path.join(target_dir, data_split)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    
    # 加载JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    episode_all = []
    # 处理每个场景的数据
    for scene_id, trajectories in data.items():
        for traj in trajectories:
            episode_id = traj['episode_id']
            episode_all.append(episode_id)
            print(f"处理场景 {scene_id} 中的轨迹 {episode_id}")
    
    # 保存episode ID到txt文件
    txt_file = os.path.join(main_dir, "episode_ids.txt")
    with open(txt_file, 'w') as f:
        for episode in episode_all:
            f.write(f"{episode}\n")
    
    # 创建分类文件
    object_path_file = os.path.join(main_dir, "object.txt")
    object_id_file = os.path.join(main_dir, "object_id.txt")
    room_path_file = os.path.join(main_dir, "room.txt") 
    room_id_file = os.path.join(main_dir, "room_id.txt")
    
    # 清空文件内容
    for file_path in [object_path_file, object_id_file, room_path_file, room_id_file]:
        with open(file_path, 'w') as f:
            pass
    
    # 分类处理轨迹
    for scene_id, trajectories in data.items():
        for traj in trajectories:
            episode_id = traj['episode_id']
            trajectory_id = traj['trajectory_id']
            instruction = traj['instruction']['instruction_text']
            
            # 检查是否有楼梯(通过检查路径z坐标变化)
            has_stairs = judge_stair_exists(traj['reference_path'])
            
            if not has_stairs:
                if not ('room' in instruction.lower() or 'hall' in instruction.lower() or 'way' in instruction.lower()):
                    with open(object_path_file, 'a') as f:
                        f.write(f"{scene_id},{trajectory_id},{episode_id},{instruction}\n")
                    with open(object_id_file, 'a') as f:
                        f.write(f"{episode_id}\n")
                else:
                    with open(room_path_file, 'a') as f:
                        f.write(f"{scene_id},{trajectory_id},{episode_id},{instruction}\n")
                    with open(room_id_file, 'a') as f:
                        f.write(f"{episode_id}\n")
    
    print(f"共处理 {len(episode_all)} 个轨迹")
    return episode_all

def get_sub_trajectory_id_list(txt_file):
    """读取txt文件中的轨迹ID列表"""
    with open(txt_file, 'r') as f:
        sub_trajectory_id_list = [int(line.strip()) for line in f.readlines()]
        # 重新按大小排序
        sub_trajectory_id_list.sort()
    return sub_trajectory_id_list

def judge_stair_exists(path):
    '''
    判断路径中是否存在楼梯
    path: 路径点列表
    如果最大z坐标与最小z坐标差值 > 1.5，则认为存在楼梯
    '''
    path_z = []
    for point in path:
        path_z.append(point[2])  # 假设z坐标是第二个元素
    return max(path_z) - min(path_z) > 1.5

def find_gpu_id(trajectory_id, num_of_gpus, trajectory_id_list):
    """根据轨迹ID分配GPU"""
    for idx, id in enumerate(trajectory_id_list):
        if (id == trajectory_id):
            return idx % num_of_gpus

def load_data_zip(splits, num_of_gpus, data_split='val_unseen', target_dir='vlmaps/docker/valid_paths'):
    """
    加载数据并处理
    """
    starrt_pos_dict = {}
    episode_id = []
    trajectory_id_list = []

    target_dir = os.path.join(target_dir, data_split)

    object_path_file = os.path.join(target_dir, "object.txt")
    object_id_file = os.path.join(target_dir, "object_id.txt")
    room_path_file = os.path.join(target_dir, "room.txt") 
    room_id_file = os.path.join(target_dir, "room_id.txt")
    with open(object_path_file, 'w') as f:
        pass
    with open(object_id_file, 'w') as f:
        pass
    with open(room_path_file, 'w') as f:
        pass
    with open(room_id_file, 'w') as f:
        pass

    dataset_root_dir = '/ssd/xiaxinyuan/code/VLN/VLNCE/R2R_VLNCE_v1-3'
    scene_id_list = []
    total_scans = []
    load_data = []
    split_data = [[] for _ in range(num_of_gpus)]
    trajectory_id_list = get_sub_trajectory_id_list(os.path.join(target_dir, "episode_ids.txt"))
    print(len(trajectory_id_list))
    for split in splits:
        with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
            data = json.load(f)
        for item in data["episodes"]:
            scene_id = item['scene_id'].split('/')[1]
            if scene_id not in scene_id_list:
                scene_id_list.append(scene_id)
        for item in data["episodes"]:
            if not judge_stair_exists(item['reference_path'])  and (item['trajectory_id'] in trajectory_id_list):
                instruction = item['instruction']['instruction_text']
                scene_id = item['scene_id'].split('/')[1]
                if not ('room' in instruction or 'hall' in instruction or 'way' in instruction):
                    with open(object_path_file, 'a') as f:
                        f.write(f"{scene_id},{item['trajectory_id']},{item['episode_id']},{item['instruction']['instruction_text']}\n")
                    with open(object_id_file, 'a') as f:
                        f.write(f"{item['episode_id']}\n")
                else:
                    with open(room_path_file, 'a') as f:
                        f.write(f"{scene_id},{item['trajectory_id']},{item['episode_id']},{item['instruction']['instruction_text']}\n")
                    with open(room_id_file, 'a') as f:
                        f.write(f"{item['episode_id']}\n")
    return split_data

def split_dataset_for_multi_gpu(data_dir, gpu_num, output_dir=None, max_num = 150):
    """
    读取object.txt和object_id.txt文件，将数据分配到多个GPU
    
    Args:
        data_dir: 数据目录，包含object.txt文件
        gpu_num: GPU数量
        output_dir: 输出目录，默认为data_dir的父目录下的multi_gpu_{gpu_num}目录
    
    Returns:
        分配给每个GPU的数据列表
    """
    # 设置默认输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(data_dir), f"multi_gpu_{gpu_num}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取object.txt文件获取场景ID
    object_path_file = os.path.join(data_dir, "room.txt")
    
    # 读取场景信息
    scene_data = []
    scene_id_list = []
    
    with open(object_path_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                scene_id, trajectory_id, episode_id = parts[0], parts[1], parts[2]
                scene_data.append((scene_id, trajectory_id, episode_id))
                if scene_id not in scene_id_list:
                    scene_id_list.append(scene_id)
    total_scene_num = len(scene_data)
    if total_scene_num > max_num:
        import random
        random.seed(0)
        random.shuffle(scene_data)
        scene_data = scene_data[:max_num]
        # 按照scene_id, trajectory_id, episode_id排序
        scene_data.sort(key=lambda x: (x[0], x[1], x[2]))
    print(f"总场景数: {total_scene_num}, 选取场景数: {len(scene_data)}")
    scene_id_list = []
    for scene_id, trajectory_id, episode_id in scene_data:
        if scene_id not in scene_id_list:
            scene_id_list.append(scene_id)
    # 将场景分配给GPU
    scene_gpu_map = {}
    for i, scene_id in enumerate(scene_id_list):
        scene_gpu_map[scene_id] = i % gpu_num
    
    # 按GPU分组数据
    gpu_data = [[] for _ in range(gpu_num)]
    for scene_id, trajectory_id, episode_id in scene_data:
        gpu_idx = scene_gpu_map[scene_id]
        gpu_data[gpu_idx].append((scene_id, trajectory_id, episode_id))
    
    # 写入文件
    for i in range(gpu_num):
        output_file = os.path.join(output_dir, f"{i}.txt")
        with open(output_file, 'w') as f:
            for scene_id, trajectory_id, episode_id in gpu_data[i]:
                f.write(f"{scene_id},{trajectory_id},{episode_id}\n")
        print(f"已写入GPU {i}数据到文件: {output_file}, 共{len(gpu_data[i])}条数据")
    
    return gpu_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='deal with dataset')
    parser.add_argument('--action', type=str, choices=['load', 'split'], default='load',
                      help='Executed Movement: load=load json file, split=split dataset')
    parser.add_argument('--json_file', type=str,
                      default='/ssd/xiaxinyuan/dataset/grutopia10/gather_data/val_unseen_gather_data.json',
                      help='json file route')
    parser.add_argument('--data_split', type=str, default='val_unseen',
                      help='splited dataset')
    parser.add_argument('--target_dir', type=str, default='vlmaps/docker/valid_paths/r2r/',
                      help='target output')
    parser.add_argument('--gpu_num', type=int, default=2,
                      help='GPU num')
    
    args = parser.parse_args()
    
    if args.action == 'load':
        episodes = load_json_file(args.json_file, args.data_split, args.target_dir)
        print(f"Processed {len(episodes)} trajectories")
    elif args.action == 'split':
        data_dir = os.path.join(args.target_dir, args.data_split)
        output_dir = os.path.join(args.target_dir, f"multi_gpu_{args.gpu_num}")
        gpu_data = split_dataset_for_multi_gpu(data_dir, args.gpu_num, output_dir)
        print(f"Split the dataset to {args.gpu_num} gpus")
        for i, data in enumerate(gpu_data):
            print(f"GPU {i}: {len(data)} data")

