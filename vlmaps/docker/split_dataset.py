import argparse
import os
import json
import gzip
import numpy as np
import time

def find_gpu_id(trajectory_id, num_of_gpus, trajectory_id_list):
    for idx, id in enumerate(trajectory_id_list):
        if id == trajectory_id:
            return idx % num_of_gpus

def judge_stair_exists(path):
    '''
    path_z: ndarray, if max(path_z)- min(path_z) > 1.5, then there is a stair
    '''
    path_z = [p[1] for p in path]
    return max(path_z) - min(path_z) > 1.5

def get_sub_trajectory_id_list(base_path='./'):
    file_path = os.path.join(base_path, 'valid_paths/success_paths_id_all.txt')
    with open(file_path, 'r') as f:
        sub_trajectory_id_list = [int(line.strip()) for line in f.readlines()]
        # 重新按大小排序
        sub_trajectory_id_list.sort()
    return sub_trajectory_id_list


def load_and_split_data(data_dir, num_of_gpus, splits):
    dataset_root_dir = data_dir
    scene_id_list = []
    split_data = [[] for _ in range(num_of_gpus)]

    trajectory_id_list = get_sub_trajectory_id_list(os.path.dirname(__file__))
    for split in splits:
        with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
            data = json.load(f)
        for item in data["episodes"]:
            scene_id = item['scene_id'].split('/')[1]
            if scene_id not in scene_id_list:
                scene_id_list.append(scene_id)
        for item in data["episodes"]:
            if not judge_stair_exists(item['reference_path'])  and (item['trajectory_id'] in trajectory_id_list):

                scene_id = item['scene_id'].split('/')[1]
                gpu_id = find_gpu_id(scene_id, num_of_gpus, scene_id_list)
                split_data[gpu_id].append((scene_id,item['trajectory_id'],item['episode_id']))
                # print(scene_id,item['trajectory_id'],item['episode_id'])

    #! never sort the split_data by the scene_id
    # for i in range(num_of_gpus):
    #     split_data[i].sort(key=lambda x: x[0])
    return split_data

def save_split_data(split_data, output_dir, gpu_list):
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # 使用gpu_list中的值作为文件名
    for idx, (gpu_id, episodes) in enumerate(zip(gpu_list, split_data)):
        output_file = os.path.join(output_dir, f"{gpu_id}.txt")
        print(f"Writing data for GPU {gpu_id} to {output_file}")
        with open(output_file, 'w') as f:
            for scene_id, trajectory_id, episode_id in episodes:
                f.write(f"{scene_id},{trajectory_id},{episode_id}\n")

def main():
    parser = argparse.ArgumentParser(description='Split dataset for multi-GPU processing.')
    parser.add_argument('--gpu_list', nargs='+', type=str, help='List of GPU IDs to use.')
    parser.add_argument('--data_dir', type=str, default='/ssd/xiaxinyuan/code/VLN/VLNCE/R2R_VLNCE_v1-3', 
                        help='Directory of the dataset.')
    parser.add_argument('--output_dir', type=str, 
                        default=f'./multi_gpu_list_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}', 
                        help='Output directory for split data.')
    parser.add_argument('--splits', type=str, default=['train'], help='Splits to process.')
    args = parser.parse_args()

    # 使用GPU列表的长度
    num_gpus = len(args.gpu_list)
    print(f"Splitting data for {num_gpus} GPUs: {args.gpu_list}")

    split_data = load_and_split_data(args.data_dir, num_gpus, args.splits)
    save_split_data(split_data, args.output_dir, args.gpu_list)

if __name__ == "__main__":
    main()