import argparse
import os
import json
import gzip

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

def load_and_split_data(data_dir, num_of_gpus, splits, type = 'all'):
    dataset_root_dir = data_dir
    trajectory_id_list = []
    split_data = [[] for _ in range(num_of_gpus)]
    if type == 'all':
        for split in splits:
            with gzip.open(os.path.join(dataset_root_dir, f"{split}", f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
                data = json.load(f)
                for item in data["episodes"]:
                    if item['trajectory_id'] not in trajectory_id_list:
                        trajectory_id_list.append(item['trajectory_id'])
                for item in data["episodes"]:
                    if judge_stair_exists(item['reference_path']):
                        # print(item['episode_id'], item['trajectory_id'], item['instruction']['instruction_text'])
                        continue
                    gpu_id = find_gpu_id(item['trajectory_id'], num_of_gpus, trajectory_id_list)
                    split_data[gpu_id].append((item['episode_id'], item['trajectory_id']))
    else:
        with open()
        trajectory_id_list = 
    return split_data

def save_split_data(split_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, episodes in enumerate(split_data):
        with open(os.path.join(output_dir, f"{idx}.txt"), 'w') as f:
            for episode_id, trajectory_id in episodes:
                # Write both episode_id and trajectory_id
                f.write(f"{episode_id},{trajectory_id}\n")
    
    # with open(os.path.join(output_dir, f"last_scan_{idx}.txt"), 'w') as f:
    #     f.write(str())

def main():
    parser = argparse.ArgumentParser(description='Split dataset for multi-GPU processing.')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use.')
    parser.add_argument('--data_dir', type=str, default = '/ssd/xiaxinyuan/code/VLN/VLNCE/R2R_VLNCE_v1-3', help='Directory of the dataset.')
    parser.add_argument('--output_dir', type=str, default='/vlmaps/docker/multi_gpu_list', help='Output directory for split data.')
    #! 注意和vln/config/vln_cfg_vlmap.yaml中的splits一致性
    parser.add_argument('--splits', type=str, default=['val_seen'], help='Splits to process.')
    args = parser.parse_args()

    split_data = load_and_split_data(args.data_dir, args.num_gpus, args.splits)
    save_split_data(split_data, args.output_dir)

if __name__ == "__main__":
    main()