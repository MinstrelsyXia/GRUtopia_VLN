import argparse
from vln import PROJECT_ROOT_PATH
import sys
import os
from vln.src.dataset.data_utils_multi_env import load_gather_data
from vln.src.utils.utils import Config
import os
import lmdb
import msgpack_numpy
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="cfg_file",
    )
    args = parser.parse_args()
    cfg_file = args.cfg_file
    print(f"cfg_file:{cfg_file}")

    project_path = PROJECT_ROOT_PATH
    cfg_file_path = f"{project_path}/{cfg_file}"
    if not os.path.exists(cfg_file_path):
        print(f"{cfg_file_path} not exist")
        sys.exit()
    with open(cfg_file_path, 'r') as file:
        config = json.load(file)
    print(f"config:{config}")

    split_number = config["total_rank"]
    task_type = config["task_type"]
    name = config["name"]
    print(f"split_number:{split_number}")
    print(f"task_type:{task_type}")
    print(f"name:{name}")
    if task_type == 'eval':
        split_data_types = ['val_unseen','val_seen']
        filter_same_trajectory = False
        prefix = "eval_rank"
    elif task_type == 'sample':
        split_data_types = ['train']
        filter_same_trajectory = True
        prefix = "sample_rank"
    else:
        print(f"unknown task_type:{task_type}")
        sys.exit()
    
    base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
    lmdb_path = project_path + f'/data/sample_episodes/{name}'

    #获取所有数据
    args_dict = {
        "datasets":{
            "base_data_dir":base_data_dir
        }
    }
    path_key_map = {}
    count=0
    for split_data_type in split_data_types:
        data_map, _ = load_gather_data(Config(args_dict), split_data_type, filter_same_trajectory=filter_same_trajectory, filter_stairs=True)
        for scan,path_list in data_map.items():
            path_key_list = []
            for path in path_list:
                trajectory_id = path['trajectory_id']
                episode_id = path['episode_id']
                path_key = f"{trajectory_id}_{episode_id}"
                path_key_list.append(path_key)
            path_key_map[scan]=path_key_list
            count += len(path_key_list)

    print(f"toatl:{count}")

    # 划分 rank
    rank_map = {}
    split_length = count // split_number
    index = -1
    for scan, path_key_list in path_key_map.items():
        for path_key in path_key_list:
            index += 1
            rank = index // split_length
            if rank >= split_number:
                rank = split_number - 1
            rank_map[path_key]=rank

    ranked_data = {}
    for i in range(split_number):
        filtered_path_key_map = {}
        for scan, path_key_list in path_key_map.items():
            filtered_list = []
            for path_key in path_key_list:
                if rank_map[path_key] == i:
                    filtered_list.append(path_key)
            if len(filtered_list) > 0:
                filtered_path_key_map[scan] = filtered_list
        ranked_data[i] = filtered_path_key_map

    for rank, path_key_map in ranked_data.items():
        count = 0
        for scan, path_key_list in path_key_map.items():
            count += len(path_key_list)
            print(f"[rank:{rank}][scan:{scan}][count:{len(path_key_list)}]")
        print(f"[rank:{rank}][count:{count}]")

    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
    with database.begin(write=True) as txn:
        for rank, path_key_map in ranked_data.items():
            key = f"{prefix}_{rank}".encode()
            value = msgpack_numpy.packb(path_key_map, use_bin_type=True)
            txn.put(key, value)
            print(f"finish [key:{key}]")
    database.close()