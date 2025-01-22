import argparse
from vln import PROJECT_ROOT_PATH
import sys
import os
import json
import lmdb
import msgpack_numpy
from vln.src.dataset.data_utils_multi_env import load_gather_data
from vln.src.utils.utils import Config
from vln.src.v2.util.eval import generate_eval_key


def get_split_map(
    project_path,
    split_data_types,
    filter_same_trajectory,
):
    split_map={}
    base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
    mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
    args_dict = {
        "datasets":{
            "mp3d_data_dir":mp3d_data_dir,
            "base_data_dir":base_data_dir,
        }
    }
    for split_data_type in split_data_types:
        load_data_map, _ = load_gather_data(Config(args_dict), split_data_type, filter_same_trajectory=filter_same_trajectory, filter_stairs=True)
        for scan,path_list in load_data_map.items():
            for path in path_list:
                trajectory_id = path['trajectory_id']
                episode_id = path['episode_id']
                path_key = f"{trajectory_id}_{episode_id}"
                split_map[path_key] = split_data_type
    return split_map

def count_one_rank(
    database_read,
    rank,
    ckpt_name,
    prefix,
    task_type,
    split=None,
    split_map=None,
):
    total_count=0
    finist_count=0
    success_count=0
    count_map={}

    key = f"{prefix}_{rank}".encode()
    with database_read.begin() as txn:
        value = txn.get(key)
        if value is None:
            print.info(f"[rank{rank}]获取抓取列表失败")
            return
        value = msgpack_numpy.unpackb(value)
    for scan, path_key_list in value.items():
        for path_key in path_key_list:
            if split is not None:
                if split_map[path_key] != split:
                    continue
            total_count+=1
            if task_type == 'eval':
                info_key = generate_eval_key(ckpt_name,path_key)
            else:
                info_key = path_key.split('_')[0]
            with database_read.begin() as txn:
                info_value = txn.get(info_key.encode())
                if info_value is None:
                    continue
                info_value = msgpack_numpy.unpackb(info_value)
                finist_count +=1
                ret_type=info_value['fail_reason']
                if ret_type == '':
                    ret_type = 'success'
                if ret_type not in count_map:
                    count_map[ret_type] = 1
                else:
                    count_map[ret_type] = count_map[ret_type] + 1
                if task_type == 'eval':
                    success = info_value['success']
                    if success > 0:
                        success_count += 1
                else:
                    if ret_type == 'success':
                        success_count += 1
    return total_count, finist_count, success_count, count_map

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
    device_map = config["device_map"]
    print(f"device_map:{device_map}")
    name = config["name"]
    print(f"name:{name}")
    ranks=[]
    for rank, gpus in device_map.items():
        ranks.append(int(rank))
    print(f"ranks:{ranks}")
    ckpt_file_name = config["ckpt_to_load"].split('/')[-1]
    ckpt_name=f"{name}_{ckpt_file_name}"
    print(f"ckpt_name:{ckpt_name}")

    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, readonly=True, lock=False)
    
    task_type = config["task_type"]
    print(f"task_type:{task_type}")
    if task_type == "eval":
        split_data_types=['val_unseen','val_seen']
        filter_same_trajectory=False
        prefix="eval_rank"
    elif task_type == "sample":
        split_data_types=["train"]
        filter_same_trajectory=True
        prefix="sample_rank"
    else:
        print(f"unknown task_type: {task_type}")
    if len(split_data_types) > 1:
        split_map = get_split_map(
            project_path,
            split_data_types,
            filter_same_trajectory,
        )
    print(f"############################[total]###########################")
    all_rank_total_count=0
    all_rank_finist_count=0
    all_rank_success_count=0
    all_rank_count_map={}
    for rank in ranks:
        total_count, finist_count, success_count, count_map = count_one_rank(
            database_read,
            rank,
            ckpt_name,
            prefix,
            task_type,
        )
        print(f"[rank:{rank}][split:all][ {finist_count} / {total_count}][success:{success_count}]:{count_map}")
        all_rank_total_count +=total_count
        all_rank_finist_count +=finist_count
        all_rank_success_count +=success_count
        for k,v in count_map.items():
            if k not in all_rank_count_map:
                all_rank_count_map[k] = v
            else:
                all_rank_count_map[k] = all_rank_count_map[k] + v
    print(f"[all_rank][split:all][ {all_rank_finist_count} / {all_rank_total_count}][success:{all_rank_success_count}]:{all_rank_count_map}")
    
    if len(split_data_types) > 1:
        for split in split_data_types:
            print(f"############################[split:{split}]###########################")
            all_rank_total_count=0
            all_rank_finist_count=0
            all_rank_success_count=0
            all_rank_count_map={}
            for rank in ranks:
                total_count, finist_count, success_count, count_map = count_one_rank(
                    database_read,
                    rank,
                    ckpt_name,
                    prefix,
                    task_type,
                    split,
                    split_map,
                )
                print(f"[rank:{rank}][split:{split}][ {finist_count} / {total_count}][success:{success_count}]:{count_map}")
                all_rank_total_count +=total_count
                all_rank_finist_count +=finist_count
                all_rank_success_count +=success_count
                for k,v in count_map.items():
                    if k not in all_rank_count_map:
                        all_rank_count_map[k] = v
                    else:
                        all_rank_count_map[k] = all_rank_count_map[k] + v
            print(f"[all_rank][split:{split}][ {all_rank_finist_count} / {all_rank_total_count}][success:{all_rank_success_count}]:{all_rank_count_map}")
            
print(f"##############################################################")
database_read.close()