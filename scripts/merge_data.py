import lmdb
import msgpack_numpy
import sys
from vln import PROJECT_ROOT_PATH
import os
import json
import argparse
from vln.src.v2.util.eval import generate_eval_key

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
    name = config["name"]
    print(f"name:{name}")
    merge_config = config["merge_config"]
    print(f"merge_config:{merge_config}")
    task_type = config["task_type"]
    print(f"task_type:{task_type}")
    if task_type == 'eval':
        ckpt_file_name = config["ckpt_to_load"].split('/')[-1]
        ckpt_name=f"{name}_{ckpt_file_name}"
        print(f"ckpt_name:{ckpt_name}")

    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
    for folder, rank_list in merge_config.items():
        sub_lmdb_path = project_path + f'/{folder}/sample_episodes/{name}'
        if not os.path.exists(sub_lmdb_path):
            print(f"sub_lmdb_path {sub_lmdb_path} not exist!")
            sys.exit()
        sub_database = lmdb.open(f"{sub_lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, readonly=True, lock=False)
        all_path_key_list=[]
        for rank in rank_list:
            if task_type == 'eval': 
                key = f"eval_rank_{rank}"
            elif task_type == 'sample': 
                key = f"sample_rank_{rank}"
            with database.begin() as txn:
                value = txn.get(key.encode())
                value = msgpack_numpy.unpackb(value)
                if value is None:
                    print(f"[rank:{rank}] value of {key} from {lmdb_path}/sample_data.lmdb is None")
                    sys.exit()
                for scan,path_key_list in value.items():
                    for path_key in path_key_list:
                        all_path_key_list.append(path_key)
        
        ids = []
        for path_key in all_path_key_list:
            if  task_type == 'eval': 
                ids.append(generate_eval_key(ckpt_name,path_key))
            elif task_type == 'sample': 
                trajectory_id = path_key.split('_')[0]
                ids.append(trajectory_id)
        print(f"total_path_key:{len(ids)}")
        print(f"#############################开始合并数据#############################")
        print(f"源目录:{sub_lmdb_path}")
        print(f"目标目录:{lmdb_path}")
        print(f"总数据量:{len(ids)}")
        total = len(ids)
        index = 0   
        for id in ids:
            index = index + 1
            desc = f'[ {index} / {total} ]'
            key = id.encode()
            with sub_database.begin() as txn:
                value = txn.get(key)
                if value is None:
                    print(f"{desc} [key:{id}] of {folder} not exist")
                    continue
            with database.begin(write=True) as txn:
                txn.put(key, value)
            print(f"{desc} done!")
        print(f"#############################数据合并完成#############################")
        sub_database.close()
    database.close()