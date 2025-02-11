from vln.src.dataset.data_utils_multi_env import load_gather_data
from vln.src.utils.utils import Config
import os
import lmdb
import msgpack_numpy

#参数
split_number = 16
# split_data_types = ['val_unseen','val_seen']
split_data_types = ['train']
filter_same_trajectory = True
project_path = '/ssd/zhaohui/workspace/w61_grutopia_0107'
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
name = '20250120_dagger'
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
        key = f"sample_rank_{rank}".encode()
        value = msgpack_numpy.packb(path_key_map, use_bin_type=True)
        txn.put(key, value)
        print(f"finish [key:{key}]")
database.close()