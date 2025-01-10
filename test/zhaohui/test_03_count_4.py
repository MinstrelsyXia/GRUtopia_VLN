import lmdb
import msgpack_numpy
from vln.src.dataset.data_utils_multi_env import load_gather_data
from vln.src.utils.utils import Config

def get_split_map():
    split_map={}
    base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
    mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
    args_dict = {
        "datasets":{
            "mp3d_data_dir":mp3d_data_dir,
            "base_data_dir":base_data_dir,
        }
    }
    split_data_types=['train']
    for split_data_type in split_data_types:
        load_data_map, _ = load_gather_data(Config(args_dict), split_data_type, filter_same_trajectory=False, filter_stairs=True)
        for scan,path_list in load_data_map.items():
            for path in path_list:
                trajectory_id = path['trajectory_id']
                episode_id = path['episode_id']
                path_key = f"{trajectory_id}_{episode_id}"
                split_map[path_key] = split_data_type
    return split_map

ranks =list(range(0,8))
project_path = '/ssd/zhaohui/workspace/w61_grutopia_0107'
name = '20250110_dagger'

lmdb_path = project_path + f'/data/sample_episodes/{name}'
database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)

all_rank_total_count=0
all_rank_finist_count=0
all_rank_success_count=0

all_rank_count_map={
    "success":0,
    "max_step":0,
    "fast_fall":0,
    "stuck":0,
    "fall":0,
}

for rank in ranks:
    count_map={
        "success":0,
        "max_step":0,
        "fast_fall":0,
        "stuck":0,
        "fall":0,
    }
    key = f"sample_rank_{rank}".encode()
    with database_read.begin() as txn:
        value = txn.get(key)
        if value is None:
            print.info(f"[rank{rank}]获取抓取列表失败")
            continue
        value = msgpack_numpy.unpackb(value)
    total_count = 0
    finist_count = 0
    success_count=0
    for scan,path_key_list in value.items():
        for path_key in path_key_list:
            total_count += 1
            trajectory_id = path_key.split('_')[0]
            info_key = str(trajectory_id)
            with database_read.begin() as txn:
                info_value = txn.get(info_key.encode())
                if info_value is None:
                    continue
                info_value = msgpack_numpy.unpackb(info_value)
                finist_count +=1
                fail_reason=info_value['fail_reason']
                count_map[fail_reason] = count_map[fail_reason] + 1
                if fail_reason == 'success':
                    success_count += 1
    print(f"[rank:{rank}][ {finist_count} / {total_count}][success:{success_count}]:{count_map}")
    all_rank_total_count +=total_count
    all_rank_finist_count +=finist_count
    all_rank_success_count +=success_count
    for k,v in count_map.items():
        all_rank_count_map[k] = all_rank_count_map[k] + v

print(f"[all_rank][ {all_rank_finist_count} / {all_rank_total_count}][success:{all_rank_success_count}]:{all_rank_count_map}")
database_read.close()