import lmdb
import msgpack_numpy
from vln.src.dataset.data_utils_multi_env import load_gather_data
from vln.src.utils.utils import Config

def generate_result_key(ckpt_name, path_key):
    return f"eval_{ckpt_name}_{path_key}"

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
    split_data_types=['val_unseen','val_seen']
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
ckpt_name="ckpt.cma"
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
name = '20250101_sample_episodes'
# split = 'val_seen'
# split = 'val_unseen'
split = ''
lmdb_path = project_path + f'/data/sample_episodes/{name}'
database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
if split != '':
    split_map = get_split_map()

all_rank_total_count=0
all_rank_finist_count=0
all_rank_success_count=0

all_rank_count_map={
        "success":0,
        "exceed_per_action_max_step":0,
        "exceed_total_max_step":0,
        "stuck":0,
        "fall":0,
        "not_reach_goal":0,
        "goal_in_obstacle":0,
        "open_set_empty":0,
        "path_planning":0,
    }

for rank in ranks:
    count_map={
        "success":0,
        "exceed_per_action_max_step":0,
        "exceed_total_max_step":0,
        "stuck":0,
        "fall":0,
        "not_reach_goal":0,
        "goal_in_obstacle":0,
        "open_set_empty":0,
        "path_planning":0,
    }
    key = f"eval_rank_{rank}".encode()
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
            if split != '':
                if split_map[path_key] != split:
                    continue
            total_count += 1
            info_key = generate_result_key(ckpt_name,path_key)
            with database_read.begin() as txn:
                info_value = txn.get(info_key.encode())
                if info_value is None:
                    continue
                info_value = msgpack_numpy.unpackb(info_value)
                finist_count +=1
                fail_reason=info_value['fail_reason']
                if fail_reason != '':
                    count_map[fail_reason] = count_map[fail_reason] + 1
                else:
                    count_map['success'] = count_map['success'] + 1
                success = info_value['success']
                if success > 0:
                    success_count += 1
    print(f"[rank:{rank}][split:{split}][ {finist_count} / {total_count}][success:{success_count}]:{count_map}")
    all_rank_total_count +=total_count
    all_rank_finist_count +=finist_count
    all_rank_success_count +=success_count
    for k,v in count_map.items():
        all_rank_count_map[k] = all_rank_count_map[k] + v

print(f"[all_rank][split:{split}][ {all_rank_finist_count} / {all_rank_total_count}][success:{all_rank_success_count}]:{all_rank_count_map}")
database_read.close()