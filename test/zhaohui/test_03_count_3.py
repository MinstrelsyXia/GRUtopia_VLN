import lmdb
import msgpack_numpy
from vln.src.dataset.data_utils_multi_env import load_gather_data
from vln.src.utils.utils import Config

def generate_eval_key(ckpt_name, path_key):
    return f"eval_{ckpt_name}_{path_key}"

def get_split_map(project_path):
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
        path_key_list=[]
        for scan,path_list in load_data_map.items():
            for path in path_list:
                trajectory_id = path['trajectory_id']
                episode_id = path['episode_id']
                path_key = f"{trajectory_id}_{episode_id}"
                path_key_list.append(path_key)
        split_map[split_data_type]=path_key_list
    return split_map

ckpt_name="ckpt.70"
project_path = '/ssd/zhaohui/workspace/w61_grutopia_0107'
name = '20250112_eval_70'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
split_map = get_split_map(project_path)


for split,path_key_list in split_map.items():
    data_list = []
    for path_key in path_key_list:
        data_key = generate_eval_key(ckpt_name,path_key)
        with database_read.begin() as txn:
            value = txn.get(data_key.encode())
            if value is None:
                print(f"[key:{data_key}] value is None ")
                continue
            value = msgpack_numpy.unpackb(value)
        data_list.append(value)
    count=len(data_list)
    print(f"[split:{split}] 总共获取数据 {count} 条")
    total_TL = 0
    total_NE = 0
    total_osr = 0
    total_success = 0
    total_spl = 0

    for data in data_list:
        # TL Trajectory Length (TL) - 轨迹总长度 (0)
        TL = data['TL'] 
        # NE Navigation Error (NE) - 当前位置到目标的欧氏距离 (-1)
        NE = data['NE']
        if NE < 0:
            NE = 0
        # OS Oracle Success Rate (OSR) - 轨迹中是否有点达到目标(-1)
        osr = data['osr'] 
        if osr < 0:
            osr = 0
        # SR Success Rate (SR) - 是否到达目标点(0)
        success = data['success']
        # SPL (Success weighted by Path Length)(0)
        spl = data['spl']

        total_TL +=TL
        total_NE += NE
        total_osr += osr
        total_success += success
        total_spl += spl
    print(f"############[{split}]#############")
    print(f"TL = {total_TL} / {count} = {round((total_TL / count),2)}")
    print(f"NE = {total_NE} / {count} = {round((total_NE / count),2)}")
    print(f"osr = {total_osr} / {count} = {round((total_osr / count),2)}")
    print(f"success = {total_success} / {count} = {round((total_success / count),2)}")
    print(f"spl = {total_spl} / {count} = {round((total_spl / count),2)}")
    print(f"##########################")
database_read.close()