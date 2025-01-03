from vln.src.v2.dataloader.base import BasePathKeyDataloader
import numpy as np
from vln.src.v2.util.eval import(
    generate_result_key
)
import lmdb
import msgpack_numpy

split_data_types = ['val_seen'] #'val_seen'
project_path = '/ssd/zhaohui/workspace/w61_grutopia_0102'
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
robot_offset = np.array([0.   , 0.   , 1.05])
filter_same_trajectory=False
ckpt_name="ckpt.cma"
name = '20250102_ckpt_cma'
lmdb_path = project_path + f'/data/sample_episodes/{name}'

dataloader = BasePathKeyDataloader(
    base_data_dir,
    split_data_types,
    robot_offset,
    filter_same_trajectory,
)

database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
path_key_data = dataloader.path_key_data
path_key_scan = dataloader.path_key_scan
path_key_split = dataloader.path_key_split
for path_key in path_key_data:
    info_key = generate_result_key(ckpt_name,path_key)
    with database_read.begin() as txn:
        info_value = txn.get(info_key.encode())
        if info_value is None:
            print(f"[key:{info_key}] value is None")
        info_value = msgpack_numpy.unpackb(info_value)
        fail_reason = info_value['fail_reason']
        steps = info_value['steps']
        if fail_reason == 'fall' and steps < 100:
            scan = path_key_scan[path_key]
            split = path_key_split[path_key]
            print(f"[split:{split}][scan:{scan}][path_key:{path_key}][step:{steps}]")