import os
import lmdb
import msgpack_numpy

project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
name = '20241216_sample_episodes'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
rank = 0
key = f"eval_rank_{rank}".encode()
with database.begin() as txn:
    value = txn.get(key)
    value = msgpack_numpy.unpackb(value)
    if value is None:
        print(f"value is None")
    else:
        print({value})