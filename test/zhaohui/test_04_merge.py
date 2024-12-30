import lmdb
import msgpack_numpy
import sys

def generate_result_key(ckpt_name, path_key):
    return f"eval_{ckpt_name}_{path_key}"


ranks =list(range(8,16))
ckpt_name="ckpt.44"
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
name = '20241229_sample_episodes'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
all_path_key_list=[]
for rank in ranks:
    # 获取该rank 所有需要抓取的数据
    key = f"eval_rank_{rank}"
    with database.begin() as txn:
        value = txn.get(key.encode())
        value = msgpack_numpy.unpackb(value)
        if value is None:
            print(f"[rank:{rank}] value of {key} from {lmdb_path}/sample_data.lmdb is None")
            sys.exit()
        for scan,path_key_list in value.items():
            for path_key in path_key_list:
                all_path_key_list.append(path_key)
database.close()
print(f"total_path_key:{len(all_path_key_list)}")
ids = []
for path_key in all_path_key_list:
    ids.append(generate_result_key(ckpt_name,path_key))

total = len(ids)

lmdb_path_0507 = f'{project_path}/data/sample_episodes/{name}/sample_data.lmdb'
lmdb_path_0508 = f'{project_path}/data_0508/sample_episodes/{name}/sample_data.lmdb'
index = 0
env_0508 = lmdb.open(lmdb_path_0508, readonly=True, lock=False)
env_0507 = lmdb.open(lmdb_path_0507, map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
for id in ids:
    index = index + 1
    desc = f'[ {index} / {total} ]'
    key = f"{id}".encode()
    with env_0508.begin() as txn:
        value = txn.get(key)
        if value is None:
            print(f"{desc} g0508 value is None")
            continue
    with env_0507.begin(write=True) as txn:
        # value = txn.get(key)
        # if value is not None:
        #     print(f"{desc} g0507 value already exist")
        #     continue
        txn.put(key, value)
    print(f"{desc} done!")
env_0508.close()
env_0507.close()