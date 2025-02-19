import os, sys
import lmdb
from tqdm import tqdm
import json
import gzip

def combine_lmdb(source_lmdb, add_lmdb, target_lmdb, commit_frequency=1000,added_prefix='_01'):
    source_env = lmdb.open(source_lmdb, readonly=True, lock=False)
    add_env = lmdb.open(add_lmdb, readonly=True, lock=False)

    os.makedirs(os.path.dirname(target_lmdb), exist_ok=True)
    
    # clear all existing contents
    with lmdb.open(
        target_lmdb,
        map_size=int(1e12),
    ) as target_env, target_env.begin(write=True) as txn:
        txn.drop(target_env.open_db())
    
    target_env = lmdb.open(target_lmdb, map_size=int(1e12), readonly=False)
    
    with source_env.begin() as txn:
        last_key = source_env.stat()["entries"]
        print(f"source_env has {last_key} keys")
    
    with add_env.begin() as txn:
        add_length = add_env.stat()["entries"]
        print(f"add_env has {add_length} keys")

    txn = target_env.begin(write=True)
    with source_env.begin() as source_txn:
        for i, (key, value) in enumerate(tqdm(source_txn.cursor(), desc="source_env", total=last_key)):
            txn.put(key, value)
            if (i+1) % commit_frequency == 0:
                txn.commit()
                txn = target_env.begin(write=True)
        
        txn.commit()
        txn = target_env.begin(write=True)
        
    with add_env.begin() as add_txn:
        for i, (key, value) in enumerate(tqdm(add_txn.cursor(), desc="add_env", total=add_length)):
            key_decoded = key.decode('utf-8')
            new_key = f"{key_decoded}{added_prefix}".encode()
            txn.put(new_key, value)
            if (i+1) % commit_frequency == 0:
                txn.commit()
                txn = target_env.begin(write=True)
        
        txn.commit()
        txn = target_env.begin(write=True)
    
    source_env.close()
    add_env.close()
    target_env.close()

def combine_dataset_json_gz(source_data_path, add_dataset_path, target_dataset_path, create_gt_file=False):
    source_data = read_data_file(source_data_path)
    add_data = read_data_file(add_dataset_path)
    target_data = source_data.copy()
    target_data['episodes'] = target_data['episodes'] + add_data['episodes']
    print("writing to file...")
    with gzip.open(target_dataset_path, 'wb') as f:
        f.write(json.dumps(target_data).encode('utf-8'))
    print(f"combined dataset saved to {target_dataset_path}")


def read_data_file(data_file):
    data = json.load(gzip.open(data_file, 'rb'))
    return data

if __name__ == "__main__":
    mode = 'combine_lmdb'

    if mode == 'combine_lmdb':
        source_lmdb_path = "data/sample_episodes/20250211_sample_origin/sample_data.lmdb"
        add_lmdb_path = "data/sample_episodes/20250214_sample_aliengo/sample_data.lmdb"
        target_lmdb_path = "data/sample_episodes/20250217_sample_h1_aliengo/sample_data.lmdb"
        combine_lmdb(source_lmdb_path, add_lmdb_path, target_lmdb_path, added_prefix='_aliengo')

