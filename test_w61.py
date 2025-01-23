import lmdb

lmdb_file = '/ssd/wangliuyi/code/w61_grutopia/data/sample_episodes/20250122_seq2seq_eval/sample_data.lmdb'

# 打开LMDB环境
env = lmdb.open(lmdb_file, readonly=True)

# 创建事务
with env.begin() as txn:
    # 创建游标
    cursor = txn.cursor()
    
    # 遍历所有的key-value对
    for key, value in cursor:
        # 将bytes类型的key解码为字符串
        key_str = key.decode('utf-8')
        print(f"Key: {key_str}")

# 关闭环境
env.close()