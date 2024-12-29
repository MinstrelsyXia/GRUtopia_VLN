import lmdb
import msgpack_numpy

def generate_result_key(ckpt_name, path_key):
    return f"eval_{ckpt_name}_{path_key}"

ranks =[8,9,10,11,12,13,14,15] 
ckpt_name="ckpt.44"
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
name = '20241229_sample_episodes'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)

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
    print(f"[rank:{rank}][ {finist_count} / {total_count}][success:{success_count}]:{count_map}")
    all_rank_total_count +=total_count
    all_rank_finist_count +=finist_count
    all_rank_success_count +=success_count
    for k,v in count_map.items():
        all_rank_count_map[k] = all_rank_count_map[k] + v

print(f"[all_rank][ {all_rank_finist_count} / {all_rank_total_count}][success:{all_rank_success_count}]:{all_rank_count_map}")
