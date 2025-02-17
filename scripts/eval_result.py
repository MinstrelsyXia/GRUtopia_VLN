import lmdb
import msgpack_numpy
import sys
from vln import PROJECT_ROOT_PATH
import os
import json
import argparse
from vln.src.v2.util.eval import generate_eval_key
from vln.src.v2.util.common import load_data

def get_split_map(project_path):
    split_map={}
    base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
    split_data_types=['val_seen', 'val_unseen']
    for split_data_type in split_data_types:
        load_data_map = load_data(base_data_dir, split_data_type, filter_same_trajectory=False, filter_stairs=True)
        path_key_list=[]
        for scan,path_list in load_data_map.items():
            for path in path_list:
                trajectory_id = path['trajectory_id']
                episode_id = path['episode_id']
                path_key = f"{trajectory_id}_{episode_id}"
                path_key_list.append(path_key)
        split_map[split_data_type]=path_key_list
    return split_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="cfg_file",
    )
    args = parser.parse_args()
    cfg_file = args.cfg_file
    
    # 创建日志文件
    log_content = []
    def log_print(content):
        print(content)
        log_content.append(str(content))

    log_print(f"cfg_file:{cfg_file}")

    project_path = PROJECT_ROOT_PATH
    cfg_file_path = f"{project_path}/{cfg_file}"
    if not os.path.exists(cfg_file_path):
        log_print(f"{cfg_file_path} not exist")
        sys.exit()
    with open(cfg_file_path, 'r') as file:
        config = json.load(file)
    log_print(f"config:{config}")
    name = config["name"]
    log_print(f"name:{name}")
    ckpt_file_name = config["ckpt_to_load"].split('/')[-1]
    ckpt_name=f"{name}_{ckpt_file_name}"
    log_print(f"ckpt_name:{ckpt_name}")
    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, readonly=True, lock=False)
    split_map = get_split_map(project_path)

    for split,path_key_list in split_map.items():
        data_list = []
        for path_key in path_key_list:
            data_key = generate_eval_key(ckpt_name,path_key)
            with database_read.begin() as txn:
                value = txn.get(data_key.encode())
                if value is None:
                    # print(f"[key:{data_key}] value is None ")
                    continue
                value = msgpack_numpy.unpackb(value)
            value['path_key']=path_key
            data_list.append(value)
        count=len(data_list)
        log_print(f"[split:{split}] 总共获取数据 {count} 条")
        total_TL = 0
        total_NE = 0
        total_osr = 0
        total_success = 0
        total_spl = 0
        reason_map = {
            "reach_goal":0
        }

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

            ret_type=data['fail_reason']
            if ret_type == '':
                ret_type = 'success'
            if ret_type not in reason_map:
                reason_map[ret_type] = 1
            else:
                reason_map[ret_type] = reason_map[ret_type] + 1
            if success > 0:
                reason_map['reach_goal']= reason_map['reach_goal'] + 1

        log_print(f"############[{split}]#############")
        if count == 0:
            log_print(f"############[count == 0,skip]#############")
            continue
        log_print(f"TL = {total_TL} / {count} = {round((total_TL / count),4)}")
        log_print(f"NE = {total_NE} / {count} = {round((total_NE / count),4)}")
        log_print(f"FR = {reason_map['fall']} / {count} = {round((reason_map['fall'] / count),4) * 100}%")
        if 'stuck' in reason_map:
            log_print(f"StR = {reason_map['stuck']} / {count} = {round((reason_map['stuck'] / count),4) * 100}%")
        else:
            log_print(f"StR = 0 / {count} = 0%")
        log_print(f"OS = {total_osr} / {count} = {round((total_osr / count),4) * 100}%")
        log_print(f"SR = {total_success} / {count} = {round((total_success / count),4) * 100}%")
        log_print(f"SPL = {total_spl} / {count} = {round((total_spl / count),4) * 100}%")
        log_print("detail:")
        for k,v in reason_map.items():
            log_print(f"[{k}]:{v}")
        log_print(f"##########################")
    
    # 将日志内容写入文件
    log_file_path = os.path.join(lmdb_path, 'eval.log')
    with open(log_file_path, 'w') as f:
        f.write('\n'.join(log_content))
    
    database_read.close()