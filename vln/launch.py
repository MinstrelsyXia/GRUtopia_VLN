import argparse
import lmdb
import msgpack_numpy
import sys
import time
import subprocess
import threading
import json
import os
from vln import PROJECT_ROOT_PATH

def start_reading_threads(process):
    def read_output(stream):
        for line in iter(stream.readline, ""):
            if line:
                print(line.rstrip())
        print('stream.close()')
        stream.close()
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr,))
    stdout_thread.start()
    stderr_thread.start()
    return stdout_thread, stderr_thread

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="rank",
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="cfg_file",
    )
    args = parser.parse_args()
    rank = args.rank
    cfg_file = args.cfg_file
    print(f"rank:{rank}")
    print(f"cfg_file:{cfg_file}")
    project_path = PROJECT_ROOT_PATH
    cfg_file_path = f"{project_path}/{cfg_file}"
    if not os.path.exists(cfg_file_path):
        print(f"{cfg_file_path} not exist")
        sys.exit()
    with open(cfg_file_path, 'r') as file:
        config = json.load(file)
    
    name = config["name"]
    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    task_type=config["task_type"]
    if task_type == 'eval':
        key_prefix = 'eval_rank'
        task_file = "eval_one_scan.py"
    elif task_type == 'sample':
        key_prefix = 'sample_rank'
        task_file = "sample_one_scan.py"
    elif task_type == 'sixth_floor':
        key_prefix = 'sample_rank'
        task_file = 'sample_one_scan.py'
    else:
        print(f'unknown task_type: {task_type}')
        sys.exit()
    
    round_count = 0
    while True:
        round_count +=1

        # 获取该rank 所有需要抓取的数据
        database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, readonly=True, lock=False)
        key = f"{key_prefix}_{rank}"
        with database.begin() as txn:
            value = txn.get(key.encode())
            value = msgpack_numpy.unpackb(value)
            if value is None:
                print(f"value of {key} from {lmdb_path}/sample_data.lmdb is None")
                sys.exit()
        database.close()
        #获取所有 rank 列表
        scan_list = []
        for k,v in value.items():
            scan_list.append(k)
        for scan in scan_list:
            print(f"[round:{round_count}]开始抓取scan:{scan}")
            command = [
                'python', 
                f'{project_path}/vln/src/task/{task_file}',
                '--rank',
                str(rank),
                '--scan',
                scan,
                "--cfg_file",
                cfg_file,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout_thread, stderr_thread = start_reading_threads(process)
            process.wait()
            stdout_thread.join()
            stderr_thread.join()
            exit_code = process.returncode
            print(f"[round:{round_count}][scan:{scan}][exit_code:{exit_code}]")
                
        time.sleep(60)