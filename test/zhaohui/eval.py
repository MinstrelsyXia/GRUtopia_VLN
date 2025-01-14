import argparse
import lmdb
import msgpack_numpy
import sys
import time
import subprocess
import threading

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
    args = parser.parse_args()
    rank = args.rank
    print(f"rank:{rank}")

    project_path = '/isaac-sim/GRUtopia'
    name = '20250114_eval_navid'
    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    
    round_count = 0
    while True:
        round_count +=1

        # 获取该rank 所有需要抓取的数据
        database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
        key = f"eval_rank_{rank}"
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
                f'{project_path}/test/zhaohui/eval_one_scan.py',
                '--rank',
                str(rank),
                '--scan',
                scan,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout_thread, stderr_thread = start_reading_threads(process)
            process.wait()
            stdout_thread.join()
            stderr_thread.join()
            exit_code = process.returncode
            print(f"[round:{round_count}][scan:{scan}][exit_code:{exit_code}]")
                
        time.sleep(60)