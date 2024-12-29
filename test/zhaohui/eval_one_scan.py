import argparse
import time
from context import Context
import os
import threading
import sys

def check_process_stuck(context:Context):
    index = 0
    while True:
        index+=1
        current_time = time.time()
        duration = round(current_time - context.timestamp,2)
        if  duration > 600:
            print("10分钟时间戳未更新,杀死进程")
            os.kill(os.getpid(), 9) 
        else:
            if index % 60 == 0:
                print(f"check_process_stuck 存活[{context.timestamp}]")
        sys.stdout.flush()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="rank",
    )
    parser.add_argument(
        "--scan",
        type=str,
        required=True,
        help="scan",
    )
    args = parser.parse_args()
    rank = args.rank
    scan = args.scan
    
    context = Context(rank, scan)
    monitor_thread = threading.Thread(target=check_process_stuck, args=(context,))
    monitor_thread.start()

    try:
        print("context.run()")
        context.run()
        os.kill(os.getpid(), 9) 
    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        monitor_thread.join()
        print("Program terminated.")