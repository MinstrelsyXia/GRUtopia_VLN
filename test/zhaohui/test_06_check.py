import threading
import time
import logging
import os
latest_time = time.time()

# 定义一个测试函数
def test():
    global latest_time
    # 模拟测试操作，这里可以是任何需要执行的任务
    time.sleep(1)  # 模拟一些工作时间
    # 重新触发监控线程的检查
    latest_time = time.time()

# 定义监控线程
def monitor_test():
    global latest_time
    while True:
        current_time = time.time()
        diff = round(current_time - latest_time,2)
        if  diff > 10:
            print("10 秒没有调用了")
            os.kill(os.getpid(), 9) 
        else:
            print(f"正常调用:{diff}")
        time.sleep(2)  # 每1秒检查一次

# 启动监控线程
monitor_thread = threading.Thread(target=monitor_test)
monitor_thread.start()

# 主程序逻辑
try:
    while True:
        test()
        time.sleep(250)
except KeyboardInterrupt:
    logging.info("Program stopped by user.")
finally:
    monitor_thread.join()
    logging.info("Program terminated.")