import logging
import os
from dataclasses import dataclass
import time
from vln.src.v2.util.common_log_util import get_name
from vln import PROJECT_ROOT_PATH

progress_logger = logging.getLogger('progress_logger')
progress_logger.setLevel(logging.INFO)

@dataclass(order=True)
class TrajectoryInfo:
    trajectory_id: str
    start_time: time
    end_time: time
    start_step: int
    end_step: str
    result:str


class ProgressInfo:
    def __init__(self, scan ,path_count):
        self.scan = scan
        self.path_count = path_count
        self.info_map = {}
        self.start = None
        self.end = None

PROGRESS = None
LAST_TRIJECTORY_ID = ""
INITED = False

def init(scan, path_count,rank=0):
    global PROGRESS
    global INITED
    PROGRESS = ProgressInfo(scan,path_count)
    log_dir = f"{PROJECT_ROOT_PATH}/logs/{get_name()}/progress/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(f'{log_dir}/scan_{scan}_rank_{rank}.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    progress_logger.addHandler(file_handler)
    INITED = True

def last_log(trajectory_id=None, step_count=-1):
    global PROGRESS
    global LAST_TRIJECTORY_ID
    last_info = PROGRESS.info_map[LAST_TRIJECTORY_ID]
    last_str = f"[{len(PROGRESS.info_map) - 1}/{PROGRESS.path_count}][step_index:{step_count}] finish: [trajectory_id:{last_info.trajectory_id}]"
    duration = round(last_info.end_time - last_info.start_time, 2)
    step_count = last_info.end_step - last_info.start_step
    fps = round((step_count / (duration + 1e-10)) ,2)
    last_str = last_str + f"[duration:{duration} s]"
    last_str = last_str + f"[step_count:{step_count}]"
    last_str = last_str + f"[fps:{fps}]"
    last_str = last_str + f"[result:{last_info.result}]"
    if trajectory_id is None or trajectory_id == "":
        progress_logger.info(f"{last_str}")
    else:
        progress_logger.info(f"{last_str},start sampling trajectory_id: {trajectory_id}")

def trace_start(trajectory_id, step_count):
    global INITED
    if not INITED:
        return
    global PROGRESS
    global LAST_TRIJECTORY_ID
    start_time = time.time()
    ti = TrajectoryInfo(
        trajectory_id = trajectory_id,
        start_time = start_time,
        start_step = step_count,
        end_time = None,
        end_step = None,
        result = None,
    )
    PROGRESS.info_map[trajectory_id] = ti
    if LAST_TRIJECTORY_ID == "":
        progress_logger.info(f"[{0}/{PROGRESS.path_count}][step_index:{step_count}] start sampling trajectory_id: {trajectory_id}")
        PROGRESS.start = time.time()
    else:
        last_log(trajectory_id, step_count)

    LAST_TRIJECTORY_ID = trajectory_id

def trace_end(trajectory_id, step_count, result):
    global INITED
    if not INITED:
        return
    global PROGRESS
    end_time = time.time()
    ti = PROGRESS.info_map[trajectory_id]
    ti.end_time = end_time
    ti.end_step = step_count
    ti.result = result
    PROGRESS.info_map[trajectory_id] = ti

def report():
    global PROGRESS
    global LAST_TRIJECTORY_ID
    result_map = {}
    for _, v in PROGRESS.info_map.items():
        result = v.result
        if result in result_map:
            result_map[result] = result_map[result] + 1
        else:
            result_map[result] = 1
    last_log()
    PROGRESS.end = time.time()
    last_info = PROGRESS.info_map[LAST_TRIJECTORY_ID]
    duration = round((PROGRESS.end - PROGRESS.start) ,2)
    step_count = last_info.end_step
    fps = round((step_count / (duration + 1e-10)) ,2)
    progress_logger.info(f"scan:{PROGRESS.scan} finished. [duration: {duration} s][step_count: {step_count}][fps :{fps}] result: {result_map}")