import logging
import os
from vln import PROJECT_ROOT_PATH
from datetime import datetime
NAME = None

common_logger = logging.getLogger('common_logger')
common_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)
common_logger.addHandler(console_handler)
common_logger.disabled=False

def init(name='default',rank='0'):
    global NAME
    NAME=name
    log_dir = f"{PROJECT_ROOT_PATH}/logs/{name}/common/rank_{rank}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_name = f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}.log" 
    file_handler = logging.FileHandler(f'{log_dir}/{file_name}')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    common_logger.addHandler(file_handler)


def get_name():
    global NAME
    return NAME