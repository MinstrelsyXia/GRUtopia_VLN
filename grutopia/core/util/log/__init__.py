import os
from typing import Optional
from datetime import datetime
import toml
from pydantic import BaseModel

from grutopia.core.util.log.logger import Logger

'''
from grutopia.core.util.log import log

def main():
    log.info("This is an info message")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.critical("This is a critical message")
'''

# Define the configuration model
class LogConfig(BaseModel):
    filename: Optional[str] = None
    level: Optional[str] = 'info'
    fmt: Optional[str] = '[%(asctime)s][%(levelname)s] %(pathname)s[line:%(lineno)d] -: %(message)s'

# Path to the config file
config_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'config.ini')

# Read the existing configuration
with open(config_path, 'r') as f:
    config = LogConfig(**(toml.loads(f.read())['log']))

# Generate a unique log filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join("GRUtopia", "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
unique_log_filename = os.path.join(log_dir, f'_log_{timestamp}.txt')
config.filename = unique_log_filename

# Initialize the logger with the updated configuration
log = Logger(
    filename=config.filename,
    level=config.level,
    fmt=config.fmt,
).log
