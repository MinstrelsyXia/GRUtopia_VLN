from vln.src.v2.dataloader.sample import SamplePathKeyDataloader
from vln.src.v2.envs.discrete_sample import DiscreteSampleSingleScanEnv
from vln.src.dataset.data_utils_multi_env import load_scene_usd
from vln.src.utils.utils import Config
from grutopia.core.config import SimulatorConfig
from vln.src.v2.util import common_log_util 
from vln import PROJECT_ROOT_PATH
import numpy as np
import sys

headless=True
target_scan="1LXtFkjw3qL"
target_trajectory = 67
rank=0
name = '20250207_sample_discrete'
split_data_types = ['train']
robot_offset = np.array([0.   , 0.   , 1.05])
project_path = PROJECT_ROOT_PATH
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
args_dict = {
    "datasets":{
        "mp3d_data_dir":mp3d_data_dir,
        "base_data_dir":base_data_dir,
    }
}
scene_asset_path = load_scene_usd(Config(args_dict), target_scan)
retry_list=['success']
lmdb_path = project_path + f'/data/sample_episodes/{name}'
common_log_util.init(name,rank)
dataloader=SamplePathKeyDataloader(
    base_data_dir,
    split_data_types,
    robot_offset,
    rank,
    lmdb_path,
    target_scan,
    retry_list,
    target_trajectory,
)
sample_path_key_list = dataloader.sample_path_key_list
if len(sample_path_key_list) == 0:
    print("no data")
    sys.exit(0)
path_zero = dataloader.path_key_data[sample_path_key_list[0]]
start_position = path_zero['start_position']
start_rotation = path_zero['start_rotation']

sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_eval.yaml'
sim_config = SimulatorConfig(sim_cfg_file)

env = DiscreteSampleSingleScanEnv(
    sim_config=sim_config,
    scene_asset_path=scene_asset_path,
    start_position=start_position,
    start_rotation=start_rotation,
    headless=headless,
    dataloader=dataloader,
)

env.sample()
env.stop()
