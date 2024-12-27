from vln.src.models.init_policy import initialize_policy
from vln.src.utils.logger import MyLogger
from vln.src.utils.utils import Config
import logging
import os
import torch
from vln.src.dataset.data_utils_multi_env import load_gather_data
import lmdb
import msgpack_numpy
import sys
from grutopia.core.config import SimulatorConfig
from vln.src.dataset.data_utils_multi_env import load_scene_usd
import numpy as np
from grutopia.core.env import BaseEnv

def get_shortest_path(robot_pose):

    global_freemap_camera_pose = self.cam_occupancy_map_global_list[env_idx].topdown_camera.get_world_pose()[0] - self.tasks[self.task_names[env_idx]]._offset
    global_freemap, _ = self.cam_occupancy_map_global_list[env_idx].get_global_free_map(robot_pos=robot_pose[0],robot_height=1.55, update_camera_pose=False, verbose=verbose)

    # freemap, camera_pose = get_global_free_map_single(self.env_idx, verbose=False)
headless = False
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
config_dict={
    "local_rank":0,
    "DDP":{
        "use":False,
    },
    "TORCH_GPU_IDS": [0],
    "fp16":False,
    "seed":0,
    "MODEL":{
        "policy_name":"CMA_Policy",
        "ablate_instruction":False,
        "ablate_depth":False,
        "ablate_rgb":False,
        "normalize_rgb":False,
        "INSTRUCTION_ENCODER":{
            "sensor_uuid": "instruction",
            "vocab_size": 2504,
            "use_pretrained_embeddings": True,
            "embedding_file": f"{project_path}/data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz",
            "dataset_vocab": f"{project_path}/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz",
            "fine_tune_embeddings": False,
            "embedding_size": 50,
            "hidden_size": 128,
            "rnn_type": "LSTM",
            "final_state_only": True,
            "bidirectional": True,
        },
        "RGB_ENCODER":{
            "cnn_type": "TorchVisionResNet50",
            "output_size": 256,
            "trainable": False,
        },
        "DEPTH_ENCODER":{
            "cnn_type": "VlnResnetDepthEncoder",
            "output_size": 128,
            "backbone": "resnet50",
            "ddppo_checkpoint": f"{project_path}/data/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth",
            "trainable": False,
        },
        "STATE_ENCODER":{
            "hidden_size": 512,
            "rnn_type": "GRU"
        },
        "PROGRESS_MONITOR":{
            "use": False,
            "alpha": 1.0,
        }
    },
    "IL":{
        "ckpt_to_load": f"{project_path}/data/checkpoints/habitat_sotas/CMA_PM_DA_Aug_converted.pth",
        "lr_schedule":{
            "use":True,
            "type": "cosine",
            "min_lr": 1e-5,
        },
        "epochs": 60,
        "lr": 1e-4,
    }
}
config = Config(config_dict)
local_rank=0
log_dir="/ssd/zhaohui/workspace/w61_grutopia_1220/test/zhaohui/"
base_data_dir = '/ssd/share/VLN/VLNCE/R2R_VLNCE_v1-3_corrected'
mp3d_data_dir = "/ssd/share/Matterport3D/data/v1/scans"
split_data_types = ['val_unseen','val_seen']
project_path = '/ssd/zhaohui/workspace/w61_grutopia_1220'
name = '20241216_sample_episodes'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
the_scan = "zsNo4HB9uLZ"

eval_logger_filename = os.path.join(log_dir, f"eval.log")
eval_logger = MyLogger(
    name="eval", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
    filename=eval_logger_filename
)
device = torch.device("cuda", local_rank)
policy, _, _, _ = initialize_policy(
    config,
    eval_logger,
    load_from_ckpt=True,
    device=device,
    load_from_pretrain=False,
    action_stats=None,
)
policy.eval()

args_dict = {
    "datasets":{
        "mp3d_data_dir":mp3d_data_dir,
        "base_data_dir":base_data_dir,
    }
}
data_map = {}
for split_data_type in split_data_types:
    load_data_map, _ = load_gather_data(Config(args_dict), split_data_type, filter_same_trajectory=False, filter_stairs=True)
    for scan,path_list in load_data_map.items():
        path_key_list = []
        for path in path_list:
            trajectory_id = path['trajectory_id']
            episode_id = path['episode_id']
            path_key = f"{trajectory_id}_{episode_id}"
            data_map[path_key] = path
database = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
rank = 0
key = f"eval_rank_{rank}".encode()
with database.begin() as txn:
    value = txn.get(key)
    value = msgpack_numpy.unpackb(value)
    if value is None:
        print(f"value is None")
        sys.exit()
for scan,path_key_list in value.items():
    if scan != the_scan:
        continue
    else:
        path_key = path_key_list[0]
data = data_map[path_key]
print(data)

# 加载环境和机器人
sim_cfg_file = f'{project_path}/vln/configs/sample_episodes_sim_cfg.yaml'
sim_config = SimulatorConfig(sim_cfg_file)
scene_asset_path = load_scene_usd(Config(args_dict), the_scan)
sim_config.config.tasks[0].scene_asset_path = scene_asset_path
path_zero=data
start_position = np.array(path_zero["start_position"])
start_rotation = np.array(path_zero["start_rotation"])
sim_config.config.tasks[0].robots[0].position = start_position
sim_config.config.tasks[0].robots[0].orientation = start_rotation
env = BaseEnv(sim_config, headless=headless, webrtc=False)
the_task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
the_robot = the_task.robots[list(the_task.robots.keys())[0]]
the_isaac_robot = the_robot.isaac_robot
robot_pose = the_task.get_robot_poses_without_offset()
from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
from vln.src.local_nav.global_topdown_map import GlobalTopdownMap
args_dict = {
    "maps":{
        "dilation_iterations":4,
        "add_dilation":True,
        "global_topdown_config":{
            "width":500,
            "height":500,
            "aperture":200,
            "camera_transform_height":0.8,
            "voxel_size":0.1
        },
        "agent_radius":0.25,
    },
    "planners":{
        "stair_sample_step":2,
        "a_star_max_iter":5000,
    },
    "windows_head":False,
    "save_path_planning":False,
    "settings":{
        "use_llm":True,
        "max_step":25000,
        "sample_camera_list":['pano_camera_0']
    },
    "log_image_dir":"",
}
topdown_map = GlobalTopdownMap(Config(args_dict), scan, vis_verbose=False)
occupancy_map = CamOccupancyMap(Config(args_dict), the_robot.sensors['topdown_camera_500'])
gt_exe_path, shortest_path_length = get_shortest_path(robot_pose)