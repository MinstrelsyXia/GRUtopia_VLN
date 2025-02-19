from vln.src.v2.dataloader.sample import SamplePathKeyDataloader
from vln.src.v2.envs.discrete_sample_dagger import DiscreteSampleDaggerSingleScanEnv
from vln.src.dataset.data_utils_multi_env import load_scene_usd
from vln.src.utils.utils import Config
from grutopia.core.config import SimulatorConfig
import numpy as np
from vln.src.utils.utils import Config
import sys

headless=True
target_scan="7y3sRwLe3Va"
target_trajectory = 4
rank=0
name = '20250110_dagger'
split_data_types = ['train']
robot_offset = np.array([0.   , 0.   , 1.05])
project_path = '/ssd/zhaohui/workspace/w61_grutopia_0107'
base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
args_dict = {
    "datasets":{
        "mp3d_data_dir":mp3d_data_dir,
        "base_data_dir":base_data_dir,
    }
}
scene_asset_path = load_scene_usd(Config(args_dict), target_scan)
retry_list=[]
lmdb_path = project_path + f'/data/sample_episodes/{name}'
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

eval_config={
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
        "ckpt_to_load": f"{project_path}/data/checkpoints/CMA_habitat_SOTA/converted/CMA_PM_DA_Aug_converted.pth",
        "lr_schedule":{
            "use":True,
            "type": "cosine",
            "min_lr": 1e-5,
        },
        "epochs": 60,
        "lr": 1e-4,
        "camera_name": 'pano_camera_0'
    },
    "EVAL":{
        "ACTION": 'descrete',
        "step_interval":50,
        "success_distance": 3.0,
        "SAMPLE":False,
    },
    "use_pbar":False,
}


env = DiscreteSampleDaggerSingleScanEnv(
    sim_config=sim_config,
    scene_asset_path=scene_asset_path,
    start_position=start_position,
    start_rotation=start_rotation,
    headless=headless,
    dataloader=dataloader,
    eval_config=Config(eval_config),
    policy_probability=0.2,
)

env.sample()
env.stop()
