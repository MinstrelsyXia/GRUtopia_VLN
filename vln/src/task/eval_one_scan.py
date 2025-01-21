import argparse
import time
import os
import threading
import sys
from vln.src.v2.envs.discrete_eval import DiscreteEvalSingleScanEnv
from vln.src.v2.dataloader.eval import EvalPathKeyDataloader
from grutopia.core.config import SimulatorConfig
from vln.src.dataset.data_utils_multi_env import load_scene_usd
from vln.src.utils.utils import Config
import numpy as np
import sys
from grutopia.core.util.log import log
import json
from vln import PROJECT_ROOT_PATH

def check_process_stuck(env:DiscreteEvalSingleScanEnv):
    index = 0
    while True:
        index+=1
        current_time = time.time()
        duration = round(current_time - env.timestamp,2)
        if  duration > 300:
            print("5分钟时间戳未更新,杀死进程")
            os.kill(os.getpid(), 9) 
        else:
            if index % 60 == 0:
                print(f"check_process_stuck 存活[{env.timestamp}]")
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
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="cfg_file",
    )
    args = parser.parse_args()
    rank = args.rank
    scan = args.scan
    cfg_file = args.cfg_file
    project_path = PROJECT_ROOT_PATH
    cfg_file_path = f"{project_path}/{cfg_file}"
    if not os.path.exists(cfg_file_path):
        print(f"{cfg_file_path} not exist")
        sys.exit()
    with open(cfg_file_path, 'r') as file:
        config = json.load(file)
    headless = True
    split_data_types = ['val_unseen','val_seen']
    base_data_dir = f'{project_path}/data/datasets/R2R_VLNCE_v1-3_corrected'
    mp3d_data_dir = f"{project_path}/../Matterport3D/data/v1/scans"
    robot_offset = np.array([0.   , 0.   , 1.05])
    name = config["name"]
    ckpt_file_name = config["ckpt_to_load"].split('/')[-1]
    ckpt_name=f"{name}_{ckpt_file_name}"
    lmdb_path = project_path + f'/data/sample_episodes/{name}'
    retry_list = config["retry_list"]
    sim_cfg_file = f'{project_path}/vln/configs/sim_cfg_policy_eval.yaml'
    ckpt_to_load = config["ckpt_to_load"]
    sim_config = SimulatorConfig(sim_cfg_file)
    args_dict = {
        "datasets":{
            "mp3d_data_dir":mp3d_data_dir,
            "base_data_dir":base_data_dir,
        }
    }
    scene_asset_path = load_scene_usd(Config(args_dict), scan)
    dataloader=EvalPathKeyDataloader(
        base_data_dir,
        split_data_types,
        robot_offset,
        rank,
        ckpt_name,
        lmdb_path,
        scan,
        retry_list,
    )
    eval_path_key_list = dataloader.eval_path_key_list
    if len(eval_path_key_list) == 0:
        log.info(f"[scan:{scan}] has no data to eval")
        os.kill(os.getpid(), 9) 
    path_zero = dataloader.path_key_data[eval_path_key_list[0]]
    start_position = path_zero['start_position']
    start_rotation = path_zero['start_rotation']

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
            "ckpt_to_load": ckpt_to_load,
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

    env = DiscreteEvalSingleScanEnv(
        sim_config,
        scene_asset_path,
        start_position,
        start_rotation,
        headless,
        dataloader,
        Config(eval_config),
        lmdb_path,
        ckpt_name,
    )
    
    monitor_thread = threading.Thread(target=check_process_stuck, args=(env,))
    monitor_thread.start()

    try:
        print("env.eval()")
        env.eval()
        os.kill(os.getpid(), 9) 
    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        monitor_thread.join()
        print("Program terminated.")