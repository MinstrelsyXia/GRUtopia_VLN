from vln.src.v2.envs.discrete_eval import DiscreteEvalSingleScanEnv
from vln.src.v2.envs.discrete_flash_eval import DiscreteFlashEvalSingleScanEnv
from vln.src.v2.envs.continuous_sample import ContinuousSampleSingleScanEnv
from vln.src.v2.envs.discrete_flash_sample import DiscreteFlashSampleSingleScanEnv
from vln.src.v2.envs.discrete_sample import DiscreteSampleSingleScanEnv
from vln.src.v2.envs.discrete_sample_dagger import DiscreteSampleDaggerSingleScanEnv
from vln import PROJECT_ROOT_PATH
from vln.src.utils.utils import Config

def get_eval_config(
    project_path,
    ckpt_to_load,
):
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
    return Config(eval_config)

def get_env_by_config(
    config,
    sim_config,
    scene_asset_path,
    start_position,
    start_rotation,
    headless,
    dataloader,
):
    task_type = config["task_type"]
    flash = config["flash"]
    if task_type == 'eval':
        name=config["name"]
        ckpt_to_load = config["ckpt_to_load"]
        eval_config = get_eval_config(PROJECT_ROOT_PATH,ckpt_to_load)
        ckpt_file_name = ckpt_to_load.split('/')[-1]
        ckpt_name=f"{name}_{ckpt_file_name}"
        if flash:
            return DiscreteFlashEvalSingleScanEnv(
                sim_config=sim_config,
                scene_asset_path=scene_asset_path,
                start_position=start_position,
                start_rotation=start_rotation,
                headless=headless,
                dataloader=dataloader,
                eval_config=eval_config,
                lmdb_path=dataloader.lmdb_path,
                ckpt_name=ckpt_name,
            )
        else:
            return DiscreteEvalSingleScanEnv(
                sim_config=sim_config,
                scene_asset_path=scene_asset_path,
                start_position=start_position,
                start_rotation=start_rotation,
                headless=headless,
                dataloader=dataloader,
                eval_config=eval_config,
                lmdb_path=dataloader.lmdb_path,
                ckpt_name=ckpt_name,
            )
    elif task_type == 'sample':
        sample_type = config["sample_type"]
        if sample_type =='continuous':
            return ContinuousSampleSingleScanEnv(
                sim_config=sim_config,
                scene_asset_path=scene_asset_path,
                start_position=start_position,
                start_rotation=start_rotation,
                headless=headless,
                dataloader=dataloader,
            )
        else:
            if flash:
               return DiscreteFlashSampleSingleScanEnv(
                    sim_config=sim_config,
                    scene_asset_path=scene_asset_path,
                    start_position=start_position,
                    start_rotation=start_rotation,
                    headless=headless,
                    dataloader=dataloader,
               )
            else:
                dagger_percentage = 0
                if 'dagger_percentage' in config:
                    dagger_percentage = config["dagger_percentage"]
                if dagger_percentage > 0:
                    ckpt_to_load = config["ckpt_to_load"]
                    eval_config = get_eval_config(PROJECT_ROOT_PATH,ckpt_to_load)
                    return DiscreteSampleDaggerSingleScanEnv(
                        sim_config=sim_config,
                        scene_asset_path=scene_asset_path,
                        start_position=start_position,
                        start_rotation=start_rotation,
                        headless=headless,
                        dataloader=dataloader,
                        eval_config=eval_config,
                        policy_probability=dagger_percentage,
                    )
                else:
                    return DiscreteSampleSingleScanEnv(
                        sim_config=sim_config,
                        scene_asset_path=scene_asset_path,
                        start_position=start_position,
                        start_rotation=start_rotation,
                        headless=headless,
                        dataloader=dataloader,
                    )
                
