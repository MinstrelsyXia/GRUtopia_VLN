'''
Author: w61
Date: 2014/11/05
Function: the main file to support training and evluation
'''
import os,sys
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"current_dir: {current_dir}")
sys.path.append(current_dir)

import argparse
import os
import random
import yaml
import logging
import shutil

import numpy as np
import torch

from vln.src.utils.logger import MyLogger
from vln.src.trainers import dp_trainer, cma_trainer
from vln.src.utils.utils import dict_to_namespace

from vln.parser import process_args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference, collect_dataset)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--headless",
        default=True,
        action='store_true',
    )
    parser.add_argument(
        "--test_verbose",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--show_topdown_window",
        default=False,
        action='store_true',
    )
    args = parser.parse_args()
    run_exp(**vars(args))

def get_config(exp_config, opts):
    with open(exp_config, 'r') as f:
        config = dict_to_namespace(yaml.load(f.read(), yaml.FullLoader))
    if len(opts) > 0:
        # update args into vln_config
        for key, value in vars(opts).items():
            setattr(config, key, value)
    return config

def run_exp(exp_config: str, run_type: str, opts=None, local_rank=None, **kwargs) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    config.test_verbose = kwargs.get('test_verbose', False)
    config.show_topdown_window = kwargs.get('show_topdown_window', False)
    # logger.info(f"config: {config}")
    
    # Process the log dir
    if hasattr(config, 'NAME'):
        name = config.NAME
        config.run_type = run_type
        config.TENSORBOARD_DIR = config.TENSORBOARD_DIR.replace("*name", name)
        config.CHECKPOINT_FOLDER = config.CHECKPOINT_FOLDER.replace("*name", name)
        config.EVAL_CKPT_PATH_DIR = config.EVAL_CKPT_PATH_DIR.replace("*name", name)
        config.RESULTS_DIR = config.RESULTS_DIR.replace("*name", name)
        config.LOG_DIR = config.LOG_DIR.replace("*name", name)
        config.IL.DAGGER.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.replace("*name", name)
        config.IL.DAGGER.lmdb_features_dagger_update_dir = config.IL.DAGGER.lmdb_features_dagger_update_dir.replace("*name", name)
        if hasattr(config, 'VIDEO_OPTION'):
            if config.VIDEO_OPTION != -1:
                config.VIDEO_DIR = config.VIDEO_DIR.replace("*name", name)
        
        config.local_rank = local_rank 
        config.world_size = config.GPU_NUMBERS = len(config.TORCH_GPU_IDS)
        
        logdir = config.LOG_DIR
        config.LOG_FILE = os.path.join(logdir, "log.txt")
        if logdir:
            os.makedirs(logdir, exist_ok=True)
            
        logger = MyLogger(
            name="w61_grutopia", level=logging.INFO, filename=config.LOG_FILE, format_str="%(asctime)-15s %(message)s"
        )   
    
    # DDP
    # if hasattr(config, 'DDP') and config.DDP.use:
    if hasattr(config, 'DDP'):
        config.seed = config.DDP.seed
        config.fp16 = config.DDP.fp16
        config.n_workers = config.DDP.n_workers
        config.local_rank = config.DDP.local_rank
        config.node_rank = config.DDP.node_rank
        config.world_size = config.DDP.world_size
        config.cuda_first_device = config.DDP.cuda_first_device
        
        if config.local_rank == -1 and config.world_size > 1:
            # Ensure the batch size must be divisible by the number of GPUs for DP
            assert config.IL.batch_size % len(config.TORCH_GPU_IDS) == 0

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    # if torch.cuda.is_available():
    #     torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        # Read vln_config and sim_config
        vln_config, sim_config = process_args(sim_cfg_file=config.EVAL.sim_cfg_file, vln_cfg_file=config.EVAL.vln_cfg_file)
        # Combine vln_config and sim_config with the config
        config.vln_config = vln_config
        config.sim_config = sim_config

        config.GT_PATH_DIR = os.path.join(config.LOG_DIR, "gt_paths")
        if not os.path.exists(config.GT_PATH_DIR):
            os.makedirs(config.GT_PATH_DIR)
        
        if not os.path.exists(config.VIDEO_DIR):
            os.makedirs(config.VIDEO_DIR)
        
    if config.MODEL.policy_name == 'CMA_DP_ImgMultiPatch_Policy':
        trainer_init = dp_trainer.DaggerDiffusonPolicyTrainer
    elif config.MODEL.policy_name == 'CMA_Policy':
        trainer_init = cma_trainer.DaggerCMATrainer
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config, logger)

    logger.info(f"config: {config}")
    
    # copy the config yaml file to the log dir
    if config.LOG_DIR:
        shutil.copy(exp_config, os.path.join(config.LOG_DIR, os.path.basename(exp_config)))
    
    if config.CHECKPOINT_FOLDER:
        os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
    
    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    # elif run_type == "inference":
    #     trainer.inference()
    # elif run_type == 'collect_dataset':
    #     trainer.collect_dataset() # for dagger_diffusion_policy_trainer only 


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()