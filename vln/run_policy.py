'''
Author: w61
Date: 2014/11/05
Function: the main file to support training and evluation
'''
import os,sys
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

import argparse
import os
import random
import yaml
import logging
import shutil

import numpy as np
import torch
from typing import List, Optional, Union

import yacs.config
from yacs.config import CfgNode
torch.autograd.set_detect_anomaly(True)

from vln.src.utils.logger import MyLogger
from vln.src.trainers import dp_trainer, cma_trainer
from vln.src.utils.utils import dict_to_namespace, namespace_to_dict, Config

from vln.parser import process_args

def get_local_rank():
    return int(os.environ["LOCAL_RANK"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference", "preprocess_features"],
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
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank for distributed training",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--train_quiet",
        default=False,
        action='store_true',
    )
    
    args = parser.parse_args()
    
        
    run_exp(**vars(args))

def get_config(exp_config, opts):
    config = Config()
    config.merge_from_file(exp_config)
    if opts:
        config.merge_from_list(opts)
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
    config.local_rank = local_rank if local_rank is not None else kwargs.get('local_rank', 0)
    config.debug = kwargs.get('debug', False)
    config.train_quiet = kwargs.get('train_quiet', False)
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
        
        config.world_size = config.GPU_NUMBERS = len(config.TORCH_GPU_IDS)
        
        logdir = config.LOG_DIR
        config.LOG_FILE = os.path.join(logdir, "log.txt")
        if logdir:
            os.makedirs(logdir, exist_ok=True)
            
        logger = MyLogger(
            name="w61_grutopia", level=logging.INFO, filename=config.LOG_FILE, format_str="%(asctime)-15s %(message)s"
        )   
    
    # Move DDP setup earlier and consolidate local_rank handling
    if hasattr(config, 'DDP'):
        config.seed = config.DDP.seed
        config.fp16 = config.DDP.fp16
        config.n_workers = config.DDP.n_workers
        config.node_rank = config.DDP.node_rank
        config.world_size = len(config.TORCH_GPU_IDS)
        
        if config.DDP.use_dp and config.world_size > 1:
            assert config.IL.batch_size % len(config.TORCH_GPU_IDS) == 0
        
        if config.DDP.use and not config.DDP.use_dp:
            config.local_rank = get_local_rank()
            print(f"config.local_rank: {config.local_rank}")

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    # if torch.cuda.is_available():
    #     torch.set_num_threads(1)

    sim_config = None
    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        # Read vln_config and sim_config
        vln_config, sim_config = process_args(sim_cfg_file=config.EVAL.sim_cfg_file, vln_cfg_file=config.EVAL.vln_cfg_file)
        # Combine vln_config and sim_config with the config
        vln_config_dict = namespace_to_dict(vln_config)  # Convert Namespace to dict
        config.vln_config = Config(vln_config_dict)

        config.GT_PATH_DIR = os.path.join(config.LOG_DIR, "gt_paths")
        if not os.path.exists(config.GT_PATH_DIR):
            os.makedirs(config.GT_PATH_DIR)
        
        if config.VIDEO_OPTION != -1:
            if not os.path.exists(config.VIDEO_DIR):
                os.makedirs(config.VIDEO_DIR)
        
    if config.MODEL.policy_name in ['CMA_DP_ImgMultiPatch_Policy', 'DP_noRNN_Policy']:
        trainer_init = dp_trainer.DaggerDiffusonPolicyTrainer
    elif config.MODEL.policy_name in ['CMA_Policy', 'Seq2SeqPolicy']:
        trainer_init = cma_trainer.DaggerCMATrainer
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config, sim_config, logger)

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
    elif run_type == "preprocess_features":
        trainer._preprocess_features()
    # elif run_type == "inference":
    #     trainer.inference()
    # elif run_type == 'collect_dataset':
    #     trainer.collect_dataset() # for dagger_diffusion_policy_trainer only 


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()