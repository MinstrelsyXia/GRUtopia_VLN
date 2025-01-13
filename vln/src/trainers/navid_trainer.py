import gc
import os
import random
import warnings
import cv2
from collections import defaultdict

import lmdb
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import tqdm
import time
from copy import deepcopy

import json
import re


from vln.src.models.LongCLIP.model import longclip
from vln.src.models.utils.bert_token import BertTokenizer
from vln.src.utils.logger import MyLogger, logger
from vln.src.utils.utils import extract_best_eval_results, load_dataset, action_reduce, aux_reduce, get_checkpoint_id, poll_checkpoint_folder, is_slurm_batch_job, batch_obs, FixedLengthStack, _compute_actions, get_delta, normalize_data, map_action_to_2d, save_video, get_action, to_local_coords
from vln.src.utils.tensorboard_utils import TensorboardWriter
from vln.src.dataset.vlnce_cma_dataset import CMADataset, collate_fn
from vln.src.models.init_policy import initialize_policy
from vln.src.envs.env import TaskEnv
from vln.src.models.utils.feature_extract import extract_image_features, extract_instruction_tokens

import logging

from vln.src.models.navid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vln.src.models.navid.conversation import conv_templates, SeparatorStyle
from vln.src.models.navid.model.builder import load_pretrained_model
from vln.src.models.navid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

action_spaces = {
    'stop': [0],
    'go_forward': [1],
    'turn_left': [2],
    'turn_right': [3],
    'wait': [4]
}

def draw_loss_curve(N, noise_pred, noise, output_file='test.jpg'):
    losses = []
    step = N
    for i in range(0, len(noise_pred), step):
        loss = F.mse_loss(noise_pred[i:i+step], noise[i:i+step])
        losses.append(loss.item())
    # Plot the losses
    plt.clf()
    plt.plot(range(0, len(noise_pred), step), losses, marker='o')
    plt.xlabel('Range Start Index')
    plt.ylabel('MSE Loss')
    plt.title('Loss for Each Range Every 6 Steps')
    plt.grid(True)
    plt.savefig(output_file)
    print(f"save fig to {output_file}")

class NavidTrainer:
    def __init__(self, config=None, sim_config=None, logger=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir
        self.config = config
        self.logger = logger
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.is_distributed = self.world_size > 1 and (not self.config.DDP.use_dp)
        
        if self.is_distributed:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cuda", config.TORCH_GPU_IDS[0])
        
        self.use_bert = False
        self.bert_tokenizer = None
        self.is_clip_long = False
        
        self.batch_size = self.config.IL.batch_size
    
        self.action_dim = 4

        torch.cuda.set_device(self.device)

        # Init the log to save the information into the file
        log_dir = self.config.LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Init the action stats
        self.action_stats = None

        self.show_tqdm = not self.config.train_quiet

        # Init the file_logger
        if self.config.run_type in ['train', 'preprocess_features']:
            train_logger_filename = os.path.join(log_dir, "train.log")
            ## remove the existing logger first
            # if os.path.exists(train_logger_filename):
            #     os.remove(train_logger_filename)
            self.train_logger = MyLogger(
                name="train", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
                filename=train_logger_filename
            )
            if self.config.run_type == 'train':
                self.train_logger.info(f"Start Training! Good Luck!!!")
            elif self.config.run_type == 'preprocess_features':
                self.train_logger.info(f"Start Preprocessing Features! Good Luck!!!")

            self.train_dataset_data = load_dataset(config.IL.dataset_root_dir, 'train', logger=self.train_logger)
        
        elif self.config.run_type == 'eval':
            if isinstance(self.config.EVAL.SPLIT, list):
                if len(self.config.EVAL.SPLIT) > 1:
                    self.split_names = f"{self.config.EVAL.SPLIT[0]}_{self.config.EVAL.SPLIT[1]}"
                else:
                    self.split_names = f"{self.config.EVAL.SPLIT[0]}"
            else:
                self.split_names = self.config.EVAL.SPLIT
            eval_logger_filename = os.path.join(log_dir, f"{self.split_names}_eval.log")
            if self.config.EVAL.start_eval_epoch != -1:
                eval_logger_filename = os.path.join(log_dir, f"{self.config.EVAL.SPLIT}_eval_{self.config.EVAL.start_eval_epoch}.log")
            if os.path.exists(eval_logger_filename):
                os.remove(eval_logger_filename)
            self.eval_logger = MyLogger(
                name="eval", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
                filename=eval_logger_filename
            )
            # Read the previous best eval results
            if os.path.exists(eval_logger_filename):
                with open(eval_logger_filename, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        last_line = lines[-1]
            
            # create result jsons
            self.result_json_path = os.path.join(log_dir, f"{self.split_names}_results.json")
            if not os.path.exists(self.result_json_path) or self.config.EVAL.re_eval:
                with open(self.result_json_path, 'w') as f:
                    json.dump(defaultdict(list), f)
                self.eval_logger.info(f"Create new result json: {self.result_json_path}")

            self.eval_results = extract_best_eval_results(log_file=eval_logger_filename, split=self.config.EVAL.SPLIT)
            self.eval_logger.info(f"Start Eval! Good Luck!!!")
            
            # self.val_seen_dataset_data = load_dataset(config.IL.dataset_root_dir, 'val_seen', logger=self.eval_logger)
            # self.val_unseen_dataset_data = load_dataset(config.IL.dataset_root_dir, 'val_unseen', logger=self.eval_logger)
            
            self.splits = self.config.EVAL.SPLIT
            
            '''Init the eval env'''
            self.eval_env = TaskEnv(self.config, sim_config, self.splits, self.eval_logger, filter_same_trajectory=False, policy_eval=True)
            
    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        """Main method for training DAgger."""
        dagger_it = 0 # TODO: Dagger
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True, lock=False)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        # else:
        #     if not self.config.IL.DAGGER.recollect_first:
        #         raise NameError("Recollect_first and lmdb_features_dir are both set to be false. Check!")
        #         return
        #     with lmdb.open(
        #         self.lmdb_features_dir,
        #         map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        #     ) as lmdb_env, lmdb_env.begin(write=True) as txn:
        #         txn.drop(lmdb_env.open_db())

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()
               
        self.policy, self.optimizer, self.lr_scheduler, start_epoch = initialize_policy(
            self.config,
            self.train_logger,
            self.config.IL.load_from_ckpt,
            self.device,
            load_from_pretrain=self.config.IL.load_from_pretrain,
            action_stats=self.action_stats
        )
        
        is_distributed = self.is_distributed
        rank = self.local_rank if self.is_distributed else 0
        world_size = self.world_size
        start_epoch = 0
        
        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=30, purge_step=0) as writer:        
            if self.config.DDP.use:
                if not self.config.DDP.use_dp: # use DDP
                    is_distributed = True
                    rank = self.local_rank
                    world_size = self.world_size

            dataset = CMADataset(
                self.config,
                self.lmdb_features_dir,
                self.config.IL.use_iw,
                dataset_data=self.train_dataset_data,
                inflection_weight_coef=self.config.IL.inflection_weight_coef,
                lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                batch_size=self.config.IL.batch_size,
                is_distributed=is_distributed, 
                rank=rank,
                world_size=world_size,
            )
            
            num_workers = 4 if not self.config.debug else 0
            diter = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.IL.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,  # drop last batch if smaller
                num_workers=num_workers,
            )

            step_id = 0
            last_least_loss = 9999
            last_best_cossims = -1
            least_loss_epoch = 0
            best_cossims_epoch = 0
            for epoch in tqdm.trange(
                start_epoch, self.config.IL.epochs, dynamic_ncols=True
            ):
                losses = []
                cos_sims= []

                if self.show_tqdm:
                    batch_iterator = tqdm.tqdm(
                        diter,
                        total=len(diter),
                        leave=False,
                        dynamic_ncols=True,
                    )
                else:
                    batch_iterator = diter

                for batch in batch_iterator:
                    (
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch,
                        weights_batch,
                    ) = batch

                    observations_batch = {
                        k: v.to(
                            device=self.device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                        for k, v in observations_batch.items()
                    }
                    
                    if step_id % 100 == 0:
                        torch.cuda.empty_cache()
                    
                    prev_actions_batch = prev_actions_batch.to(device=self.device, non_blocking=True)
                    not_done_masks = not_done_masks.to(device=self.device, non_blocking=True) if not_done_masks is not None else None  
                    loss, pm_loss = self._update_agent(
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch.to(
                            device=self.device, non_blocking=True
                        ),
                        weights_batch.to(
                            device=self.device, non_blocking=True
                        ),
                    )

                    if self.local_rank < 1:
                        losses.append(loss)
                        if step_id % 300 == 0:
                            self.train_logger.info(f"train_loss: {loss}")
                            self.train_logger.info(f"train_pm_loss: {pm_loss}")
                            self.train_logger.info(f"Batches processed: {step_id}.")
                            self.train_logger.info(
                                f"On DAgger iter {dagger_it}, Epoch {epoch}."
                            )
                        writer.add_scalar(
                            f"train_loss_iter_{dagger_it}", loss, step_id
                        )
                        writer.add_scalar(
                            f"train_pm_loss_iter_{dagger_it}", pm_loss, step_id
                        )
                        step_id += 1  # noqa: SIM113
                    
                
                # save the log
                self.train_logger.info(f"*******Epoch {epoch}*********")
                epoch_loss = sum(losses) / len(losses)
                if epoch_loss < last_least_loss:
                    least_loss_epoch = epoch
                    last_least_loss = epoch_loss
                self.train_logger.info(
                    f"loss: {epoch_loss:.6f}"
                )
                self.train_logger.info(
                    f"Epoch {least_loss_epoch} has the least loss: {last_least_loss:.6f}"
                )     
                # epoch_cos_sim = sum(cos_sims) / len(cos_sims)
                # if epoch_cos_sim > last_best_cossims:
                #     best_cossims_epoch = epoch
                #     last_best_cossims = epoch_cos_sim
                #     self.train_logger.info(
                #         f"cos sim: {epoch_cos_sim:.6f}")
                #     self.train_logger.info(
                #         f"Epoch {best_cossims_epoch} has the highest cos sim: {last_best_cossims:.6f}"
                #     )

                if self.local_rank < 1 and epoch % self.config.IL.save_interval_epochs==0:
                    self.save_checkpoint(
                        f"ckpt.{dagger_it * self.config.IL.epochs + epoch}.pth",
                        filter_frozen_weights=self.config.IL.save_filter_frozen_weights
                    )
            
    def save_checkpoint(self, file_name: str, filter_frozen_weights=False) -> None:
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """
        # Check if the policy is wrapped in nn.DataParallel and get the correct state_dict
        if isinstance(self.policy, torch.nn.DataParallel):
            state_dict = self.policy.module.state_dict()  # Access the module state_dict
        else:
            state_dict = self.policy.state_dict()  # Regular model state_dict
        
        if filter_frozen_weights:
            # filter the frozen weights
            trainable_params = {name: param for name, param in self.policy.named_parameters() if param.requires_grad}
            state_dict = trainable_params

        checkpoint = {
            "state_dict": state_dict,
            "config": self.config,
        }
            
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )
    
    def save_predicted_actions(self, un_actions, gt_actions):
        for item_idx in range(20):
            plt.clf()
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            
            # Plot predicted actions with arrows
            plt.scatter(un_actions[item_idx][:, 0], un_actions[item_idx][:, 1], label='un_actions', color='blue', alpha=0.5)
            for i in range(un_actions[item_idx].shape[0]):
                # Calculate arrow direction components using yaw angle
                arrow_length = 0.2  # Adjust this value to change arrow length
                dx = arrow_length * np.cos(un_actions[item_idx][i, 2])
                dy = arrow_length * np.sin(un_actions[item_idx][i, 2])
                
                # Draw arrow
                plt.arrow(un_actions[item_idx][i, 0], 
                        un_actions[item_idx][i, 1], 
                        dx, dy, 
                        head_width=0.05, 
                        head_length=0.1, 
                        fc='blue', 
                        ec='blue',
                        alpha=0.5)
                
                # Add point index
                plt.text(un_actions[item_idx][i, 0], un_actions[item_idx][i, 1], 
                        str(i), fontsize=9, color='blue', ha='left')

            # Plot ground truth actions with arrows
            plt.scatter(gt_actions[item_idx][:, 0], gt_actions[item_idx][:, 1], label='gt_actions', color='red', alpha=0.5)
            for i in range(gt_actions[item_idx].shape[0]):
                # Calculate arrow direction components using yaw angle
                arrow_length = 0.2  # Adjust this value to change arrow length
                dx = arrow_length * np.cos(gt_actions[item_idx][i, 2])
                dy = arrow_length * np.sin(gt_actions[item_idx][i, 2])
                
                # Draw arrow
                plt.arrow(gt_actions[item_idx][i, 0], 
                        gt_actions[item_idx][i, 1], 
                        dx, dy, 
                        head_width=0.05, 
                        head_length=0.1, 
                        fc='red', 
                        ec='red',
                        alpha=0.5)
                
                # Add point index
                plt.text(gt_actions[item_idx][i, 0], gt_actions[item_idx][i, 1], 
                        str(i), fontsize=9, color='red', ha='right')

            plt.legend()
            plt.grid(True)
            plt.axis('equal')  # Make sure the aspect ratio is equal
            
            save_path = f'data/images/debug_{item_idx}.jpg'
            plt.savefig(save_path)
            print(f"save fig to {save_path}")

            plt.close()
    
    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        if self.world_size > 1:
            net = self.policy.module
        else:
            net = self.policy
            
        recurrent_hidden_states = torch.zeros(
            N,
            net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        ) 
        
        batch = {
            'mode': 'train',
            'observations': observations,
            'rnn_states': recurrent_hidden_states,
            'prev_actions': prev_actions,
            'masks': not_done_masks
        }
        logits, rnn_states_out = self.policy(batch)

        # for train
        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = aux_reduce(aux_mask, action_loss)

        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item(), aux_loss.item()
      
    def eval(self, use_gt=False) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_IDS[0])
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        spl, sr = self._eval_checkpoint()
        
        # update the eval log
        self.eval_logger.info(f"Current {self.config.EVAL.SPLIT} SPL: {spl:.4f}")
        self.eval_logger.info(f"Current {self.config.EVAL.SPLIT} SR: {sr:.4f}")

    def _eval_checkpoint(
        self, split=None
    ) -> None:
        """Evaluates a single checkpoint.
        """
        config = self.config

        # split = config.EVAL.SPLIT if split is None else split
        if split is None:
            if isinstance(config.EVAL.SPLIT, list):
                split = config.EVAL.SPLIT[0]
            else:
                split = config.EVAL.SPLIT

        config.use_pbar = not is_slurm_batch_job()

        total_rgb_list = []
        total_topdown_rgb_list = []
        
        '''Init the policy'''
        self.conv_mode = "vicuna_v1"
        self.model_name = get_model_name_from_path(self.config.EVAL.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.config.EVAL.model_path, None, get_model_name_from_path(self.config.EVAL.model_path))
        
        self.eval_logger.info("Initialization Complete")
        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."
        
        self.rgb_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.reset()
        
        '''Init the task env'''
        obs = self.eval_env.construct_env(init_omni_env=True, result_json_path=self.result_json_path)
        if isinstance(obs, str):
            if obs == 'shortest_path_planning_failed':
                while isinstance(obs, str) and obs == 'shortest_path_planning_failed':
                    obs = self.eval_env.construct_env(init_omni_env=False, result_json_path=self.result_json_path)
            elif obs == 'all_data_evaluated':
                self.eval_logger.info(f"All data in {self.eval_env.current_split} and {split} have been evaluated.")
                return 0, 0

        '''Get the observations'''
        observations = self.eval_env.get_obs()
        batch = observations[0]

        batch_size = 1

        stats_episodes = {}

        rgb_frames = [[] for _ in range(self.eval_env.env_nums)]
        if config.VIDEO_OPTION != -1:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(self.eval_env.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        pbar_iter = 0
        log_str = (
            f"[Ckpt: NAVID]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        steps = [0] * batch_size
        sim_steps = [0] * batch_size

        for env_idx in range(self.eval_env.env_nums):
            if config.VIDEO_OPTION != -1:
                total_rgb_list.append(observations[env_idx]["rgb"])
                total_topdown_rgb_list.append(observations[env_idx]["topdown_rgb"])
        
        spl_dict = {}
        total_actions = []
        current_episode_start_time = time.time()

        while len(stats_episodes) < num_eps:
            # steps[:] = [x+1 for x in steps]
            current_episodes = self.eval_env.data_item
            if len(spl_dict) > 0 and np.mean(list(spl_dict.values())) < 0.02 and len(stats_episodes) > 100:
                # this ckpt is too bad to continue
                self.eval_logger.info(f"Break. This ckpt is too bad to continue with average SPL {np.mean(list(spl_dict.values())):.3f}")
                break
            
            action = self.act(observations[0]) # only one env
            actions = [action]

            if self.config.EVAL.ACTION == 'descrete':
                for bs_i, a in enumerate(actions):
                    cur_a = a['action']
                    if cur_a == 0:
                        action = [
                            {'h1': {'stop': ['stop']}}
                        ]
                    else:
                        action = [
                            {'h1': {'move_by_descrete': [cur_a]}}
                        ] 
                        
            outputs = self.eval_env.step(action,total_rgb_list=total_rgb_list, total_topdown_rgb_list=total_topdown_rgb_list,verbose=self.config.test_verbose)
                        
            outputs_dict = outputs['outputs_dict']
            dones = outputs['dones']
            infos = outputs['infos']
            sim_steps = outputs['current_step_list']
            total_rgb_list = outputs['total_rgb_list']
            total_topdown_rgb_list = outputs['total_topdown_rgb_list']
            steps[self.eval_env.env_idx] += 1

            observations = outputs_dict

            # reset envs and observations if necessary
            for i in range(self.eval_env.env_nums):
                if not dones[i] and steps[i] < self.config.EVAL.MAX_STEPS:
                    # continue if not done
                    continue
                
                '''The i-th episode has done'''
                self.eval_logger.info(f"*******Episode {current_episodes['episode_id']} has done")
                # Log episode metrics
                for metric_name, metric_value in infos[i].items():
                    if metric_name == "fail_reason":
                        self.eval_logger.info(f"{metric_name}: {metric_value}")
                    else:
                        self.eval_logger.info(f"{metric_name}: {metric_value:.3f}")
                
                self.update_result_json(self.result_json_path, infos[i])
                # self.eval_env.draw_visited_map()
                
                ep_id = current_episodes['episode_id']
                stats_episodes[ep_id] = infos[i]

                ep_time = time.time() - current_episode_start_time
                self.eval_logger.info(f"Episode {ep_id} time: {ep_time:.2f}s")

                observations[i] = self.eval_env.construct_env(step_time=steps[i])[0]
                if isinstance(observations[i], str):
                    if observations[i] == 'shortest_path_planning_failed':
                        while isinstance(observations[i], str) and observations[i] == 'shortest_path_planning_failed':
                            observations[i] = self.eval_env.construct_env(init_omni_env=True, result_json_path=self.result_json_path)
                    elif observations[i] == 'all_data_evaluated':
                        self.eval_logger.info(f"All data in {self.eval_env.current_split} and {self.eval_env.current_split} have been evaluated.")
                        break

                current_episode_start_time = time.time()
                self.reset()
                
                # Initialize parameters
                steps[i] = 0
                dones[i] = False

                total_actions = []

                if config.use_pbar:
                    pbar.update()
                    pbar_iter += 1
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if config.VIDEO_OPTION != -1:
                    # ensure the same size in rgb_frames[i]
                    init_width, init_height = total_rgb_list[0].shape[:2]
                    for j in range(len(total_rgb_list)):
                        total_rgb_list[j] = cv2.resize(total_rgb_list[j], (init_height, init_width))
                    
                    init_width, init_height = total_topdown_rgb_list[0].shape[:2]
                    for j in range(len(total_topdown_rgb_list)):
                        total_topdown_rgb_list[j] = cv2.resize(total_topdown_rgb_list[j], (init_height, init_width))
                    # save rgbs as videos
                    save_video(config.VIDEO_DIR, total_rgb_list, split, ep_id, 0, stats_episodes[ep_id]["spl"])
                    save_video(config.VIDEO_DIR, total_topdown_rgb_list, split, ep_id, 0, stats_episodes[ep_id]["spl"], is_topdown=True)
                    # generate_video(
                    #     video_option=config.VIDEO_OPTION,
                    #     video_dir=config.VIDEO_DIR,
                    #     images=rgb_frames[i],
                    #     episode_id=ep_id,
                    #     checkpoint_idx=checkpoint_index,
                    #     metrics={"spl": stats_episodes[ep_id]["spl"]},
                    #     tb_writer=writer,
                    # )
                    # del stats_episodes[ep_id]["top_down_map_vlnce"]
                    total_rgb_list = []
                    total_topdown_rgb_list = []
                # else:
                    # print stats_episodes[ep_id]["spl"]
                    # self.eval_logger.info(
                    #     f"Episode {ep_id} SPL: {stats_episodes[ep_id]['spl']:.6f}"
                    # )
                spl_dict[ep_id] = float(stats_episodes[ep_id]["spl"])
                mean_spl = np.mean(list(spl_dict.values()))
                self.eval_logger.info(f"Average SPL: {mean_spl}") # !!!

                # construct the next environment
                observations[i] = self.eval_env.construct_env(step_time=steps[i], result_json_path=self.result_json_path)[0]
                if isinstance(observations[i], str):
                    if observations[i] == 'shortest_path_planning_failed':
                        while isinstance(observations[i], str) and observations[i] == 'shortest_path_planning_failed':
                            observations[i] = self.eval_env.construct_env(init_omni_env=True, result_json_path=self.result_json_path)
                    elif observations[i] == 'all_data_evaluated':
                        self.eval_logger.info(f"All data in {self.eval_env.current_split} and {self.eval_env.current_split} have been evaluated.")
                        break

                total_actions = []


        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        self.eval_logger.info(f"Episodes evaluated: {num_episodes}")
        for k, v in aggregated_stats.items():
            self.eval_logger.info(f"{k}: {v:.6f}")

        return aggregated_stats['spl'], aggregated_stats['success']
    
    def update_result_json(self, result_json_path, episode_info):
        with open(result_json_path, 'r') as f:
            data = json.load(f)
        if self.eval_env.current_split not in data:
            data[self.eval_env.current_split] = {}
            data[self.eval_env.current_split]["finished_scans"] = []
            data[self.eval_env.current_split]["episodes"] = defaultdict(list)
        if self.eval_env.current_scan not in data[self.eval_env.current_split]["episodes"]:
            data[self.eval_env.current_split]["episodes"][self.eval_env.current_scan] = []
        data[self.eval_env.current_split]["episodes"][self.eval_env.current_scan].append(episode_info)
        with open(result_json_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    #### Copy from navid
    def reset(self):
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.first_forward = False

    def act(self, observations):
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            return {"action": temp_action}

        navigation_qs = self.promt_template.format(observations["instruction"])
        navigation = self.predict_inference(navigation_qs)
        if self.config.test_verbose:
            self.eval_logger.info(f"Navigation Output: {navigation}")
        
        action_index, num = self.extract_result(navigation[:-1])

        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1:
            for _ in range(min(3, int(num/25))):
                self.pending_action_list.append(1)

        elif action_index == 2:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(2)

        elif action_index == 3:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(3)
        
        if action_index is None or len(self.pending_action_list)==0:
            self.pending_action_list.append(random.randint(1, 3))
            # Primarily unused, intended to complete the pipeline logic.

        return {"action": self.pending_action_list.pop(0)}

    def process_images(self, rgb_list):
        batch_image = np.asarray(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        return [video]


    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)


        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs



    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)

        return None, None



    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]



        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line


        if line:

            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)


        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image