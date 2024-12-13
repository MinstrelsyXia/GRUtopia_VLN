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

from vln.src.models.LongCLIP.model import longclip
from vln.src.models.utils.bert_token import BertTokenizer
from vln.src.utils.logger import MyLogger, logger
from vln.src.utils.utils import extract_best_eval_results, load_dataset, action_reduce, get_checkpoint_id, poll_checkpoint_folder, is_slurm_batch_job, batch_obs, FixedLengthStack, _compute_actions, get_delta, normalize_data, map_action_to_2d, save_video, get_action, to_local_coords
from vln.src.utils.tensorboard_utils import TensorboardWriter
from vln.src.dataset.vlnce_dp_dataset import VLNCE_DP_Dataset, collate_fn
from vln.src.models.init_policy import initialize_policy
from vln.src.envs.env import TaskEnv
from vln.src.models.utils.feature_extract import extract_image_features, extract_instruction_tokens

import logging

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

class DaggerDiffusonPolicyTrainer:
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
        if config.MODEL.TEXT_ENCODER.type == 'roberta':
            self.bert_tokenizer = BertTokenizer(
                max_length=config.MODEL.INSTRUCTION_ENCODER.max_length,
                load_model=config.MODEL.INSTRUCTION_ENCODER.load_model,
                device=self.device
            )
            self.use_bert = True
        elif config.MODEL.TEXT_ENCODER.type == 'clip-long':
            self.bert_tokenizer = longclip.tokenize
            self.use_bert = True
            self.is_clip_long = True
        
        self.batch_size = self.config.IL.batch_size
        
        if self.config.MODEL.learn_angle:
            self.action_dim = 3
        else:
            self.action_dim = 2

        torch.cuda.set_device(self.device)

        # Init the log to save the information into the file
        log_dir = self.config.LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Init the action stats
        self.action_stats = None
        if hasattr(self.config.MODEL, 'Diffusion_Policy'):
            self.action_stats = {
                'min': torch.from_numpy(np.array(self.config.MODEL.Diffusion_Policy.action_stats.min)).to(self.device),
                'max': torch.from_numpy(np.array(self.config.MODEL.Diffusion_Policy.action_stats.max)).to(self.device)
            }
        
        # use rnn or not
        self.use_rnn = 'noRNN' not in self.config.MODEL.policy_name
        
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
                
        self.policy, self.optimizer = initialize_policy(
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
            if self.world_size > 1:
                img_encoder = self.policy.module.image_encoder
                if not self.config.DDP.use_dp: # use DDP
                    is_distributed = True
                    rank = self.local_rank
                    world_size = self.world_size
            else:
                img_encoder = self.policy.image_encoder

            dataset = VLNCE_DP_Dataset(
                self.config,
                self.lmdb_features_dir,
                self.policy,
                self.device,
                dataset_data=self.train_dataset_data,
                batch_size=self.config.IL.batch_size,
                bert_tokenizer=self.bert_tokenizer,
                is_distributed=is_distributed, 
                rank=rank,
                world_size=world_size,
                lmdb_save_episode_id=self.config.IL.DAGGER.lmdb_save_episode_id,
                use_stack=self.config.MODEL.IMAGE_ENCODER.use_stack
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

                for batch in tqdm.tqdm(
                    diter,
                    total=len(diter),
                    leave=False,
                    dynamic_ncols=True,
                ):
                    if self.config.IL.analysis_time:
                        start_time = time.time()
                    (
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                    ) = batch
                    if self.config.IL.analysis_time:
                        end_time = time.time()
                        self.train_logger.info(f"Time taken for loading batch: {end_time - start_time:.2f} seconds")

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
                    loss, diffusion_loss, dist_loss, pm_loss, stop_pm_loss = self._update_agent(
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        denoise_action=self.config.IL.DAGGER.denoise_action,
                        need_instr_extraction=dataset.need_extract_instr_features
                    )

                    if self.local_rank < 1:
                        losses.append(loss)
                        if step_id % 300 == 0:
                            self.train_logger.info(f"train_loss: {loss}")
                            self.train_logger.info(f"train_diffusion_policy_loss: {diffusion_loss}")
                            self.train_logger.info(f"train_dist_loss: {dist_loss}")
                            self.train_logger.info(f"train_pm_loss: {pm_loss}")
                            self.train_logger.info(f"train_stop_pm_loss: {stop_pm_loss}")
                            self.train_logger.info(f"Batches processed: {step_id}.")
                            self.train_logger.info(
                                f"On DAgger iter {dagger_it}, Epoch {epoch}."
                            )
                        writer.add_scalar(
                            f"train_loss_iter_{dagger_it}", loss, step_id
                        )
                        writer.add_scalar(
                            f"train_diffusion_policy_loss_iter_{dagger_it}",
                            diffusion_loss,
                            step_id,
                        )
                        writer.add_scalar(
                            f"train_dist_loss_iter_{dagger_it}",
                            dist_loss,
                            step_id,
                        )
                        writer.add_scalar(
                            f"train_pm_loss_iter_{dagger_it}",
                            pm_loss,
                            step_id,
                        )
                        writer.add_scalar(
                            f"train_stop_pm_loss_iter_{dagger_it}",
                            stop_pm_loss,
                            step_id,
                        )
                        step_id += 1  # noqa: SIM113
                        
                        # save the ckpt according to the steps
                        if not self.use_rnn and step_id % self.config.IL.save_interval_steps == 0:
                            self.save_checkpoint(
                                f"ckpt-epoch-{epoch}-step-{step_id}.pth",
                                filter_frozen_weights=self.config.IL.save_filter_frozen_weights
                            )
                
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
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
        denoise_action=False,
        need_instr_extraction=True
    ):
        # T, N = prev_actions.size()
        N = self.config.IL.batch_size
        masks = not_done_masks

        if self.world_size > 1:
            net = self.policy.module
        else:
            net = self.policy
        
        if self.use_rnn:
            recurrent_hidden_states = torch.zeros(
                N,
                net.num_recurrent_layers,
                self.config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            ) 
        else:
            recurrent_hidden_states = None
        
        if 'rgb_features' not in observations \
            or self.config.MODEL.IMAGE_ENCODER.RGB.update_rgb_encoder \
                or self.config.MODEL.IMAGE_ENCODER.DEPTH.update_depth_encoder\
                    or self.config.MODEL.LORA.add_for_rgb_encoder \
                        or self.config.MODEL.LORA.add_for_depth_encoder:
            need_img_extraction = True
        else:
            need_img_extraction = False
        
        if self.config.MODEL.IMAGE_ENCODER.DEPTH.update_depth_encoder:
            need_img_extraction = True
        
        depth_return_x_before_fc = True if self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck == 'resnet' else False
        
        if self.config.IL.analysis_time:
            start_time = time.time()
        batch = {
            'mode': 'pred_actions',
            'observations': observations,
            'rnn_states': recurrent_hidden_states,
            'prev_actions': prev_actions,
            'masks': not_done_masks,
            'add_noise_to_action': True,
            'denoise_action': denoise_action,
            'need_img_extraction': need_img_extraction, # need img embedding in model
            'depth_return_x_before_fc': depth_return_x_before_fc,
            'img_mod': self.config.MODEL.IMAGE_ENCODER.RGB.img_mod,
            'proj': self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
            'process_images': False, # has processed in dataLoader
            'train_cls_free_guidance': self.config.MODEL.Diffusion_Policy.use_cls_free_guidance,
            'sample_cls_free_guidance': False,
            'need_txt_extraction': need_instr_extraction
        }
        # if observations['stack_depth'].shape[1] == 1:
        #     observations['stack_depth'] = observations['stack_depth'].squeeze(1)
        # if observations['stack_rgb'].shape[1] == 1:
        #     observations['stack_rgb'] = observations['stack_rgb'].squeeze(1)
        noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_hat, denoise_action_list, stop_progress_pred = self.policy(batch)
        if self.config.IL.analysis_time:
            end_time = time.time()
            self.train_logger.info(f"Time taken for pred_actions: {end_time - start_time:.2f} seconds")
        
        # !!!
        # draw_loss_curve(N, noise_pred, noise, output_file='test.jpg')
        if denoise_action:
            # for watch results
            un_actions = get_action(diffusion_output, self.action_stats).cpu().detach().numpy()
            gt_actions = get_action(batch['observations']['actions'], self.action_stats).cpu().detach().numpy()
            self.save_predicted_actions(un_actions, gt_actions)

        # for train
        dist_loss = 0
        if dist_pred is not None:
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), observations['step_distance'])
            dist_loss = (dist_loss * (masks.float())).mean() / (1e-2 +(masks.float()).mean())
        
        # L2 loss
        if self.config.MODEL.Diffusion_Policy.pred_type == 'epsilon':
            # pred noise
            diffusion_loss = action_reduce(masks, F.mse_loss(noise_pred, noise, reduction="none"))
        elif self.config.MODEL.Diffusion_Policy.pred_type == 'sample':
            # pred x_0
            diffusion_loss = action_reduce(masks, F.mse_loss(noise_pred, observations['actions'], reduction="none"))

        # Aux loss
        pm_loss = 0
        if self.config.MODEL.PROGRESS_MONITOR.use:
            progress_loss = F.mse_loss(
                progress_hat.squeeze(),
                observations["progress"],
                reduction="none",
            )
            pm_loss = action_reduce(masks, progress_loss)
        
        stop_pm_loss = 0
        if self.config.MODEL.STOP_PROGRESS_PREDICTOR.use:
            stop_pm_loss = F.mse_loss(
                stop_progress_pred.squeeze(),
                observations["stop_progress"],
                reduction="none",
            )
            stop_pm_loss = action_reduce(masks, stop_pm_loss)
        
        # Total loss
        loss = self.config.MODEL.LOSS.alpha * self.config.MODEL.LOSS.dist_scale * dist_loss + (1-self.config.MODEL.LOSS.alpha) * diffusion_loss + pm_loss + stop_pm_loss
        
        loss = loss / loss_accumulation_scalar
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40.)

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # if isinstance(aux_loss, torch.Tensor):
        #     aux_loss = aux_loss.item()
        return_dist_loss = dist_loss.item() if dist_pred is not None else 0
        return loss.item(), diffusion_loss.item(), return_dist_loss, pm_loss.item(), stop_pm_loss.item()
      
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

        self.rotation_threshold = float(self.config.EVAL.rotation_threshold)

        # if "tensorboard" in self.config.VIDEO_OPTION:
        #     assert (
        #         len(self.config.TENSORBOARD_DIR) > 0
        #     ), "Must specify a tensorboard directory for video display"
        #     os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        # if "disk" in self.config.VIDEO_OPTION:
        #     assert (
        #         len(self.config.VIDEO_DIR) > 0
        #     ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=30
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(
                    self.config.EVAL_CKPT_PATH_DIR
                )
                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self.eval_logger.info(f'====Eval Ckpt File:{self.config.EVAL_CKPT_PATH_DIR}====')
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                save_eval_mode = 'best_spl_sr'
                prev_ckpt_ind = -1
                first_find_start_epoch = True
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        # print(current_ckpt)
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind,
                            start_eval_epoch=self.config.EVAL.start_eval_epoch,
                            first_find_start_epoch=first_find_start_epoch
                        )
                        if current_ckpt == -1:
                            print('Waiting ckpts...')
                            current_ckpt = None
                            time.sleep(2)  # sleep for 2 secs before polling again
                            continue
                        if current_ckpt is not None and len(current_ckpt) == 2:
                            # assign the start_eval_epoch and return the new prev_ckpt_ind
                            current_ckpt, prev_ckpt_ind = current_ckpt
                            first_find_start_epoch = False
                        time.sleep(2)  # sleep for 2 secs before polling again
                    prev_ckpt_ind += 1
                    ckpt_file_ind = current_ckpt.split('/')[-1].split('.')[1]
                    if use_gt:
                        self.gt_eval(
                            checkpoint_path=current_ckpt,
                            writer=writer,
                            checkpoint_index=ckpt_file_ind,
                        )
                    else:
                        spl, sr = self._eval_checkpoint(
                            checkpoint_path=current_ckpt,
                            writer=writer,
                            checkpoint_index=ckpt_file_ind,
                        )
                        if spl == 0 and sr == 0:
                            # skip the ckpt if it has 0 SPL and 0 SR
                            continue
                        
                        self.eval_logger.info(f"=======Eval Current_ckpt: {current_ckpt}=======")
                        # update results
                        if not os.path.exists(os.path.join(self.config.EVAL_CKPT_PATH_DIR, 'ckpts')):
                            os.makedirs(os.path.join(self.config.EVAL_CKPT_PATH_DIR, 'ckpts'))
                        if spl > self.eval_results['best_spl']:
                            self.eval_results['best_spl'] = spl
                            self.eval_results['best_spl_index'] = ckpt_file_ind
                            # copy the current ckpt file to best_spl
                            # shutil.copy(current_ckpt, os.path.join(self.config.EVAL_CKPT_PATH_DIR, 'ckpts', f'{self.config.EVAL.SPLIT}_best_spl.pth'))
                        if sr > self.eval_results['best_sr']:
                            self.eval_results['best_sr'] = sr
                            self.eval_results['best_sr_index'] = ckpt_file_ind
                            # copy the current ckpt file to best_sr
                            # shutil.copy(current_ckpt, os.path.join(self.config.EVAL_CKPT_PATH_DIR, 'ckpts', f'{self.config.EVAL.SPLIT}_best_sr.pth'))
                        if spl+sr > self.eval_results['best_spl_sr']:
                            self.eval_results['best_spl_sr'] = spl+sr
                            self.eval_results['best_spl_sr_index'] = ckpt_file_ind
                            # copy the current ckpt file to best_spl_sr
                            shutil.copy(current_ckpt, os.path.join(self.config.EVAL_CKPT_PATH_DIR, 'ckpts', f'{self.config.EVAL.SPLIT}_best_spl_sr.pth'))
                            self.eval_results['best_spl_sr_spl'] = spl
                            self.eval_results['best_spl_sr_sr'] = sr
                        
                        if self.config.EVAL.auto_remove:
                            if ckpt_file_ind != self.eval_results['best_spl_index'] and ckpt_file_ind != self.eval_results['best_sr_index']:
                                # remove the ckpt_file
                                os.remove(current_ckpt)
                        
                        # update the eval log
                        self.eval_logger.info(f"Current {self.config.EVAL.SPLIT} SPL: {spl:.4f} at index {ckpt_file_ind}")
                        self.eval_logger.info(f"Current {self.config.EVAL.SPLIT} SR: {sr:.4f} at index {ckpt_file_ind}")
                        self.eval_logger.info(f"Current {self.config.EVAL.SPLIT} SPL and SR: {spl+sr:.4f} at index {ckpt_file_ind}")
                        
                        self.eval_logger.info(f"Best {self.config.EVAL.SPLIT} SPL: {self.eval_results['best_spl']:.4f} at index {self.eval_results['best_spl_index']}")
                        self.eval_logger.info(f"Best {self.config.EVAL.SPLIT} SR: {self.eval_results['best_sr']:.4f} at index {self.eval_results['best_sr_index']}")
                        self.eval_logger.info(f"Best {self.config.EVAL.SPLIT} SPL and SR: {self.eval_results['best_spl_sr']:.4f} at index {self.eval_results['best_spl_sr_index']} . SPL: {self.eval_results['best_spl_sr_spl']:.4f} , SR: {self.eval_results['best_spl_sr_sr']:.4f}")

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer=None,
        checkpoint_index: int = 0,
        split=None,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        self.eval_logger.info(f"checkpoint_path: {checkpoint_path}")
        config = self.config

        if self.config.EVAL.USE_CKPT_CONFIG:
            # TODO
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

        # split = config.EVAL.SPLIT if split is None else split
        if split is None:
            if isinstance(config.EVAL.SPLIT, list):
                split = config.EVAL.SPLIT[0]
            else:
                split = config.EVAL.SPLIT

        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if config.VIDEO_OPTION != -1:
            # TODO
            # config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            total_rgb_list = []
            total_topdown_rgb_list = []

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.split_names}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return 0, 0

        '''Init the task env'''
        obs = self.eval_env.construct_env(init_omni_env=True, result_json_path=self.result_json_path)
        if isinstance(obs, str):
            if obs == 'shortest_path_planning_failed':
                while isinstance(obs, str) and obs == 'shortest_path_planning_failed':
                    obs = self.eval_env.construct_env(init_omni_env=True, result_json_path=self.result_json_path)
            elif obs == 'all_data_evaluated':
                self.eval_logger.info(f"All data in {self.eval_env.current_split} and {split} have been evaluated.")
                return 0, 0

        '''Init the policy'''
        self.policy, _ = initialize_policy(
            self.config,
            self.eval_logger,
            load_from_ckpt=True, # config.IL.load_from_ckpt
            device=self.device,
            load_from_pretrain=self.config.IL.load_from_pretrain,
            action_stats=self.action_stats
        )
        self.policy.eval()

        observations = self.eval_env.get_obs()
        start_positions = [x['globalgps'][[0,1]] for x in observations]
        start_positions = torch.from_numpy(np.stack(start_positions, axis=0)).to(self.device)
        start_yaws = [x['globalyaw'] for x in observations]
        start_yaws = torch.from_numpy(np.stack(start_yaws, axis=0)).to(self.device)

        observations = extract_instruction_tokens(
            observations, 
            bert_tokenizer=self.bert_tokenizer,
            is_clip_long=self.is_clip_long
        )
        batch = batch_obs(observations, self.device)

        if self.config.MODEL.IMAGE_ENCODER.use_stack:
            batch_stack_rgb_length = [1 for _ in range(len(observations))]
            h, w, c = batch['rgb'].shape[1:]
            batch_stack_rgb = torch.zeros(len(observations), self.config.MODEL.IMAGE_ENCODER.img_stack_nums, h, w, c, device=self.device)
            batch_stack_rgb[:, 0, :, :, :] = batch['rgb']

            h, w, c = batch['depth'].shape[1:]
            batch_stack_depth = torch.zeros(len(observations), self.config.MODEL.IMAGE_ENCODER.img_stack_nums, h, w, c, device=self.device)
            batch_stack_depth[:, 0, :, :, :] = batch['depth']

        else:
            batch_stack_rgb, batch_stack_depth, batch_stack_rgb_length = None, None, None

        classifier_free_mask_depth = self.config.MODEL.Diffusion_Policy.use_cls_free_guidance and self.config.MODEL.IMAGE_ENCODER.DEPTH.update_depth_encoder

        batch = extract_image_features(
            self.policy, batch, 
            img_mod=self.config.MODEL.IMAGE_ENCODER.RGB.img_mod,
            len_traj_act=self.config.MODEL.IMAGE_ENCODER.img_stack_nums,
            world_size=self.world_size,
            depth_encoder_type=self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck,
            stack_rgb = batch_stack_rgb,
            stack_depth = batch_stack_depth,
            batch_stack_rgb_length = batch_stack_rgb_length,
            proj=self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
            need_rgb_extraction=True,
            classifier_free_mask_depth=classifier_free_mask_depth,
            )
        
        batch_size = batch['instruction'].shape[0]

        if self.world_size > 1:
            net = self.policy.module
        else:
            net = self.policy

        if self.use_rnn:
            rnn_states = torch.zeros(
                self.eval_env.env_nums,
                net.num_recurrent_layers,
                config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
        else:
            rnn_states = None
        prev_actions = torch.zeros(
            self.eval_env.env_nums, config.MODEL.len_traj_act, self.action_dim, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.eval_env.env_nums, 1, dtype=torch.uint8, device=self.device
        )

        # IMU
        if self.config.MODEL.IMU_ENCODER.use:
            imu = torch.zeros(self.eval_env.env_nums, self.config.MODEL.IMU_ENCODER.input_size, device=self.device)
            batch["imu"] = imu.float()

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
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        steps = [0] * batch_size
        sim_steps = [0] * batch_size
        steps_batch = torch.from_numpy(np.array(steps)).to(self.device)
        batch["steps"] = steps_batch
        
        # init fix_length_stack
        stack_rgb_length = self.config.MODEL.IMAGE_ENCODER.img_stack_nums if config.MODEL.IMAGE_ENCODER.use_stack else 1
        # stack_rgb_length = self.config.MODEL.len_traj_act
        stack_rgb = [FixedLengthStack(stack_rgb_length) for _ in range(self.eval_env.env_num)]
        stack_depth = [FixedLengthStack(stack_rgb_length) for _ in range(self.eval_env.env_nums)]
        prev_globalgps = [FixedLengthStack(self.config.MODEL.len_traj_act+1) for _ in range(self.eval_env.env_nums)] # TODO !!! act length
        prev_globalyaw = [FixedLengthStack(self.config.MODEL.len_traj_act+1) for _ in range(self.eval_env.env_nums)]
        
        for env_idx in range(self.eval_env.env_nums):
            # record the current position before action
            stack_rgb[env_idx].push(observations[env_idx]["rgb"])
            stack_depth[env_idx].push(observations[env_idx]["depth"])
            prev_globalgps[env_idx].push(batch[env_idx]['globalgps'].detach().cpu().numpy())
            prev_globalyaw[env_idx].push(batch[env_idx]['global_rotation'][-1].detach().cpu().item())

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

            with torch.no_grad():
                batch_settings = {
                    'mode': 'act',
                    'observations': batch,
                    'rnn_states': rnn_states,
                    'prev_actions': prev_actions,
                    'masks': not_done_masks,
                    'add_noise_to_action': False,
                    'denoise_action': True,
                    'num_sample': self.config.EVAL.num_sample,
                    # 'vis': self.config.test_verbose,
                    'vis': False,
                    'step': sim_steps[0],
                    'episode_ids': current_episodes['episode_id'],
                    'stop_mode': self.config.EVAL.stop_mode,
                    'steps': steps,
                    'predicted_actions_save_dir': self.eval_env.EP_DIR,
                    'num_sample': self.config.EVAL.num_sample,
                    'train_cls_free_guidance': False,
                    'sample_cls_free_guidance': self.config.MODEL.Diffusion_Policy.use_cls_free_guidance,
                    'need_txt_extraction': True,
                }
                
                actions, rnn_states, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum, pm_pred, stop_progress_pred = net(batch_settings)
                # print(steps[0])
            
            prev_actions = [[] for _ in range(self.eval_env.env_nums)]
            
            # len_traj_act = self.config.MODEL.len_traj_act
            len_traj_act = self.config.EVAL.len_traj_act
            # len_traj_act = 1 # !!!
            stop_envs = [False] * len(actions)
            first_stop_envs = [False] * len(actions)

            # for step_i in range(len_traj_act):
            #     tmp_a = []
            if self.config.EVAL.ACTION == 'descrete':
                for bs_i, action in enumerate(actions):
                    a = action[step_i]
                    if a != action_spaces['stop']:
                        if a == action_spaces['wait']:
                            a = [{'action':{
                                'action': 'GO_TOWARD_XYYAW',
                                'action_args': {
                                    'actions': np.zeros(self.action_dim)
                                }
                            }}]
                        tmp_a.append(a)
                    else:
                        if not stop_envs[bs_i]:
                            first_stop_envs[bs_i] = True
                        tmp_a.append(a)

                exe_list = []
                for a_idx, a in enumerate(tmp_a):
                    if stop_envs[a_idx]:
                        continue
                    exe_list.append(a[0])
                    steps[a_idx] += 1

                # outputs = envs.step([a[0] for a in tmp_a])
                outputs = envs.step(exe_list)
                for b_idx in range(len(actions)):
                    if first_stop_envs[b_idx] and not stop_envs[b_idx]:
                        envs.pause_at(b_idx, working_thread_names=working_thread_names, global_env_threads_paused=global_env_threads_paused)
                        
            elif self.config.EVAL.ACTION == 'xyyaw':
                for a_i, a in enumerate(actions):
                    if isinstance(a, str) and a == 'STOP':
                        action = [
                            {'h1': {'stop': ['stop']}}
                        ]
                    else:
                        target_poses, target_quats, exe_actions = self.eval_env.predicted_action_to_global(a, step_i=-1, verbose=self.config.test_verbose, len_traj_act=len_traj_act)
                        
                        # TODO: check if all nodes are valid on the map?
                        action = []
                        if len(exe_actions) > 1:
                            action = [
                                {'h1': {'move_along_path': [exe_actions[:-1]]}}
                            ]
                        rot_action = [
                            {'h1': {'rotate': [exe_actions[-1]]}}
                        ]
                        outputs = self.eval_env.step(action, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list, rot_action)
                        steps[0] += len(exe_actions)
                        total_actions.append(exe_actions)

            elif self.config.EVAL.ACTION == 'speed':
                for a_i, a in enumerate(actions):
                    if isinstance(a[0], str) and a[0] == 'STOP':
                        action = [
                            {'h1': {'stop': ['stop']}}
                        ]
                    else:
                        if self.config.test_verbose:
                            target_poses, target_quats, exe_actions = self.eval_env.predicted_action_to_global(a, step_i=-1, verbose=self.config.test_verbose) # for debug. drawing the predicted actions
                        speed_actions = self.eval_env.get_speed_actions(a, len_traj_act=len_traj_act,verbose=self.config.test_verbose)
                        exe_action = speed_actions
                        action = [
                            {'h1': {'move_along_speeds': [exe_action]}}
                        ]
                        # if len(exe_action) == 0:
                        #     action = [
                        #         {'h1': {'stop': ['stop']}}
                        #     ]

                outputs = self.eval_env.step(action, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list, verbose=self.config.test_verbose) 
                steps[0] += len(speed_actions)
                total_actions.append(speed_actions)

            outputs_dict = outputs['outputs_dict']
            dones = outputs['dones']
            infos = outputs['infos']
            sim_steps = outputs['current_step_list']
            stack_rgb = outputs['stack_rgb']
            stack_depth = outputs['stack_depth']
            prev_globalgps = outputs['prev_globalgps']
            prev_globalyaw = outputs['prev_globalyaw']
            total_rgb_list = outputs['total_rgb_list']
            total_topdown_rgb_list = outputs['total_topdown_rgb_list']
      
            if self.use_rnn:
                # update RNN states
                ## Update prev_actions
                if len_traj_act > 1:
                    for idx in range(len(actions)):
                        # reverse to make the latest frame to be 0 position
                        prev_globalgps_numpy = np.array(prev_globalgps[idx].get_stack(reverse=True))
                        prev_globalyaw_numpy = np.array(prev_globalyaw[idx].get_stack(reverse=True))
                        prev_act = _compute_actions( 
                            prev_globalgps_numpy, prev_globalyaw_numpy,
                            curr_time=0, fill_mode="constant",
                            len_traj_pred=self.config.MODEL.len_traj_act,
                            waypoint_spacing=self.config.MODEL.Diffusion_Policy.waypoint_spacing,
                            learn_angle=self.config.MODEL.learn_angle,
                            metric_waypoint_spacing=self.config.MODEL.Diffusion_Policy.metric_waypoint_spacing,
                            num_action_params=self.action_dim,
                            normalize=False)
                        prev_act_delta = torch.from_numpy(get_delta(prev_act)).to(self.device)
                        prev_act_delta_norm = normalize_data(prev_act_delta, self.action_stats)
                        prev_actions[idx] = prev_act_delta_norm
                    
                    ## Update image features in batch
                    # if self.config.MODEL.IMAGE_ENCODER.use_stack:
                    batch_stack_rgb, batch_stack_depth, batch_stack_rgb_length = [], [], []
                    for env_idx in range(len(stack_rgb)):
                        cur_rgb = np.array(stack_rgb[env_idx].get_stack(reverse=True))
                        cur_depth = np.array(stack_depth[env_idx].get_stack(reverse=True))
                        batch_stack_rgb_length.append(len(cur_rgb))
                        if len(cur_rgb) < stack_rgb_length:
                            cur_rgb = np.concatenate([cur_rgb, np.zeros((stack_rgb_length-len(cur_rgb), *cur_rgb.shape[1:]))], axis=0)
                            cur_depth = np.concatenate([cur_depth, np.zeros((stack_rgb_length-len(cur_depth), *cur_depth.shape[1:]))], axis=0)
                        batch_stack_rgb.append(cur_rgb)
                        batch_stack_depth.append(cur_depth) 
                    batch_stack_rgb = torch.from_numpy(np.array(batch_stack_rgb).astype(np.uint8)).to(self.device)
                    batch_stack_depth = torch.from_numpy(np.array(batch_stack_depth)).to(self.device)

                    # else:
                    #     batch_stack_rgb, batch_stack_depth = None, None

                    # if self.config.MODEL.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling' and not self.config.MODEL.IMAGE_ENCODER.use_stack:
                    if not self.config.MODEL.IMAGE_ENCODER.use_stack:
                        batch['rgb'] = batch_stack_rgb.squeeze(1)
                        batch['depth'] = batch_stack_depth.squeeze(1)
                        batch_stack_rgb, batch_stack_depth = None, None  

                    batch = extract_image_features(
                        self.policy, batch, 
                        img_mod=self.config.MODEL.IMAGE_ENCODER.RGB.img_mod,
                        len_traj_act=self.config.MODEL.len_traj_act,
                        world_size=self.world_size,
                        depth_encoder_type=self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck,
                        stack_rgb = batch_stack_rgb,
                        stack_depth = batch_stack_depth,
                        batch_stack_rgb_length = batch_stack_rgb_length,
                        proj=self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
                        need_rgb_extraction=True,
                        classifier_free_mask_depth=classifier_free_mask_depth,
                        )

                    batch["steps"] = torch.from_numpy(np.array(steps)).to(self.device)
                    
                    if self.use_rnn:
                        with torch.no_grad():
                            prev_actions_batch = torch.stack(prev_actions, axis=0).to(self.device)
                            batch_settings = {
                                'mode': 'update_rnn',
                                'observations': batch,
                                'rnn_states': rnn_states,
                                'prev_actions': prev_actions_batch,
                                'masks': not_done_masks,
                            }
                            
                            _, update_rnn_states= net(batch_settings)
                            rnn_states = update_rnn_states

            for idx in range(len(actions)):
                # reverse to make the latest frame to be 0 position
                prev_globalgps_numpy = np.array(prev_globalgps[idx].get_stack(reverse=True))
                prev_globalyaw_numpy = np.array(prev_globalyaw[idx].get_stack(reverse=True))
                prev_act = _compute_actions( 
                    prev_globalgps_numpy, prev_globalyaw_numpy,
                    curr_time=0, fill_mode="constant",
                    len_traj_pred=self.config.MODEL.len_traj_act,
                    waypoint_spacing=self.config.MODEL.Diffusion_Policy.waypoint_spacing,
                    learn_angle=self.config.MODEL.learn_angle,
                    metric_waypoint_spacing=self.config.MODEL.Diffusion_Policy.metric_waypoint_spacing,
                    num_action_params=self.action_dim,
                    normalize=False)
                action_deltas = get_delta(prev_act)
                if self.config.MODEL.learn_angle: 
                    # [x,y,yaw]
                    prev_act_delta = torch.from_numpy(action_deltas).to(self.device)
                    prev_act_delta_norm = normalize_data(prev_act_delta, self.action_stats)
                    prev_actions[idx] = prev_act_delta_norm
                else:
                    # [forward, rotation]
                    prev_act_delta = torch.from_numpy(map_action_to_2d(action_deltas)).to(self.device)
                    prev_actions[idx] = prev_act_delta
            
            # convert list to tensor
            current_stack_rgb_list = [x.get_stack(reverse=True) for x in stack_rgb]
            current_stack_depth_list = [x.get_stack(reverse=True) for x in stack_depth]
            
            stack_rgb_tensor = torch.from_numpy(np.stack(current_stack_rgb_list, axis=1)).to(self.device)
            stack_depth_tensor = torch.from_numpy(np.stack(current_stack_depth_list, axis=1)).to(self.device)
            prev_actions = torch.stack(prev_actions, axis=0).to(self.device)

            # outputs_dict, _, dones, infos = [list(x) for x in zip(*outputs)]

            observations = outputs_dict

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # reset envs and observations if necessary
            for i in range(self.eval_env.env_nums):
                # if config.VIDEO_OPTION != -1:
                    # for stack_id in range(stack_rgb_length):
                    #     frame = observations_to_image(observations[i], infos[i],stack_rgb=stack_rgb_for_video[i][stack_id], stack_depth=stack_depth_for_video[i][stack_id])
                    #     frame = append_text_to_image(
                    #         frame, current_episodes[i].instruction.instruction_text
                    #     )

                    #     # resize the image
                    #     new_width = (frame.shape[1] // 16) * 16
                    #     new_height = (frame.shape[0] // 16) * 16
                    #     resized_image = cv2.resize(frame, (new_width, new_height))
                    #     rgb_frames[i].append(resized_image)

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
                # self.eval_env.draw_visited_map() # TODO: save the visited map
                
                ep_id = current_episodes['episode_id']
                stats_episodes[ep_id] = infos[i]

                ep_time = time.time() - current_episode_start_time
                self.eval_logger.info(f"Episode {ep_id} time: {ep_time//60:.0f}min {ep_time%60:.0f}s")

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
                    save_video(config.VIDEO_DIR, total_rgb_list, split, ep_id, checkpoint_index, stats_episodes[ep_id]["spl"])
                    save_video(config.VIDEO_DIR, total_topdown_rgb_list, split, ep_id, checkpoint_index, stats_episodes[ep_id]["spl"], is_topdown=True)
                    total_rgb_list = []
                    total_topdown_rgb_list = []

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

                current_episode_start_time = time.time()
                
                # Initialize parameters
                prev_actions[i] = torch.zeros(self.config.MODEL.len_traj_act, self.action_dim)
                if self.use_rnn:
                    rnn_states[i] = torch.zeros(
                        net.num_recurrent_layers,
                        config.MODEL.STATE_ENCODER.hidden_size,
                        device=self.device,
                    )
                start_positions[i] = torch.from_numpy(observations[i]['globalgps'][[0,1]]).to(self.device)
                start_yaws[i] = torch.from_numpy(np.array(observations[i]['globalyaw'])).to(self.device)
                steps[i] = 0
                stack_rgb[i] = FixedLengthStack(stack_rgb_length)
                stack_depth[i] = FixedLengthStack(stack_rgb_length)
                prev_globalgps[i] = FixedLengthStack(self.config.MODEL.len_traj_act+1) # TODO !!! act length
                prev_globalyaw[i] = FixedLengthStack(self.config.MODEL.len_traj_act+1)

                stack_rgb[i].push(observations[i]['rgb'])
                stack_depth[i].push(observations[i]['depth'])
                prev_globalgps[i].push(observations[i]['globalgps'])
                prev_globalyaw[i].push(observations[i]['global_rotation'][-1])

                dones[i] = False
                not_done_masks[i] = 1

                total_actions = []

            observations = extract_instruction_tokens(
                observations, 
                bert_tokenizer=self.bert_tokenizer,
                is_clip_long=self.is_clip_long
            )
            batch = batch_obs(observations, self.device)

            ## Update image features in batch
            # if self.config.MODEL.IMAGE_ENCODER.use_stack:
            batch_stack_rgb, batch_stack_depth, batch_stack_rgb_length = [], [], []
            for env_idx in range(len(stack_rgb)):
                cur_rgb = np.array(stack_rgb[env_idx].get_stack(reverse=True))
                cur_depth = np.array(stack_depth[env_idx].get_stack(reverse=True))
                batch_stack_rgb_length.append(len(cur_rgb))
                if len(cur_rgb) < stack_rgb_length:
                    cur_rgb = np.concatenate([cur_rgb, np.zeros((stack_rgb_length-len(cur_rgb), *cur_rgb.shape[1:]))], axis=0)
                    cur_depth = np.concatenate([cur_depth, np.zeros((stack_rgb_length-len(cur_depth), *cur_depth.shape[1:]))], axis=0)
                batch_stack_rgb.append(cur_rgb)
                batch_stack_depth.append(cur_depth) 
            batch_stack_rgb = torch.from_numpy(np.array(batch_stack_rgb).astype(np.uint8)).to(self.device)
            batch_stack_depth = torch.from_numpy(np.array(batch_stack_depth)).to(self.device)

            # else:
            #     batch_stack_rgb, batch_stack_depth = None, None

            # if self.config.MODEL.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling' and not self.config.MODEL.IMAGE_ENCODER.use_stack:
            if not self.config.MODEL.IMAGE_ENCODER.use_stack:
                batch['rgb'] = batch_stack_rgb.squeeze(1)
                batch['depth'] = batch_stack_depth.squeeze(1)
                batch_stack_rgb, batch_stack_depth = None, None 

            batch = extract_image_features(
                self.policy, batch, 
                img_mod=self.config.MODEL.IMAGE_ENCODER.RGB.img_mod,
                len_traj_act=self.config.MODEL.IMAGE_ENCODER.img_stack_nums,
                world_size=self.world_size,
                depth_encoder_type=self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck,
                stack_rgb = batch_stack_rgb,
                stack_depth = batch_stack_depth,
                batch_stack_rgb_length=batch_stack_rgb_length,
                proj=self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
                need_rgb_extraction=True,
                classifier_free_mask_depth=classifier_free_mask_depth,
                )

            batch["steps"] = torch.from_numpy(np.array(steps)).to(self.device)
            
            # IMU
            if self.config.MODEL.IMU_ENCODER.use:
                # initialize_imu
                batch["imu"] = torch.zeros(batch["globalgps"].shape[0], self.config.MODEL.IMU_ENCODER.input_size).to(self.device)
                if self.config.MODEL.IMU_ENCODER.to_local_coords:
                    batch["imu"][:, :2] = to_local_coords(batch["globalgps"][:, [0,1]].float(), start_positions, start_yaws)
                else:
                    batch["imu"][:, :2] = batch["globalgps"][:, [0,1]] - start_positions
                if self.config.MODEL.IMU_ENCODER.input_size == 3:
                    batch["imu"][:, 2] = batch["globalyaw"] - start_yaws

        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)
                        # Record detailed results
            if self.config.EVAL.save_details:
                detailed_fname = fname.replace('stats_ckpt', 'stats_ckpt_detailed')
                sorted_stats = {k: stats_episodes[k] for k in sorted(stats_episodes.keys(), key=int)}
                with open(detailed_fname, "w") as f:
                    json.dump(sorted_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = int(checkpoint_index) + 1
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)

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
    
    def _preprocess_features(self):
        from vln.src.trainers.preprocess_features import FeaturePreprocessor
        
        '''Init the model and load the pretrained weights'''
        self.policy, _ = initialize_policy(
            self.config,
            self.train_logger,
            self.config.IL.load_from_ckpt,
            self.device,
            load_from_pretrain=self.config.IL.load_from_pretrain,
            action_stats=self.action_stats
        )
        
        feature_preprocessor = FeaturePreprocessor(
            model=self.policy,
            config=self.config,
            train_dataset=self.train_dataset_data,
            bert_tokenizer=self.bert_tokenizer,
            input_lmdb_dir=self.config.IL.DAGGER.lmdb_features_dir,
            output_lmdb_dir=self.config.IL.DAGGER.lmdb_features_dagger_update_dir,
            device=self.device,
            del_original_rgb=True
        )

        feature_preprocessor.preprocess_features()

def plot_spl_list(spl_list):
    # Create a figure and axis
    plt.figure(figsize=(10, 5))

    # Plot the spl_list as a curve
    plt.plot(spl_list, marker='o', linestyle='-', color='b', label='SPL Curve')

    # Adding titles and labels
    plt.title('SPL List Curve')
    plt.xlabel('Index')
    plt.ylabel('SPL Value')
    
    # Adding a grid
    plt.grid()

    # Show legend
    plt.legend()

    # Display the plot
    plt.savefig('spl_curve.jpg')

def longest_zero_interval(spl_list):
    max_length = 0  # Length of the longest interval found
    max_start = -1  # Starting index of the longest interval
    max_end = -1    # Ending index of the longest interval

    current_start = -1  # Starting index of the current zero interval
    current_length = 0   # Length of the current zero interval

    for index, value in enumerate(spl_list):
        if value == 0:
            # If we encounter a zero and it's the first zero of a new interval
            if current_length == 0:
                current_start = index  # Mark the start of the new interval
            current_length += 1  # Increase the length of the current interval
        else:
            # If we encounter a non-zero value
            if current_length > max_length:
                # Check if the current interval is the longest
                max_length = current_length
                max_start = current_start
                max_end = index - 1  # End index of the interval
            # Reset the current interval
            current_length = 0

    # Final check for the last interval in case the list ends with zeros
    if current_length > max_length:
        max_length = current_length
        max_start = current_start
        max_end = len(spl_list) - 1  # End index of the interval

    return (max_start, max_end)
