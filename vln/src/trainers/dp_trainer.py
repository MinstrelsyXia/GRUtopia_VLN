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
from vln.src.utils.utils import extract_best_eval_results, load_dataset
from vln.src.dataset.vlnce_dp_dataset import VLNCE_DP_Dataset, collate_fn

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
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir
        self.config = config
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
        
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
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
        
        # Init the file_logger
        if self.config.run_type == 'train':
            train_logger_filename = os.path.join(log_dir, "train.log")
            ## remove the existing logger first
            # if os.path.exists(train_logger_filename):
            #     os.remove(train_logger_filename)
            self.train_logger = MyLogger(
                name="train", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
                filename=train_logger_filename
            )
            self.train_logger.info(f"Start Training! Good Luck!!!")
            
            self.train_dataset_data = load_dataset(config.IL.dataset_root_dir, 'train', logger=self.train_logger)
        
        elif self.config.run_type == 'eval':
            eval_logger_filename = os.path.join(log_dir, f"{self.config.EVAL.SPLIT}_eval.log")
            if self.config.EVAL.start_eval_epoch != -1:
                eval_logger_filename = os.path.join(log_dir, f"{self.config.EVAL.SPLIT}_eval_{self.config.EVAL.start_eval_epoch}.log")
            # if os.path.exists(eval_logger_filename):
            #     os.remove(eval_logger_filename)
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

            self.eval_results = extract_best_eval_results(log_file=eval_logger_filename, split=self.config.EVAL.SPLIT)
            self.eval_logger.info(f"Start Eval! Good Luck!!!")
            
            self.val_seen_dataset_data = load_dataset(config.IL.dataset_root_dir, 'val_seen', logger=self.eval_logger)
            self.val_unseen_dataset_data = load_dataset(config.IL.dataset_root_dir, 'val_unseen', logger=self.eval_logger)
            
    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True, lock=False)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            if not self.config.IL.DAGGER.recollect_first:
                raise NameError("Recollect_first and lmdb_features_dir are both set to be false. Check!")
                return
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()

        # TODO
        # start_epoch = self._initialize_policy(
        #     self.config,
        #     self.config.IL.load_from_ckpt,
        #     observation_space=observation_space,
        #     action_space=action_space,
        #     load_from_pretrain=self.config.IL.load_from_pretrain
        # )
        
        is_distributed = False
        rank = 0
        world_size = 1
        start_epoch = 0
        # if self.world_size > 1:
        #     img_encoder = self.policy.module.net.image_encoder
        #     if self.local_rank != -1: # use DDP
        #         is_distributed = True
        #         rank = self.local_rank
        #         world_size = self.world_size
        # else:
        #     img_encoder = self.policy.net.image_encoder

        dataset = VLNCE_DP_Dataset(
            self.config,
            self.lmdb_features_dir,
            dataset_data=self.train_dataset_data,
            batch_size=self.config.IL.batch_size,
            bert_tokenizer=self.bert_tokenizer,
            is_distributed=is_distributed, 
            rank=rank,
            world_size=world_size,
            lmdb_save_episode_id=self.config.IL.DAGGER.lmdb_save_episode_id,
            use_stack=self.config.MODEL.IMAGE_ENCODER.use_stack
        )
        
        diter = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.IL.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,  # drop last batch if smaller
            num_workers=8,
        )

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
                total=dataset.length // dataset.batch_size,
                leave=False,
                dynamic_ncols=True,
            ):
                (
                    observations_batch,
                    prev_actions_batch,
                    not_done_masks,
                    corrected_actions_batch,
                    weights_batch,
                    episode_ids_batch,
                    gt_actions_batch
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
                loss, diffusion_loss, dist_loss, aux_loss = self._update_agent(
                    observations_batch,
                    prev_actions_batch.to(
                        device=self.device, non_blocking=True
                    ),
                    not_done_masks.to(
                        device=self.device, non_blocking=True
                    ),
                    corrected_actions_batch.to(
                        device=self.device, non_blocking=True
                    ),
                    weights_batch.to(
                        device=self.device, non_blocking=True
                    ),
                    denoise_action=self.config.IL.DAGGER.denoise_action,
                )

                if self.local_rank < 1:
                    losses.append(loss)

                    logger.info(f"train_loss: {loss}")
                    logger.info(f"train_diffusion_policy_loss: {diffusion_loss}")
                    logger.info(f"train_dist_loss: {dist_loss}")
                    logger.info(f"train_aux_loss: {aux_loss}")
                    logger.info(f"Batches processed: {step_id}.")
                    logger.info(
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
                        f"train_aux_loss_iter_{dagger_it}",
                        aux_loss,
                        step_id,
                    )
                    step_id += 1  # noqa: SIM113

                # evaluate the model
                if step_id % self.config.EVAL.train_eval_interval == 0:
                    self.policy.eval()
                    T, N = corrected_actions_batch.size()
                    
                    if self.world_size > 1:
                        net = self.policy.module
                    else:
                        net = self.policy

                    rnn_states = torch.zeros(
                        N,
                        net.net.num_recurrent_layers,
                        self.config.MODEL.STATE_ENCODER.hidden_size,
                        device=self.device,
                    ) 

                    with torch.no_grad():
                        masks = not_done_masks.to(device=self.device, non_blocking=True)
                        # original_action_mode = self.config.EVAL.ACTION 
                        # self.config.defrost()
                        # self.config.EVAL.ACTION = 'xyyaw'
                        
                        batch_settings = {
                            'mode': 'act',
                            'observations': observations_batch,
                            'rnn_states': rnn_states,
                            'prev_actions': prev_actions_batch.to(device=self.device, non_blocking=True),
                            'masks': masks,
                            'add_noise_to_action': False,
                            'denoise_action': True,
                            'num_sample': self.config.EVAL.num_sample,
                            'vis': False,
                            'step': 0,
                            'episode_ids': None,
                            'stop_mode': self.config.EVAL.stop_mode,
                            'steps': None
                        }

                        actions, rnn_states, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum, pm_pred = net(batch_settings)

                        # assert self.config.EVAL.ACTION == 'xyyaw'
                        action_cos_similarities = action_reduce(masks.squeeze(), F.cosine_similarity(
                            un_actions_nocumsum, observations_batch['actions'], dim=-1
                        )).item()
                        
                        logger.info(f"***Eval action cos similarity Iter {step_id}: {action_cos_similarities}***")
                        writer.add_scalar(
                            f"eval_action_cos_similarity_iter_{dagger_it}",
                            action_cos_similarities,
                            step_id,
                        )

                        cos_sims.append(action_cos_similarities)
                        
                        # self.config.EVAL.ACTION = original_action_mode
                        # self.config.freeze()
                    self.policy.train()
            
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
            epoch_cos_sim = sum(cos_sims) / len(cos_sims)
            if epoch_cos_sim > last_best_cossims:
                best_cossims_epoch = epoch
                last_best_cossims = epoch_cos_sim
                self.train_logger.info(
                    f"cos sim: {epoch_cos_sim:.6f}")
                self.train_logger.info(
                    f"Epoch {best_cossims_epoch} has the highest cos sim: {last_best_cossims:.6f}"
                )

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
    
    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
        denoise_action=False
    ):
        T, N = corrected_actions.size()
        masks = not_done_masks

        if self.world_size > 1:
            net = self.policy.module.net
        else:
            net = self.policy.net
            
        recurrent_hidden_states = torch.zeros(
            N,
            net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        ) 

        AuxLosses.clear()
        
        if 'rgb_features' not in observations \
            or self.config.MODEL.IMAGE_ENCODER.RGB.update_rgb_encoder \
                or self.config.MODEL.IMAGE_ENCODER.DEPTH.update_depth_encoder\
                    or self.config.MODEL.LORA.add_for_rgb_encoder \
                        or self.config.MODEL.LORA.add_for_depth_encoder:
            batch = {
                'mode': 'img_embedding',
                'rgb_inputs': observations['stack_rgb'],
                'depth_inputs': observations['stack_depth'],
                'proj': self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj
            }
            stack_rgb, stack_depth = self.policy(batch)
            if len(stack_rgb.shape) == 2:
                observations['stack_rgb'] = stack_rgb.unsqueeze(1)
                observations['stack_depth'] = stack_depth.unsqueeze(1)
            
        batch = {
            'mode': 'pred_actions',
            'observations': observations,
            'rnn_states': recurrent_hidden_states,
            'prev_actions': prev_actions,
            'masks': not_done_masks,
            'add_noise_to_action': True,
            'denoise_action': denoise_action,
        }
        if observations['stack_depth'].shape[1] == 1:
            observations['stack_depth'] = observations['stack_depth'].squeeze(1)
        if observations['stack_rgb'].shape[1] == 1:
            observations['stack_rgb'] = observations['stack_rgb'].squeeze(1)
        noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_hat = self.policy(batch)
        
        # !!!
        # draw_loss_curve(N, noise_pred, noise, output_file='test.jpg')
        
        if denoise_action:
            # for watch results
            un_actions = get_action(diffusion_output, self.action_stats).cpu().detach().numpy()
            gt_actions = get_action(batch['observations']['actions'], self.action_stats).cpu().detach().numpy()

            # draw figures
            import matplotlib.pyplot as plt
            for item_idx in range(20):
                plt.clf()
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.scatter(un_actions[item_idx][:, 0], un_actions[item_idx][:, 1], label='un_actions')
                # Annotating the points for un_actions
                for i in range(un_actions[item_idx].shape[0]):
                    plt.text(un_actions[item_idx][i, 0], un_actions[item_idx][i, 1], str(i), fontsize=9, color='blue',ha='left')

                plt.scatter(gt_actions[item_idx][:, 0],  gt_actions[item_idx][:, 1], label='gt_actions')
                # Annotating the points for gt_actions
                for i in range(gt_actions[item_idx].shape[0]):
                    plt.text(gt_actions[item_idx][i, 0], gt_actions[item_idx][i, 1], str(i), fontsize=9, color='red', ha='right')

                plt.legend()
                save_path = f'data/images/debug_{item_idx}.jpg'
                plt.savefig(save_path)
                print(f"save fig to {save_path}")

                plt.close()
            

        # for train
        dist_loss = 0
        if dist_pred is not None:
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), observations['step_distance'])
            dist_loss = (dist_loss * (masks.float())).mean() / (1e-2 +(masks.float()).mean())
        
        # L2 loss
        if self.config.MODEL.Diffusion_Policy.pred_type == 'epsilon':
            # pred noise
            diffusion_loss = action_reduce(masks.squeeze(), F.mse_loss(noise_pred, noise, reduction="none"))
        elif self.config.MODEL.Diffusion_Policy.pred_type == 'sample':
            # pred x_0
            diffusion_loss = action_reduce(masks.squeeze(), F.mse_loss(noise_pred, observations['actions'], reduction="none"))

        # Aux loss
        aux_loss = 0
        if self.config.MODEL.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_loss = F.mse_loss(
                progress_hat,
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.config.MODEL.PROGRESS_MONITOR.alpha,
            )
            aux_mask = (weights > 0).view(-1)
            aux_loss = AuxLosses.reduce(aux_mask)
        
        # Total loss
        loss = self.config.MODEL.LOSS.alpha * self.config.MODEL.LOSS.dist_scale * dist_loss + (1-self.config.MODEL.LOSS.alpha) * diffusion_loss + aux_loss
        
        loss = loss / loss_accumulation_scalar
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40.)

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # if isinstance(aux_loss, torch.Tensor):
        #     aux_loss = aux_loss.item()
        return_dist_loss = dist_loss.item() if dist_pred is not None else 0
        return loss.item(), diffusion_loss.item(), return_dist_loss, aux_loss
      
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

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
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
        writer,
        checkpoint_index: int = 0,
        split=None
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

        split = config.EVAL.SPLIT if split is None else split

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{split}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return 0, 0

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)

        epoch = self._initialize_policy(
            config,
            load_from_ckpt=True, # config.IL.load_from_ckpt
            observation_space=observation_space,
            action_space=action_space,
            load_from_pretrain=False
        )
        self.policy.eval()

        observations = envs.reset()
        start_positions = [x['globalgps'][[0,2]] for x in observations]
        start_positions = torch.from_numpy(np.stack(start_positions, axis=0)).to(self.device)
        start_yaws = [x['global_rotation'][-1] for x in observations]
        start_yaws = torch.from_numpy(np.stack(start_yaws, axis=0)).to(self.device)

        observations = extract_instruction_tokens(
            observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            bert_tokenizer=self.bert_tokenizer,
            is_clip_long=self.is_clip_long
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if self.config.MODEL.IMAGE_ENCODER.use_stack:
            batch_stack_rgb_length = [1 for _ in range(len(observations))]
            h, w, c = batch['rgb'].shape[1:]
            batch_stack_rgb = torch.zeros(len(observations), self.config.MODEL.len_traj_act, h, w, c, device=self.device)
            batch_stack_rgb[:, 0, :, :, :] = batch['rgb']

            h, w, c = batch['depth'].shape[1:]
            batch_stack_depth = torch.zeros(len(observations), self.config.MODEL.len_traj_act, h, w, c, device=self.device)
            batch_stack_depth[:, 0, :, :, :] = batch['depth']

        else:
            batch_stack_rgb, batch_stack_depth, batch_stack_rgb_length = None, None, None

        batch = extract_image_features(
            self.policy, batch, 
            img_mod=self.config.MODEL.IMAGE_ENCODER.RGB.img_mod,
            len_traj_act=self.config.MODEL.len_traj_act,
            world_size=self.world_size,
            depth_encoder_type=self.config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck,
            stack_rgb = batch_stack_rgb,
            stack_depth = batch_stack_depth,
            batch_stack_rgb_length = batch_stack_rgb_length,
            proj=self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj
            )
        
        batch_size = batch['instruction'].shape[0]

        if self.world_size > 1:
            net = self.policy.module
        else:
            net = self.policy

        rnn_states = torch.zeros(
            envs.num_envs,
            net.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, config.MODEL.len_traj_act, self.action_dim, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        # IMU
        if self.config.MODEL.IMU_ENCODER.use:
            imu = torch.zeros(envs.num_envs, 2, device=self.device)
            batch["imu"] = imu

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
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
        steps_batch = torch.from_numpy(np.array(steps)).to(self.device)
        batch["steps"] = steps_batch
        
        # init fix_length_stack
        stack_rgb_length = self.config.MODEL.len_traj_act if config.MODEL.IMAGE_ENCODER.use_stack else 1
        # stack_rgb_length = self.config.MODEL.len_traj_act
        stack_rgb = [FixedLengthStack(stack_rgb_length) for _ in range(envs.num_envs)]
        stack_depth = [FixedLengthStack(stack_rgb_length) for _ in range(envs.num_envs)]
        prev_globalgps = [FixedLengthStack(self.config.MODEL.len_traj_act+1) for _ in range(envs.num_envs)] # TODO !!! act length
        prev_globalyaw = [FixedLengthStack(self.config.MODEL.len_traj_act+1) for _ in range(envs.num_envs)]
        
        for env_idx in range(envs.num_envs):
            # record the current position before action
            stack_rgb[env_idx].push(observations[env_idx]["rgb"])
            stack_depth[env_idx].push(observations[env_idx]["depth"])
            prev_globalgps[env_idx].push(batch[env_idx]['globalgps'].detach().cpu().numpy())
            prev_globalyaw[env_idx].push(batch[env_idx]['global_rotation'][-1].detach().cpu().item())
        
        spl_dict = {}
        specific_epoch_id = str(self.config.EVAL.specific_episode_id)

        # Create the dict to record the global status of envs
        global_env_threads_paused = {}
        for env_thread in enumerate(envs._workers):
            # env_thread: (env_idx, env_thread)
            global_env_threads_paused[env_thread[1]._name] = {'env_idx':env_thread[0], 'paused':False}

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            # steps[:] = [x+1 for x in steps]
            current_episodes = envs.current_episodes()
            if int(specific_epoch_id) != -1:
                # find the specific episode!
                find_flag = False
                while not find_flag: 
                    for episode_id in range(len(current_episodes)):
                        if current_episodes[episode_id].episode_id != specific_epoch_id:
                            envs.reset_at(episode_id)
                        else:
                            find_flag = True
                            print(f"find the target episode {specific_epoch_id} at the {episode_id}-th env.")
                    current_episodes = envs.current_episodes()
                if current_episodes['epoch_id'] != specific_epoch_id:
                    print('1')
                    continue
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
                    'vis': True,
                    'step': steps[0],
                    'episode_ids': [x.episode_id for x in current_episodes],
                    'stop_mode': self.config.EVAL.stop_mode,
                    'steps': steps
                }
                
                if batch_settings['denoise_action'] and batch_settings['num_sample'] > 1:
                    batch_copy = deepcopy(batch_settings) # !!!
                    # sample multiple candidates
                    for k,v in batch_copy['observations'].items():
                        batch_copy['observations'][k] = v.repeat_interleave(batch_copy['num_sample'], dim=0)
                    batch_copy['rnn_states'] = batch_copy['rnn_states'].repeat_interleave(batch_copy['num_sample'], dim=0)
                    batch_copy['prev_actions'] = batch_copy['prev_actions'].repeat_interleave(batch['num_sample'], dim=0)
                    batch_copy['masks'] = batch_copy['masks'].repeat_interleave(batch_copy['num_sample'], dim=0)
                
                    actions, rnn_states, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum = net.act(batch_copy)
                else:
                    actions, rnn_states, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum, pm_pred = net(batch_settings)
                # print(steps[0])
            
            prev_actions = [[] for _ in range(envs.num_envs)]
            
            len_traj_act = self.config.MODEL.len_traj_act
            # len_traj_act = 1 # !!!
            stop_envs = [False] * len(actions)
            first_stop_envs = [False] * len(actions)
            outputs_dict_copy_for_stops = [[] for _ in range(len(actions))]
            info_copy_for_stops = [[] for _ in range(len(actions))]
            # global_env_paused = [(x[0],x[3]._name) for x in envs._paused] # x[0] is the env_idx, x[3]._name is the thread name like 'Thread-11'
            working_thread_names = [x._name for x in envs._workers] # record the working thread names
            for step_i in range(len_traj_act):
                tmp_a = []
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
                        if a['action'] != 'STOP':
                            a_copy = deepcopy(a)
                            if step_i == 0:
                                a_copy['action']['action_args']['actions'] = a['action']['action_args']['actions'][step_i]
                            else:
                                a_copy['action']['action_args']['actions'] = a['action']['action_args']['actions'][step_i] - a['action']['action_args']['actions'][step_i-1] # relative action
                            tmp_a.append(a_copy)
                        else:
                            if step_i != len_traj_act - 1:
                                a_copy = {'action':{
                                    'action': 'GO_TOWARD_XYYAW',
                                    'action_args': {
                                        'actions': np.zeros(self.action_dim)
                                    }
                                }}
                                tmp_a.append(a_copy)
                            else:
                                # stop at the last stack step
                                tmp_a.append(a)
                    outputs = envs.step(tmp_a)
                    # steps[:] = [x+1 for x in steps]
                # outputs = envs.step([a[step_i] for a in actions])
                # outputs_dict, _, dones, infos = [list(x) for x in zip(*outputs)]
                if len(outputs) > 0:
                    outputs_dict, _, dones, infos = [list(x) for x in zip(*outputs)]
                else:
                    outputs_dict, infos, dones = [], [], []
                for b_idx in range(len(actions)):
                    if first_stop_envs[b_idx] and not stop_envs[b_idx]:
                        pre_pop_nums = 0
                        for pre_b_idx in range(b_idx):
                            if not first_stop_envs[pre_b_idx] and stop_envs[pre_b_idx]:
                                pre_pop_nums += 1
                        # pre_pop_nums = sum(stop_envs[:b_idx])
                        cur_idx = b_idx - pre_pop_nums
                        outputs_dict_copy_for_stops[b_idx] = deepcopy(outputs_dict[cur_idx])
                        info_copy_for_stops[b_idx] = deepcopy(infos[cur_idx])
                        stop_envs[b_idx] = True
                for b_idx in range(len(actions)):
                    # 注意这里循环不能和上面的循环合并，因为insert会影响index的具体位置
                    if not first_stop_envs[b_idx] and stop_envs[b_idx]:
                        outputs_dict.insert(b_idx, outputs_dict_copy_for_stops[b_idx])
                        infos.insert(b_idx, info_copy_for_stops[b_idx])
                        dones.insert(b_idx, True)
                    first_stop_envs[b_idx] = False

                for idx in range(len(outputs_dict)):
                    stack_rgb[idx].push(outputs_dict[idx]["rgb"])
                    stack_depth[idx].push(outputs_dict[idx]["depth"])
                    
                    prev_globalgps[idx].push(outputs_dict[idx]["globalgps"])
                    prev_globalyaw[idx].push(outputs_dict[idx]["global_rotation"][-1])
                
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
                        batch_stack_rgb_length=batch_stack_rgb_length,
                        proj=self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj
                        )

                    batch["steps"] = torch.from_numpy(np.array(steps)).to(self.device)
                    
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
            
            # resume the paused env for multi_step actions
            for i in range(len(actions)):
                if stop_envs[i]:
                    envs.resume_at(i, working_thread_names[i])
            stack_rgb_for_video = deepcopy([x.get_stack() for x in stack_rgb])
            stack_depth_for_video = deepcopy([x.get_stack() for x in stack_depth])

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
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    for stack_id in range(stack_rgb_length):
                        frame = observations_to_image(observations[i], infos[i],stack_rgb=stack_rgb_for_video[i][stack_id], stack_depth=stack_depth_for_video[i][stack_id])
                        frame = append_text_to_image(
                            frame, current_episodes[i].instruction.instruction_text
                        )

                        # resize the image
                        new_width = (frame.shape[1] // 16) * 16
                        new_height = (frame.shape[0] // 16) * 16
                        resized_image = cv2.resize(frame, (new_width, new_height))
                        rgb_frames[i].append(resized_image)

                if not dones[i] and steps[i] < self.config.EVAL.MAX_STEPS:
                    # continue if not done
                    continue
                
                '''The i-th episode has done'''
                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                
                # Initialize parameters
                prev_actions[i] = torch.zeros(self.config.MODEL.len_traj_act, self.action_dim)
                rnn_states[i] = torch.zeros(
                    net.net.num_recurrent_layers,
                    config.MODEL.STATE_ENCODER.hidden_size,
                    device=self.device,
                )
                start_positions[i] = torch.from_numpy(observations[i]['globalgps'][[0,2]]).to(self.device)
                start_yaws[i] = torch.from_numpy(np.array(observations[i]['global_rotation'][-1])).to(self.device)
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

                if len(config.VIDEO_OPTION) > 0:
                    # ensure the same size in rgb_frames[i]
                    init_width, init_height = rgb_frames[i][0].shape[:2]
                    for j in range(len(rgb_frames[i])):
                        rgb_frames[i][j] = cv2.resize(rgb_frames[i][j], (init_height, init_width))
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []
                # else:
                    # print stats_episodes[ep_id]["spl"]
                    # self.eval_logger.info(
                    #     f"Episode {ep_id} SPL: {stats_episodes[ep_id]['spl']:.6f}"
                    # )
                spl_dict[ep_id] = float(stats_episodes[ep_id]["spl"])
                mean_spl = np.mean(list(spl_dict.values()))
                print('Average SPL: ', mean_spl) # !!!

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                bert_tokenizer=self.bert_tokenizer,
                is_clip_long=self.is_clip_long
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

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
                batch_stack_rgb_length=batch_stack_rgb_length,
                proj=self.config.MODEL.IMAGE_ENCODER.RGB.rgb_proj
                )

            batch["steps"] = torch.from_numpy(np.array(steps)).to(self.device)
            
            # IMU
            if self.config.MODEL.IMU_ENCODER.use:
                delta_pos = batch["globalgps"][:, [0,2]] - start_positions
                batch["imu"] = delta_pos

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
                start_positions,
                stack_rgb,
                stack_depth,
                prev_globalgps,
                prev_globalyaw,
                global_env_threads_paused,
                steps
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
                start_positions,
                stack_rgb,
                stack_depth,
                prev_globalgps,
                prev_globalyaw,
                global_env_threads_paused,
                steps
            )

        envs.close()
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
