import os, sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from transformers import PretrainedConfig
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from vln.src.models.encoders.my_diffusers.diffusers import myDDPMScheduler
import matplotlib.pyplot as plt

from copy import deepcopy

from torch import Tensor

from vln.src.models.diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vln.src.models.diffusion_policy.diffusion_policy.model.diffusion.transformer_for_diffusion_modified import TransformerForDiffusion 

import vln.src.models.encoders as encoders

from vln.src.models.utils.utils import get_delta, get_action, get_data_stats, normalize_data, unnormalize_data, action_reduce

action_spaces = {
    'stop': [0],
    'go_forward': [1],
    'turn_left': [2],
    'turn_right': [3],
    'wait': [4]
}

class CMA_DP_Net(nn.Module):
    def __init__(
        self, config, observation_space, action_stats=None
    ) -> None:
        super().__init__()
        self.config = config
        self.model_config = config.MODEL
        if self.model_config.learn_angle:
            self.num_actions = 3
        else:
            self.num_actions = 2
        self.action_stats = action_stats
        
        self.model_config.TEXT_ENCODER.final_state_only = False
        self.use_stack = self.model_config.IMAGE_ENCODER.use_stack
        self.stack_num = self.model_config.IMAGE_ENCODER.img_stack_nums
        self.patch_num = self.model_config.IMAGE_ENCODER.RGB.multi_patches_num
        # Note that I use TEXT_ENCODER to represent the instruction encoder rather than the original INSTRUCTION_ENCODER
        
        if self.model_config.TEXT_ENCODER.model_name == 'clip-long':
            self.instruction_encoder = encoders.InstructionLongCLIPEncoder(self.model_config.TEXT_ENCODER, self.model_config.LORA)
        else:
            if self.model_config.TEXT_ENCODER.model_name in ['meter', 'roberta']:
                config_name = 'roberta-base'
            else:
                config_name = self.model_config.TEXT_ENCODER.model_name
            bert_config = PretrainedConfig.from_pretrained(config_name)
            # Init the instruction encoder
            text_encoder_config = copy.deepcopy(bert_config)
            for k,v in self.model_config.TEXT_ENCODER.items():
                setattr(text_encoder_config, k, v)
            # add LORA settings
            setattr(text_encoder_config, 'LORA', self.model_config.LORA)

            self.instruction_encoder = encoders.LanguageEncoder(text_encoder_config)
        
        # Init the RGB & depth encoder
        self.image_encoder = encoders.ImageEncoder(self.model_config, self.model_config.IMAGE_ENCODER, observation_space, self.model_config.LORA, analysis_time=self.config.IL.analysis_time)
        
        # Init the cross-modal fusion network
        # try:
        #     bert_config = PretrainedConfig.from_pretrained('roberta-base')
        # except Exception as e:
        bert_config = PretrainedConfig.from_pretrained('data/pretrained/roberta')
        cross_modal_config = copy.deepcopy(bert_config)
        for k,v in self.model_config.CROSS_MODAL_ENCODER.items():
            setattr(cross_modal_config, k, v)

        # self.cross_modal_encoder = encoders.VisionLanguageEncoder(cross_modal_config)
        # self.his_txt_cross_encoder = encoders.VisionLanguageEncoder(cross_modal_config)
        if self.model_config.CROSS_MODAL_ENCODER.txt_to_img:
            txt_to_img_cross_encoder_config = copy.deepcopy(cross_modal_config)
            txt_to_img_cross_encoder_config.num_x_layers = self.model_config.CROSS_MODAL_ENCODER.txt_to_img_layer
            self.txt_img_cross_encoder = encoders.VisionLanguageEncoder(txt_to_img_cross_encoder_config)
        self.img_txt_cross_encoder = encoders.VisionLanguageEncoder(cross_modal_config)
        
        # Init the prev action embedding
        if self.model_config.IMAGE_ENCODER.use_stack:
            prev_action_encoder_size = self.model_config.IMAGE_ENCODER.RGB.feature_dim
        else:
            prev_action_encoder_size = self.model_config.PREV_ACTION_ENCODER.encoding_size
        self.prev_action_embedding = nn.Linear(self.num_actions, prev_action_encoder_size)
        self.prev_action_embedding_dp = nn.Linear(self.num_actions, self.model_config.STATE_ENCODER.hidden_size)
        self.prev_act_ln = nn.LayerNorm(self.model_config.PREV_ACTION_ENCODER.encoding_size)
        self.prev_action_pos_embedding = encoders.PositionalEncoding(self.model_config.PREV_ACTION_ENCODER.encoding_size, self.model_config.len_traj_act)
        
        # Init the step embedding
        if self.model_config.STEP_ENCODER.use:
            self.step_embeddings = nn.Embedding(self.model_config.STEP_ENCODER.max_steps, self.model_config.STEP_ENCODER.encoding_size)
            self.step_embeddings_dp = nn.Embedding(self.model_config.STEP_ENCODER.max_steps, self.model_config.STATE_ENCODER.hidden_size)

        # concat_size = model_config.IMAGE_ENCODER.RGB.feature_dim +\
        #     self.model_config.PREV_ACTION_ENCODER.encoding_size*model_config.len_traj_act
        
        if self.model_config.IMAGE_ENCODER.RGB.img_mod == 'cls':
            concat_size = self.model_config.IMAGE_ENCODER.RGB.projection_dim
        elif self.model_config.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling':
            if self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'flat':
                concat_size = self.model_config.IMAGE_ENCODER.RGB.projection_dim * self.model_config.IMAGE_ENCODER.RGB.multi_patches_num
            elif self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'first':
                concat_size = self.model_config.IMAGE_ENCODER.RGB.projection_dim
                
        # Init the IMU encoder
        if self.model_config.IMU_ENCODER.use:
            self.imu_linear = nn.Linear(self.model_config.IMU_ENCODER.input_size, self.model_config.IMU_ENCODER.encoding_size)
            self.imu_linear_dp = nn.Linear(self.model_config.IMU_ENCODER.input_size, self.model_config.STATE_ENCODER.hidden_size) # This is used to encode IMU to hidden_states as the inputs for diffusion transformer
            concat_size += self.model_config.IMU_ENCODER.encoding_size
        
        if self.model_config.STEP_ENCODER.use:
            concat_size += self.model_config.STEP_ENCODER.encoding_size
        
        # if not model_config.IMAGE_ENCODER.use_stack:
            # concat_size += self.model_config.PREV_ACTION_ENCODER.encoding_size*model_config.len_traj_act
            # concat_size += self.model_config.PREV_ACTION_ENCODER.encoding_size
        concat_size += self.model_config.PREV_ACTION_ENCODER.encoding_size
        
        # self.state_compress_linear = nn.Linear(concat_size, model_config.STATE_ENCODER.hidden_size)
        
        # Init the GRU network
        self.state_encoder = encoders.build_rnn_state_encoder(
            input_size=concat_size,
            hidden_size=self.model_config.STATE_ENCODER.hidden_size,
            rnn_type=self.model_config.STATE_ENCODER.rnn_type,
            num_layers=self.model_config.STATE_ENCODER.num_layers
        )
        
        if self.model_config.STATE_ENCODER.use_dropout:
            self.state_dropout = nn.Dropout(self.model_config.STATE_ENCODER.dropout_rate)
        
        # Init the diffusion policy network
        self.use_local_cond = self.model_config.Diffusion_Policy.use_local_cond
        local_cond_dim = self.model_config.STATE_ENCODER.hidden_size if self.use_local_cond else None

        self.global_cond_linear = nn.Linear(self.model_config.TEXT_ENCODER.hidden_size, self.model_config.Diffusion_Policy.encoding_size)
        
        self.dp_type = self.model_config.Diffusion_Policy.type
        if self.model_config.Diffusion_Policy.type == 'resnet_unet':
            self.action_dp_pred_net = ConditionalUnet1D(
                    input_dim=self.num_actions,
                    local_cond_dim=local_cond_dim,
                    global_cond_dim=self.model_config.Diffusion_Policy.encoding_size,
                    down_dims=self.model_config.Diffusion_Policy.down_dims,
                    diffusion_step_embed_dim=self.model_config.Diffusion_Policy.diffusion_step_embed_dim,
                    cond_predict_scale=self.model_config.Diffusion_Policy.cond_predict_scale
                )
        elif self.model_config.Diffusion_Policy.type == 'transformer':
            self.use_cls_free_guidance = self.model_config.Diffusion_Policy.use_cls_free_guidance
            # define the length of conditions
            if self.model_config.IMAGE_ENCODER.use_stack:
                vis_length = self.model_config.len_traj_act
            else:
                if self.model_config.IMAGE_ENCODER.RGB.img_mod == 'cls':
                    vis_length = 1
                elif self.model_config.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling':
                    vis_length = self.model_config.IMAGE_ENCODER.RGB.multi_patches_num
            
            txt_length = self.model_config.TEXT_ENCODER.max_length
            if self.model_config.Diffusion_Policy.cond == 'v2_cutInstr':
                txt_length = self.model_config.Diffusion_Policy.txt_len
            if self.model_config.TEXT_ENCODER.use_qformer:
                txt_length = self.model_config.TEXT_ENCODER.q_former_length
            
            rnn_length = 1
            prev_act_length = self.model_config.len_traj_act
            imu_length = 1 if self.model_config.IMU_ENCODER.use else 0
            step_length = 1 if self.model_config.STEP_ENCODER.use else 0

            if self.model_config.Diffusion_Policy.cond == 'rnn_instr_vis':
                n_obs_steps = rnn_length + 1 + vis_length
            elif self.model_config.Diffusion_Policy.cond == 'rnn_instr':
                n_obs_steps = rnn_length + 1
            elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr_vis':
                n_obs_steps = rnn_length + txt_length + vis_length
            elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr':
                n_obs_steps = rnn_length + txt_length
            elif self.model_config.Diffusion_Policy.cond == 'v2_Fullinstr':
                # rnn, txt_cls, vis, step, imu, prev_actions
                n_obs_steps = rnn_length + txt_length + vis_length+1 + imu_length + step_length + prev_act_length
            elif self.model_config.Diffusion_Policy.cond == 'v2_instr':
                n_obs_steps = rnn_length + 1 + vis_length+1 + imu_length + step_length + prev_act_length
            elif self.model_config.Diffusion_Policy.cond == 'v2_cutInstr':
                n_obs_steps = rnn_length + txt_length + vis_length+1 + imu_length + step_length + prev_act_length
            self.action_dp_pred_net = TransformerForDiffusion(
                    input_dim=self.num_actions,
                    output_dim=self.num_actions,
                    horizon=self.model_config.Diffusion_Policy.len_traj_pred,
                    n_obs_steps=n_obs_steps,
                    n_emb=self.model_config.Diffusion_Policy.transformer_encoding_size, # This is the hidden states insided the transformer!
                    p_drop_emb=self.model_config.Diffusion_Policy.transformer_p_drop_emb,
                    cond_dim=self.model_config.STATE_ENCODER.hidden_size,
                    causal_attn=True,
                    time_as_cond=True,
                    n_layer=self.model_config.Diffusion_Policy.transformer_n_layers,
                    n_cond_layers=self.model_config.Diffusion_Policy.transformer_n_cond_layers,
                    use_dp=self.model_config.Diffusion_Policy.use # if not, the noise inputs will be learnable parameters
                )
            self.action_type_embeds = nn.Embedding(10, self.model_config.Diffusion_Policy.transformer_encoding_size)
        
        if self.model_config.Diffusion_Policy.scheduler == 'DDPM':
            # self.noise_scheduler = DDPMScheduler(
            #     num_train_timesteps=model_config.Diffusion_Policy.num_diffusion_iters,
            #     beta_schedule='squaredcos_cap_v2',
            #     clip_sample=True,
            #     prediction_type=model_config.Diffusion_Policy.pred_type
            # )
            self.noise_scheduler = myDDPMScheduler(
                num_train_timesteps=self.model_config.Diffusion_Policy.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type=self.model_config.Diffusion_Policy.pred_type
            )
        elif self.model_config.Diffusion_Policy.scheduler == 'DDIM':
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.model_config.Diffusion_Policy.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type=self.model_config.Diffusion_Policy.pred_type
            )
            self.noise_scheduler.set_timesteps(self.model_config.Diffusion_Policy.num_diffusion_iters)
       
        # Init the distance prediction network
        if self.model_config.DISTANCE_PREDICTOR.use:
            self.distance_pred_net = encoders.DistanceNetwork(
                embedding_dim=self.model_config.STATE_ENCODER.hidden_size, normalize=self.model_config.DISTANCE_PREDICTOR.normalize)

        # self._output_size = (
        #     model_config.STATE_ENCODER.hidden_size
        #     + model_config.RGB_ENCODER.output_size
        #     + model_config.DEPTH_ENCODER.output_size
        #     + self.instruction_encoder.output_size
        # )

        # self.register_buffer(
        #     "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        # )

        # self._output_size = model_config.STATE_ENCODER.hidden_size
        if self.model_config.PROGRESS_MONITOR.use:
            if self.model_config.PROGRESS_MONITOR.concat_state_txt:
                self.progress_monitor = encoders.DistanceNetwork(
                    embedding_dim=self.model_config.STATE_ENCODER.hidden_size*2, 
                    normalize=True) # pm_pred 
            else:
                self.progress_monitor = encoders.DistanceNetwork(
                    embedding_dim=self.model_config.STATE_ENCODER.hidden_size, 
                    normalize=True) # pm_pred 

            self._init_pm_layers(self.progress_monitor)
        
        # Init the stop progress predictor
        if self.model_config.STOP_PROGRESS_PREDICTOR.use:
            if self.model_config.STOP_PROGRESS_PREDICTOR.concat_state_txt:
                stop_hidden_dim = self.model_config.STATE_ENCODER.hidden_size*2
            else:
                stop_hidden_dim = self.model_config.STATE_ENCODER.hidden_size

            if self.model_config.STOP_PROGRESS_PREDICTOR.type == 'continuous':
                self.stop_progress_predictor = encoders.DistanceNetwork(
                    embedding_dim=stop_hidden_dim, 
                    normalize=True) # stop_progress_pred
            elif self.model_config.STOP_PROGRESS_PREDICTOR.type == 'logits':
                self.stop_progress_predictor = encoders.StopNetwork(
                    embedding_dim=stop_hidden_dim
                )

            self._init_pm_layers(self.stop_progress_predictor)
        
        self._output_size = self.num_actions

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        # TODO
        # return self.rgb_encoder.is_blind or self.depth_encoder.is_blind
        return False

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers

    def _init_pm_layers(self, pm_net) -> None:
        for param in pm_net.parameters():
            if param.ndim == 2:  # Typically weights are 2D
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif param.ndim == 1:  # Typically biases are 1D
                nn.init.constant_(param, 0)

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)
    
    def denoise_actions(self, noisy_diffusion_output, lv_state, type_embeds, device, sample_classifier_free_guidance=False, cls_free_guidance_scale=4, cond_mask=None, y_cond=None, y_cond_mask=None):
        noise = deepcopy(noisy_diffusion_output)
        batch_size = noisy_diffusion_output.shape[0]
        diffusion_output = noisy_diffusion_output

        for k in self.noise_scheduler.timesteps[:]:
            noise_pred = self.action_dp_pred_net(
                sample=diffusion_output, 
                timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                cond=lv_state.float(),
                type_embeds=type_embeds,
                cond_mask=cond_mask,
                y_cond=y_cond,
                y_cond_mask=y_cond_mask
                )

            if k!=0 and sample_classifier_free_guidance: #if k!=0
                noise_out, noise_out_null = noise_pred[:batch_size//2], noise_pred[batch_size//2:]
                noise_out = noise_out_null + cls_free_guidance_scale * (noise_out - noise_out_null) # TODO: check the scale of cls_free_guidance_scale
                # diff_out = cls_free_guidance_scale*(diff_out - diff_out_null)
                # noise_pred = torch.cat([noise_out, noise_out_null], dim=0)
                noise_pred = torch.cat([noise_out, noise_out], dim=0) #!!!???

            # inverse diffusion step (remove noise)
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output,
                add_random_variance=self.model_config.Diffusion_Policy.add_random_variance
            ).prev_sample
        
        if sample_classifier_free_guidance:
            diffusion_output = diffusion_output[:batch_size//2]

        return diffusion_output
    
    def pred_actions(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
        add_noise_to_action=True,
        denoise_action=False,
        num_sample=1,
        train_classifier_free_guidance=False,
        sample_classifier_free_guidance=False,
        need_txt_extraction=True,
        analysis_time=False
    ):
        # Note: stack images have not been adaptive yet.
        device = observations['instruction'].device
        batch_size = observations['instruction'].shape[0]
        
        # classifier-free guidance
        if self.model_config.Diffusion_Policy.use:
            if train_classifier_free_guidance and self.model_config.Diffusion_Policy.random_mask_instr and self.model_config.Diffusion_Policy.cls_mask_method == 'mask_inputs':
                # randomly mask the condition tokens for classifier-free guidance during training
                cls_free_mask = torch.rand(batch_size) < self.model_config.Diffusion_Policy.cls_mask_ratio
                cls_free_mask = cls_free_mask.to(device)
                observations['instruction'][cls_free_mask, :] = torch.ones_like(observations['instruction'][cls_free_mask, :]) * self.model_config.TEXT_ENCODER.eot_token # Not use zero_tokens since it is used as padding token

            if sample_classifier_free_guidance and self.model_config.Diffusion_Policy.cls_mask_method == 'mask_inputs':
                # copy condition to null for sampling
                obs_null = observations.copy()
                if self.model_config.Diffusion_Policy.random_mask_instr:
                    obs_null['instruction'] = torch.zeros_like(obs_null['instruction'])
                if self.model_config.Diffusion_Policy.random_mask_rgb:
                    obs_null['stack_rgb'] = torch.zeros_like(obs_null['stack_rgb'])
                    if not self.model_config.IMAGE_ENCODER.DEPTH.update_depth_encoder:
                        # if update_depth_encoder, the null depth has been masked during img feature extraction
                        obs_null['stack_depth'] = torch.zeros_like(obs_null['stack_depth'])
                    else:
                        obs_null['stack_depth'] = obs_null['stack_null_depth']

                for k,v in obs_null.items():
                    observations[k] = torch.cat([observations[k], obs_null[k]], dim=0)
                prev_actions = torch.cat([prev_actions, prev_actions], dim=0)
                batch_size = observations['instruction'].shape[0]
        
        if analysis_time:
            start_time = time.time()
        
        '''1. Encoding text'''
        text_embeds, txt_masks, text_cls_embeds = self.instruction_encoder(
            observations['instruction'], need_txt_extraction=need_txt_extraction
        ) 
        if analysis_time:
            end_time = time.time()
            print(f"Time taken to encode text: {end_time - start_time:.2f} seconds")
                
        '''2. Encoding previous actions and steps'''
        prev_actions_masks = prev_actions.float() * masks.unsqueeze(-1).float() # [bs, act_length, 3]
        prev_action_embeds = self.prev_action_embedding(prev_actions_masks)
        prev_action_dp_embeds = self.prev_action_embedding_dp(prev_actions_masks)
        latest_prev_action_embeds = prev_action_embeds[:,0]

        if self.model_config.STEP_ENCODER.use:
            assert 'steps' in observations
            steps_masks = (observations['steps'] * masks.squeeze()).type(torch.int) # [bs,1]
            steps_embeds = self.step_embeddings(steps_masks)
            steps_dp_embeds = self.step_embeddings_dp(steps_masks)

        # if not self.model_config.IMAGE_ENCODER.use_stack:
        #     # prev_action_embeds = self.prev_action_pos_embedding(prev_action_embeds)
        #     # prev_action_embeds = prev_action_embeds[:,0,:]
        #     prev_action_embeds = prev_action_embeds.reshape(batch_size, -1) # use the stacked prev_action_embeds!
        
        '''3. Encoding images'''
        if analysis_time:
            start_time = time.time()
        rgb_depth_embeds = self.image_encoder(observations['stack_rgb'], observations['stack_depth'], prev_action_embeds=prev_action_embeds, use_stack=self.model_config.IMAGE_ENCODER.use_stack, img_mod=self.model_config.IMAGE_ENCODER.RGB.img_mod)
        rgb_patch_num = observations['stack_rgb'].shape[1]
        if analysis_time:
            end_time = time.time()
            print(f"Time taken to encode images: {end_time - start_time:.2f} seconds")

        '''4. Update GRU'''
        # GPU inputs: [rgb_depth_embeds, latest_prev_act_embeds, imu_embeds]
        if self.model_config.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling':
            if self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'flat':
                rgb_depth_for_rnn = torch.flatten(rgb_depth_embeds, 1) # [bs, 5, h_dim] -> [bs, 5*h_dim]
            elif self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'first':
                rgb_depth_for_rnn = rgb_depth_embeds[:,0,:] # 只取第1维的特征做RNN
        else:
            rgb_depth_for_rnn = rgb_depth_embeds.squeeze(1)
        concat_embeds = torch.cat([rgb_depth_for_rnn, latest_prev_action_embeds], dim=1)
        if self.model_config.IMU_ENCODER.use:
            imu_embeds = self.imu_linear(observations['imu'])
            imu_dp_embeds = self.imu_linear_dp(observations['imu'])
            
            # Concat GRU input features
            concat_embeds = torch.cat([concat_embeds, imu_embeds], dim=1)
        
        if self.model_config.STEP_ENCODER.use:
            concat_embeds = torch.cat([concat_embeds, steps_embeds], dim=1)
        
        '''5. Compute GRU features'''
        state, rnn_states_out = self.state_encoder(concat_embeds, rnn_states, masks.bool()) # TODO: check sequence RNN
        state = state.unsqueeze(1)
        
        if self.model_config.STATE_ENCODER.use_dropout:
            state = self.state_dropout(state)
        # state:[total_bs, 512]
        # rnn_states_out: [bs, 1, 512]
        # if self.model_config.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling':
        #     state = state.reshape(batch_size, rgb_patch_num, -1)  # [bs, 5*h_dim] -> [bs, 5, h_dim]
        
        '''6. Encoding vision-and-language''' 
        if not self.model_config.IMAGE_ENCODER.use_stack and self.model_config.IMAGE_ENCODER.RGB.img_mod != 'multi_patches_avg_pooling':
            do_self_attn = False
        else:
            do_self_attn = True

        # 6.1 GRU historical features combine with the text features
        # his_txt_embeds, his_txt_attn_probs = self.his_txt_cross_encoder(state.unsqueeze(1), text_embeds, txt_masks, output_attentions=True,do_self_attn=False) # TODO
        # his_txt_attn_probs = his_txt_attn_probs[:,0,:]

        # 6.2 Current img features combine with the text features
        if analysis_time:
            start_time = time.time()
        rgb_depth_his_embeds = torch.cat((rgb_depth_embeds, state), dim=1)
        try:
            img_txt_embeds, img_txt_attn_probs = self.img_txt_cross_encoder(rgb_depth_his_embeds, text_embeds, q_masks=masks, kv_masks=txt_masks, output_attentions=True,do_self_attn=do_self_attn)
        except Exception as e:
            print(e)
            img_txt_embeds, img_txt_attn_probs = self.img_txt_cross_encoder(rgb_depth_his_embeds, text_embeds, q_masks=masks, kv_masks=txt_masks, output_attentions=True,do_self_attn=do_self_attn)
        img_txt_attn_probs = img_txt_attn_probs[:,0,:]
        if analysis_time:
            end_time = time.time()
            print(f"Time taken to encode cross-modal features: {end_time - start_time:.2f} seconds")

        # 6.3 Current text features combine with the historical img features
        # txt_his_embeds, txt_hit_attn_probs = self.txt_img_cross_encoder(text_embeds, state.unsqueeze(1), q_masks=txt_masks, kv_masks=masks, output_attentions=True,do_self_attn=do_self_attn)
        # txt_hit_attn_probs = txt_hit_attn_probs[:,0,:]

        # 6.4 Current text features combine with the current img features
        if self.model_config.CROSS_MODAL_ENCODER.txt_to_img:
            if analysis_time:
                start_time = time.time()    
            txt_img_embeds, txt_img_attn_probs = self.txt_img_cross_encoder(text_embeds, rgb_depth_his_embeds, q_masks=txt_masks, kv_masks=None, output_attentions=True,do_self_attn=do_self_attn) # kv_masks set to be None since there is no mask for imgs
            if analysis_time:
                end_time = time.time()
                print(f"Time taken to encode text cross-modal features: {end_time - start_time:.2f} seconds")
            fused_update_txt_embeds = txt_img_embeds
        else:
            fused_update_txt_embeds = text_embeds

        # 6.5 Combine the text features
        # fused_update_txt_embeds = (txt_his_embeds + txt_img_embeds) / 2
        # fused_update_txt_embeds = txt_img_embeds

        # fused_cross_modal_embeds, attention_probs = self.cross_modal_encoder(rgb_depth_embeds, text_embeds, txt_masks, output_attentions=True,do_self_attn=do_self_attn)
        # attention_probs = attention_probs[:,0,:]
        
        '''7. Predict action distribution using diffusion policy'''
        # Sample a diffusion iteration for each data point
        if self.model_config.Diffusion_Policy.use:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
        else:
            timesteps = None
        
        noise = noise_pred = diffusion_output = None
        denoise_action_list = []
        
        if denoise_action:
            # initialize action from Gaussian noise
            noisy_diffusion_output = torch.randn(
                (batch_size, self.model_config.Diffusion_Policy.len_traj_pred, self.num_actions), device=device)
            noise = deepcopy(noisy_diffusion_output)
            diffusion_output = noisy_diffusion_output
            # predict noise
            if self.dp_type == 'transformer':
                if self.model_config.Diffusion_Policy.cond == 'rnn_instr_vis':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1), rgb_depth_embeds), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_instr':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1)), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1)), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr_vis':
                    lv_state = torch.cat((state.unsqueeze(1), text_embeds, rgb_depth_embeds), dim=1)
                elif 'v2' in self.model_config.Diffusion_Policy.cond:
                    # Inputs: [steps, imu, prev_act, his, img, txt]
                    if self.model_config.STEP_ENCODER.use:
                        steps_dp_embeds = steps_dp_embeds.unsqueeze(1)
                    if self.model_config.IMU_ENCODER.use:
                        imu_dp_embeds = imu_dp_embeds.unsqueeze(1)
                    prev_action_dp_embeds = prev_action_dp_embeds
                    if self.model_config.Diffusion_Policy.cond == 'v2_instr':
                        txt_dp_embeds = fused_update_txt_embeds[:,0,:].unsqueeze(1)
                    elif self.model_config.Diffusion_Policy.cond == 'v2_Fullinstr':
                        txt_dp_embeds = fused_update_txt_embeds
                    elif self.model_config.Diffusion_Policy.cond == 'v2_cutInstr':
                        # assign the specified length of txt_dp_embeds
                        txt_dp_embeds = fused_update_txt_embeds[:,:self.model_config.Diffusion_Policy.txt_len,:]
                    lv_state = torch.cat([img_txt_embeds, txt_dp_embeds, state], dim=1)
                    if self.model_config.STEP_ENCODER.use:
                        lv_state = torch.cat([lv_state, steps_dp_embeds],dim=1)
                    if self.model_config.IMU_ENCODER.use:
                        lv_state = torch.cat([lv_state, imu_dp_embeds], dim=1)
                    lv_state = torch.cat([lv_state, prev_action_dp_embeds], dim=1)
                    # lv_state = torch.cat([img_txt_embeds, txt_dp_embeds, state, steps_dp_embeds, imu_dp_embeds, prev_action_dp_embeds], dim=1)

                    type_embeds = ([0] * img_txt_embeds.shape[1] +  # img_txt_embeds
                                [1] * txt_dp_embeds.shape[1] +  # txt_dp_embeds
                                [2] * state.shape[1])
                    
                    if self.model_config.STEP_ENCODER.use:
                        type_embeds += [3] * steps_dp_embeds.shape[1]
                    if self.model_config.IMU_ENCODER.use:
                        type_embeds += [4] * imu_dp_embeds.shape[1]
                    type_embeds += [5] * prev_action_dp_embeds.shape[1]

                type_embeds = torch.from_numpy(np.array(type_embeds)).to(device)
                type_embeds = self.action_type_embeds(type_embeds).repeat(batch_size, 1, 1)
                
                if sample_classifier_free_guidance and self.model_config.Diffusion_Policy.cls_mask_method == 'mask_token':
                    uncond_mask = torch.zeros(batch_size, lv_state.shape[1]).to(device)
                    if self.model_config.Diffusion_Policy.random_mask_instr:
                        uncond_mask[:, img_txt_embeds.shape[1]:img_txt_embeds.shape[1]+txt_dp_embeds.shape[1]] = 1
                    if self.model_config.Diffusion_Policy.random_mask_rgb:
                        uncond_mask[:, :img_txt_embeds.shape[1]] = 1
                    
                    t_token_mask = torch.zeros(batch_size, 1).to(device)
                    uncond_mask = torch.cat([t_token_mask, uncond_mask], dim=1)
                    
                    cond_mask = torch.zeros_like(uncond_mask)
                    cond_mask = torch.cat([cond_mask, uncond_mask], dim=0)
                    batch_size = cond_mask.shape[0]
                    
                    type_embeds = torch.cat([type_embeds, type_embeds], dim=0)
                    lv_state = torch.cat([lv_state, lv_state], dim=0)
                    
                    if self.model_config.Diffusion_Policy.state_concat_with_noise:
                        # create y-token for cross-attn, and original state-token for self-attn
                        # y-token: [cur_obs, txt_cls, imu]
                        # Convert mask values of 1 to -inf
                        if cond_mask is not None:
                            cond_mask = torch.where(cond_mask == 1, float('-inf'), cond_mask)

                        if sample_classifier_free_guidance and self.model_config.Diffusion_Policy.cls_mask_method == 'mask_token':
                            y_cond_mask = torch.zeros(batch_size, y_cond.shape[1]+1).to(device)
                            if self.model_config.Diffusion_Policy.random_mask_instr:
                                y_cond_mask[batch_size//2:, 2] = 1
                            if self.model_config.Diffusion_Policy.random_mask_rgb:
                                y_cond_mask[batch_size//2:, 1] = 1
                        
                            y_cond = torch.cat([y_cond, y_cond], dim=0)
                    else:
                        y_cond = None
                        y_cond_mask = None      
                else:
                    if self.model_config.Diffusion_Policy.state_concat_with_noise:
                        cond_mask = torch.zeros(batch_size, lv_state.shape[1]+1).to(device)
                        y_cond_mask = torch.zeros(batch_size, y_cond.shape[1]+1).to(device)
                    else:
                        cond_mask, y_cond, y_cond_mask = None, None, None

            if self.model_config.Diffusion_Policy.use: # use standard diffusion policy
                if num_sample > 1:
                    for sample_idx in range(num_sample):
                        if sample_classifier_free_guidance:
                            noise_bs = batch_size // 2
                            noisy_diffusion_output = torch.randn(
                                (noise_bs, self.model_config.Diffusion_Policy.len_traj_pred, self.num_actions), device=device)
                            noisy_diffusion_output = torch.cat([noisy_diffusion_output, noisy_diffusion_output], dim=0)
                        else:
                            noise_bs = batch_size
                            noisy_diffusion_output = torch.randn(
                                (noise_bs, self.model_config.Diffusion_Policy.len_traj_pred, self.num_actions), device=device)
                        diffusion_output = self.denoise_actions(noisy_diffusion_output, lv_state, type_embeds, device, sample_classifier_free_guidance, cls_free_guidance_scale=self.model_config.Diffusion_Policy.cls_free_guidance_scale, cond_mask=cond_mask, y_cond=y_cond, y_cond_mask=y_cond_mask)
                        denoise_action_list.append(diffusion_output)
                else:
                    # initialize action from Gaussian noise
                    if sample_classifier_free_guidance:
                        noise_bs = batch_size // 2
                        noisy_diffusion_output = torch.randn(
                            (noise_bs, self.model_config.Diffusion_Policy.len_traj_pred, self.num_actions), device=device)
                        noisy_diffusion_output = torch.cat([noisy_diffusion_output, noisy_diffusion_output], dim=0)
                    else:
                        noise_bs = batch_size
                        noisy_diffusion_output = torch.randn(
                            (noise_bs, self.model_config.Diffusion_Policy.len_traj_pred, self.num_actions), device=device)
                    diffusion_output = self.denoise_actions(noisy_diffusion_output, lv_state, type_embeds, device, sample_classifier_free_guidance, cls_free_guidance_scale=self.model_config.Diffusion_Policy.cls_free_guidance_scale, cond_mask=cond_mask, y_cond=y_cond, y_cond_mask=y_cond_mask)
            
            else: # directly regress the action
                noisy_action = None
                diffusion_output = self.action_dp_pred_net(
                    sample=noisy_action, 
                    timestep=timesteps,
                    cond=lv_state.float(),
                    type_embeds=type_embeds,
                    cond_mask=None
                    )
                
        else:
            if self.model_config.Diffusion_Policy.use and add_noise_to_action:
                # Add noise to the clean images according to the noise magnitude at each diffusion iterationd
                # Sample noise to add to actions     
                naction = observations['actions'] # which has been normalized
                noise = torch.randn(naction.shape, device=device)
                noisy_action = self.noise_scheduler.add_noise(
                    naction, noise, timesteps)
            else:
                noisy_action = observations['actions']
            
            if self.dp_type == 'transformer':
                if self.model_config.Diffusion_Policy.cond == 'rnn_instr_vis':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1), rgb_depth_embeds), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_instr':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1)), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1)), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr_vis':
                    lv_state = torch.cat((state.unsqueeze(1), text_embeds, rgb_depth_embeds), dim=1)
                elif 'v2' in self.model_config.Diffusion_Policy.cond:
                    # Inputs: [steps, imu, prev_act, his, img, txt]
                    if self.model_config.STEP_ENCODER.use:
                        steps_dp_embeds = steps_dp_embeds.unsqueeze(1)
                    if self.model_config.IMU_ENCODER.use:
                        imu_dp_embeds = imu_dp_embeds.unsqueeze(1)
                    prev_action_dp_embeds = prev_action_dp_embeds
                    if self.model_config.Diffusion_Policy.cond == 'v2_instr':
                        txt_dp_embeds = fused_update_txt_embeds[:,0,:].unsqueeze(1)
                    elif self.model_config.Diffusion_Policy.cond == 'v2_Fullinstr':
                        txt_dp_embeds = fused_update_txt_embeds
                    elif self.model_config.Diffusion_Policy.cond == 'v2_cutInstr':
                        # assign the specified length of txt_dp_embeds
                        txt_dp_embeds = fused_update_txt_embeds[:,:self.model_config.Diffusion_Policy.txt_len,:]
                    lv_state = torch.cat([img_txt_embeds, txt_dp_embeds, state], dim=1)
                    if self.model_config.STEP_ENCODER.use:
                        lv_state = torch.cat([lv_state, steps_dp_embeds],dim=1)
                    if self.model_config.IMU_ENCODER.use:
                        lv_state = torch.cat([lv_state, imu_dp_embeds], dim=1)
                    lv_state = torch.cat([lv_state, prev_action_dp_embeds], dim=1)
                    # lv_state = torch.cat([img_txt_embeds, txt_dp_embeds, state, steps_dp_embeds, imu_dp_embeds, prev_action_dp_embeds], dim=1)

                    type_embeds = ([0] * img_txt_embeds.shape[1] +  # img_txt_embeds
                                [1] * txt_dp_embeds.shape[1] +  # txt_dp_embeds
                                [2] * state.shape[1])
                    
                    if self.model_config.STEP_ENCODER.use:
                        type_embeds += [3] * steps_dp_embeds.shape[1]
                    if self.model_config.IMU_ENCODER.use:
                        type_embeds += [4] * imu_dp_embeds.shape[1]
                    type_embeds += [5] * prev_action_dp_embeds.shape[1]

                    type_embeds = torch.from_numpy(np.array(type_embeds)).to(device)
                    type_embeds = self.action_type_embeds(type_embeds).repeat(batch_size, 1, 1)
                    
                cond_mask = torch.zeros(batch_size, lv_state.shape[1]).to(device)
                if train_classifier_free_guidance and self.model_config.Diffusion_Policy.cls_mask_method == 'mask_token':
                    mask_prob = torch.rand(batch_size) < self.model_config.Diffusion_Policy.cls_mask_ratio
                    if self.model_config.Diffusion_Policy.random_mask_instr:
                        cond_mask[mask_prob, img_txt_embeds.shape[1]:img_txt_embeds.shape[1]+txt_dp_embeds.shape[1]] = 1
                    if self.model_config.Diffusion_Policy.random_mask_rgb:
                        cond_mask[mask_prob, :img_txt_embeds.shape[1]] = 1
                    
                t_token_mask = torch.zeros(batch_size, 1).to(device)
                cond_mask = torch.cat([t_token_mask, cond_mask], dim=1)
                
                if not self.model_config.Diffusion_Policy.use:
                    cond_mask = None
                
                if self.model_config.Diffusion_Policy.state_concat_with_noise:
                    # create y-token for cross-attn, and original state-token for self-attn
                    # y-token: [cur_obs, txt_cls, imu]
                    # Convert mask values of 1 to -inf
                    if cond_mask is not None:
                        cond_mask = torch.where(cond_mask == 1, float('-inf'), cond_mask)
                    y_cond = torch.cat([rgb_depth_embeds[:,0,:].unsqueeze(1),
                                      text_cls_embeds.unsqueeze(1)], dim=1)
                    if self.model_config.IMU_ENCODER.use:
                        y_cond = torch.cat([y_cond, imu_embeds], dim=1)
                        
                    y_cond_mask = torch.zeros(batch_size, y_cond.shape[1]+1).to(device) # +1 for the time embedding
                    if train_classifier_free_guidance and self.model_config.Diffusion_Policy.cls_mask_method == 'mask_token':             
                        if self.model_config.Diffusion_Policy.random_mask_instr:
                            y_cond_mask[mask_prob, 2] = 1
                        if self.model_config.Diffusion_Policy.random_mask_rgb:
                            y_cond_mask[mask_prob, 1] = 1
                else:
                    # if not train_classifier_free_guidance:
                    #     cond_mask = None # TODO
                    y_cond = None
                    y_cond_mask = None
                
                if analysis_time:
                    start_time = time.time()
                noise_pred = self.action_dp_pred_net(
                    sample=noisy_action.float(), 
                    timestep=timesteps,
                    cond=lv_state.float(),
                    type_embeds=type_embeds,
                    cond_mask=cond_mask,
                    y_cond=y_cond,
                    y_cond_mask=y_cond_mask)
                if analysis_time:
                    end_time = time.time()
                    print(f"Time taken to diffusion pred noise: {end_time - start_time:.2f} seconds")
            elif self.dp_type == 'resnet_unet':
                # Predict the noise residual
                if self.use_local_cond:
                    # use image rnn as local_condition
                    local_cond = state.unsqueeze(1).expand(-1, self.model_config.Diffusion_Policy.len_traj_pred, -1).float()
                    # use text features as global_condition
                    # Here I use the image-text cross-attention to get the global_condition
                    global_cond = torch.mul(text_embeds, attention_probs.unsqueeze(-1)).sum(1)
                    global_cond = self.global_cond_linear(global_cond)

                    noise_pred = self.action_dp_pred_net(
                        sample=noisy_action.float(), 
                        timestep=timesteps,
                        local_cond=local_cond,
                        global_cond=global_cond)
                else:
                    noise_pred = self.action_dp_pred_net(
                        sample=noisy_action.float(), 
                        timestep=timesteps,
                        global_cond=state.float())
        
        '''9. Predict distances'''
        dist_pred = None
        if self.model_config.DISTANCE_PREDICTOR.use:
            dist_pred = self.distance_pred_net(state.squeeze(1))
        
        progress_pred = None
        # if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
        if self.model_config.PROGRESS_MONITOR.use:
            # progress_pred = torch.tanh(self.progress_monitor(state)) # pm_pred 
            if self.model_config.PROGRESS_MONITOR.concat_state_txt:
                progress_pred = self.progress_monitor(torch.cat([state.squeeze(1), fused_update_txt_embeds[:,0,:]], dim=1))
            else:
                progress_pred = self.progress_monitor(state.squeeze(1))

            # if self.model_config.Diffusion_Policy.use:
            #     if sample_classifier_free_guidance:
            #         progress_pred = torch.split(progress_pred, batch_size // 2, dim=0)
            #         progress_pred = progress_pred[0]
        
        stop_progress_pred = None
        if self.model_config.STOP_PROGRESS_PREDICTOR.use:
            if self.model_config.STOP_PROGRESS_PREDICTOR.concat_state_txt:
                stop_progress_pred = self.stop_progress_predictor(torch.cat([state.squeeze(1), fused_update_txt_embeds[:,0,:]], dim=1))
            else:
                stop_progress_pred = self.stop_progress_predictor(state.squeeze(1))
            # if self.model_config.Diffusion_Policy.use:
            #     if sample_classifier_free_guidance:
            #         stop_progress_pred = torch.split(stop_progress_pred, batch_size // 2, dim=0)
            #         stop_progress_pred = stop_progress_pred[0]

        return noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_pred, denoise_action_list, stop_progress_pred

    def update_rnn_states(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor
    ):
        # Note: stack images have not been adaptive yet.
        device = observations['instruction'].device
        batch_size = observations['instruction'].shape[0]
        
        '''1. Encoding text'''
        text_embeds, txt_masks, text_cls_embeds = self.instruction_encoder(
            observations['instruction']
        ) 
                
        '''2. Encoding previous actions and steps'''
        prev_actions_masks = prev_actions.float() * masks.unsqueeze(-1).float() # [bs, act_length, 3]
        prev_action_embeds = self.prev_action_embedding(prev_actions_masks)
        prev_action_dp_embeds = self.prev_action_embedding_dp(prev_actions_masks)
        latest_prev_action_embeds = prev_action_embeds[:,0]

        if self.model_config.STEP_ENCODER.use:
            assert 'steps' in observations
            steps_masks = (observations['steps'] * masks.squeeze()).type(torch.int) # [bs,1]
            steps_embeds = self.step_embeddings(steps_masks)
            steps_dp_embeds = self.step_embeddings_dp(steps_masks)

        # if not self.model_config.IMAGE_ENCODER.use_stack:
        #     # prev_action_embeds = self.prev_action_pos_embedding(prev_action_embeds)
        #     # prev_action_embeds = prev_action_embeds[:,0,:]
        #     prev_action_embeds = prev_action_embeds.reshape(batch_size, -1) # use the stacked prev_action_embeds!
        
        '''3. Encoding images'''
        rgb_depth_embeds = self.image_encoder(observations['stack_rgb'], observations['stack_depth'], prev_action_embeds=prev_action_embeds, use_stack=self.model_config.IMAGE_ENCODER.use_stack, img_mod=self.model_config.IMAGE_ENCODER.RGB.img_mod)
        rgb_patch_num = observations['stack_rgb'].shape[1]

        '''4. Update GRU'''
        # GPU inputs: [rgb_depth_embeds, latest_prev_act_embeds, imu_embeds]
        if self.model_config.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling':
            if self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'flat':
                rgb_depth_for_rnn = torch.flatten(rgb_depth_embeds, 1) # [bs, 5, h_dim] -> [bs, 5*h_dim]
            elif self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'first':
                rgb_depth_for_rnn = rgb_depth_embeds[:,0,:] # 只取第1维的特征做RNN
        else:
            rgb_depth_for_rnn = rgb_depth_embeds.squeeze(1)
        concat_embeds = torch.cat([rgb_depth_for_rnn, latest_prev_action_embeds], dim=1)
        if self.model_config.IMU_ENCODER.use:
            imu_embeds = self.imu_linear(observations['imu'])
            imu_dp_embeds = self.imu_linear_dp(observations['imu'])
            
            # Concat GRU input features
            concat_embeds = torch.cat([concat_embeds, imu_embeds], dim=1)
        
        if self.model_config.STEP_ENCODER.use:
            concat_embeds = torch.cat([concat_embeds, steps_embeds], dim=1)
        
        '''5. Compute GRU features'''
        state, rnn_states_out = self.state_encoder(concat_embeds, rnn_states, masks.bool()) # TODO: check sequence RNN
        return state, rnn_states_out

    def img_embedding(self, rgb_inputs, depth_inputs, img_mod, depth_return_x_before_fc=False, proj=True, process_images=False, need_rgb_extraction=True):
        if process_images:
            rgb_inputs = self.image_encoder.process_image(rgb_inputs)
            if self.model_config.IMAGE_ENCODER.DEPTH.bottleneck == 'TAC':
                depth_inputs = self.image_encoder.process_depth(depth_inputs)
        if need_rgb_extraction:
            rgb_embeds = self.image_encoder.embed_image(rgb_inputs,img_mod=img_mod, proj=proj).squeeze(1)
        else:
            rgb_embeds = rgb_inputs
            
        depth_embeds = self.image_encoder.embed_depth(depth_inputs, return_x_before_fc=depth_return_x_before_fc).squeeze(1)
        return rgb_embeds, depth_embeds
        
    def parse_action(self, diffusion_output, dist_pred, pm_pred=None, stop_mode='distance', steps=None, stop_pm_pred=None):
        cumsum = False if self.config.EVAL.ACTION == 'descrete' else True
        if self.model_config.learn_angle:
            un_actions = get_action(diffusion_output, self.action_stats, cumsum=cumsum)
            un_actions_nocumsum = get_action(diffusion_output, self.action_stats, cumsum=False)
            actions_cumsum = get_action(diffusion_output, self.action_stats, cumsum=True)
        else:
            un_actions = diffusion_output
            un_actions_nocumsum, actions_cumsum = un_actions, None

        un_actions = un_actions.detach().cpu().numpy()

        if self.config.EVAL.ACTION == 'xyyaw' or self.config.EVAL.ACTION == 'speed':
            actions = []
            output_stop = False
            # un_actions = un_actions_nocumsum.detach().cpu().numpy()
            for idx in range(un_actions_nocumsum[0].shape[0]):
                if stop_mode in ['progress', 'stop_progress']:
                    if stop_mode == 'stop_progress':
                        stop_flag = stop_pm_pred[0].item() > self.config.EVAL.continuous_stop_pm_threshold
                    else:
                        stop_flag = pm_pred[0].item() > self.config.EVAL.pm_threshold
                    M_stops = 3
                    # Check if M consecutive steps are stop actions
                    if idx + M_stops < len(un_actions_nocumsum[0]):  # Make sure we have enough steps ahead
                        consecutive_stops = True
                        for i in range(M_stops):  # Check current and next M steps
                            curr_action = un_actions_nocumsum[0][idx+i]
                            if not (abs(curr_action[0]) < float(self.config.EVAL.stop_x_threshold) and \
                                  abs(curr_action[1]) < float(self.config.EVAL.stop_y_threshold) and \
                                  abs(curr_action[2]) < float(self.config.EVAL.stop_yaw_threshold)):
                                consecutive_stops = False
                                break
                        
                        if consecutive_stops or stop_flag:
                            # Only stop if we have 4 consecutive stop actions and progress monitor threshold is met
                            actions.append("STOP")
                            output_stop = True
                            continue
                actions.append(un_actions[0][idx]) 
            
            if output_stop:
                # once stop, the action should be stop
                actions[0] = "STOP"
            actions = [actions]
        elif self.config.EVAL.ACTION == 'descrete':
            # 0: stop, 1: move forward, 2: turn left, 3: turn right
            actions = [[] for _ in range(un_actions.shape[0])]
            for bs_idx in range(un_actions.shape[0]):
                if self.model_config.learn_angle: # output is 3-dim
                    stop_th = 3 # 20241026新更新：stop必须要后面连续3个动作都为0，才停；否则就执行后面不是0的动作。
                    for step_idx in range(un_actions.shape[1]):
                        if stop_mode == 'distance':
                            assert dist_pred is not None # distance_predictor should be used for stop_mode 'distance'
                            if dist_pred[bs_idx].item() < self.config.EVAL.distance_threshold:
                                # stop
                                actions[bs_idx].append(action_spaces['stop'])
                                stop = True
                                continue
                        elif stop_mode == 'progress':
                            stop_flag = False
                            steps = None # !!! 
                            if steps is not None:
                                stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                                (steps[bs_idx] > 5 and abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1 and abs(un_actions[bs_idx][step_idx][2]) < 1e-1)
                            else:
                                stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                                    (abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1 and abs(un_actions[bs_idx][step_idx][2]) < 1e-1)
                            # stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                            #     (abs(diffusion_output[bs_idx][step_idx][0]) < 3e-1 and abs(diffusion_output[bs_idx][step_idx][1]) < 3e-1 and abs(diffusion_output[bs_idx][step_idx][2]) < 3e-1)

                            if stop_flag:
                                actions[bs_idx].append(action_spaces['stop'])
                                stop = True
                                continue
                        if abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1:
                            # turn left or turn right
                            if un_actions[bs_idx][step_idx][2] > 0:
                                actions[bs_idx].append(action_spaces['turn_left'])
                            elif un_actions[bs_idx][step_idx][2] < 0:
                                actions[bs_idx].append(action_spaces['turn_right'])
                        else:
                            # move forward
                            actions[bs_idx].append(action_spaces['go_forward'])

                    # 20241026新更新：stop必须要后面连续3个动作都为0，才停；否则就执行后面不是0的动作。
                    # for step_idx in range(self.model_config.len_traj_act):
                    #     if actions[bs_idx][step_idx] == action_spaces['stop'] and actions[bs_idx][step_idx+1] == action_spaces['stop'] and actions[bs_idx][step_idx+2] == action_spaces['stop']:
                    #         continue
                    #     else:
                    #         if actions[bs_idx][step_idx] == action_spaces['stop']:
                    #             actions[bs_idx][step_idx] = action_spaces['wait']
                    # if stop:
                    #     # stop once the stop action is selected for multiple step predictions
                    #     actions[bs_idx][0] = action_spaces['stop']
                else:
                    # output is 2-dim descrete action
                    for step_idx in range(un_actions.shape[1]):
                        stop_flag = False
                        steps = None
                        if steps is not None:
                            stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                            (steps[bs_idx] > 5 and abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1)
                        else:
                            stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                            (abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1)
                        if stop_flag:
                            actions[bs_idx].append(action_spaces['stop'])
                            continue
                        if abs(un_actions[bs_idx][step_idx][0]) > 2e-1:
                            actions[bs_idx].append(action_spaces['go_forward'])
                        elif un_actions[bs_idx][step_idx][1] < 0:
                            actions[bs_idx].append(action_spaces['turn_right'])
                        elif un_actions[bs_idx][step_idx][1] > 0:
                            actions[bs_idx].append(action_spaces['turn_left'])
                
        return actions, actions_cumsum, un_actions_nocumsum
    
    def save_predicted_actions(self, un_actions, gt_actions=None, N=1, save_dir=None, step=None):
        for item_idx in range(N):
            plt.clf()
            plt.figure(figsize=(8, 8))  # 增大图像尺寸以便更好地显示
            
            # 创建以(0,0)为中心的坐标轴
            ax = plt.gca()
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            
            # Plot predicted actions with arrows
            # X is forward, positive Y is left, negative Y is right
            plt.scatter(-un_actions[item_idx][:, 1], un_actions[item_idx][:, 0], label='un_actions', color='blue', alpha=0.5)
            for i in range(un_actions[item_idx].shape[0]):
                # Calculate arrow direction components using yaw angle
                arrow_length = 0.2  # Adjust this value to change arrow length
                dx = arrow_length * np.cos(np.pi/2+un_actions[item_idx][i, 2])
                dy = arrow_length * np.sin(np.pi/2+un_actions[item_idx][i, 2])
                
                # Draw arrow
                plt.arrow(-un_actions[item_idx][i, 1], 
                        un_actions[item_idx][i, 0], 
                        dx, dy, 
                        head_width=0.05, 
                        head_length=0.1, 
                        fc='blue', 
                        ec='blue',
                        alpha=0.5)
                
                # Add point index
                plt.text(-un_actions[item_idx][i, 1], un_actions[item_idx][i, 0], 
                        str(i), fontsize=9, color='blue', ha='left')

            # Plot ground truth actions with arrows
            if gt_actions is not None:
                plt.scatter(-gt_actions[item_idx][:, 1], gt_actions[item_idx][:, 0], label='gt_actions', color='red', alpha=0.5)
                for i in range(gt_actions[item_idx].shape[0]):
                    # Calculate arrow direction components using yaw angle
                    arrow_length = 0.2  # Adjust this value to change arrow length
                    dx = arrow_length * np.cos(np.pi/2+gt_actions[item_idx][i, 2])
                    dy = arrow_length * np.sin(np.pi/2+gt_actions[item_idx][i, 2])
                    
                    # Draw arrow
                    plt.arrow(-gt_actions[item_idx][i, 1], 
                            gt_actions[item_idx][i, 0], 
                            dx, dy, 
                            head_width=0.05, 
                            head_length=0.1, 
                            fc='red', 
                            ec='red',
                            alpha=0.5)
                    
                    # Add point index
                    plt.text(-gt_actions[item_idx][i, 1], gt_actions[item_idx][i, 0], 
                            str(i), fontsize=9, color='red', ha='right')

            # 设置坐标轴标签
            plt.xlabel('y', x=1.0, ha='center')
            plt.ylabel('x', y=1.0, ha='center')
            
            # 获取数据范围并设置对称的显示范围
            max_range = max(
                abs(plt.xlim()[0]), abs(plt.xlim()[1]),
                abs(plt.ylim()[0]), abs(plt.ylim()[1])
            )
            plt.xlim(-max_range*1.2, max_range*1.2)
            plt.ylim(-max_range*1.2, max_range*1.2)

            # 在(0,0)处画一个点
            plt.plot(0, 0, 'ko', markersize=5)  # 在原点画一个黑点
            
            # 移动图例到右上角
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.axis('equal')  # Make sure the aspect ratio is equal
            
            if save_dir is not None:
                save_path = os.path.join(save_dir, f'model_output_step_{step}.jpg')
            else:
                save_path = f'data/images/model_output_{item_idx}_step_{step}.jpg'
            plt.savefig(save_path)
            print(f"save fig to {save_path}")

            plt.close()

    def act(self, batch):
        observations = batch['observations']
        rnn_states = batch['rnn_states']
        prev_actions = batch['prev_actions']
        masks = batch['masks']
        add_noise_to_action = batch['add_noise_to_action']
        denoise_action = batch['denoise_action']
        predicted_actions_save_dir = batch['predicted_actions_save_dir'] if 'predicted_actions_save_dir' in batch else None
        # batch['mode'] = 'pred_actions'
        
        batch_size = rnn_states.shape[0]
        vis = batch['vis']
        step = batch['step']
        episode_ids = batch['episode_ids']

        noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_pred, denoise_action_list, stop_progress_pred = self.pred_actions(batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks'], batch['add_noise_to_action'], batch['denoise_action'], batch['num_sample'])

        # prev_actions = diffusion_output[:,:self.model_config.len_traj_act]
        if batch['denoise_action'] and batch['num_sample'] > 1:         
            self.draw_multiple_actions(denoise_action_list, batch, episode_ids, predicted_actions_save_dir, step)
            
            # randomly sample one from list
            # actions.append(actions_list[np.random.randint(0, len(actions_list))][0])
            # un_actions_nocumsum.append(un_actions_nocumsum_list[np.random.randint(0, len(un_actions_nocumsum_list))])
            
        else:
            if vis:
                un_actions = get_action(diffusion_output, self.action_stats).cpu().detach().numpy()
                self.save_predicted_actions(un_actions, gt_actions=None, N=1, save_dir=predicted_actions_save_dir, step=step)
        
            actions, actions_cumsum, un_actions_nocumsum = self.parse_action(diffusion_output, dist_pred, pm_pred=progress_pred, stop_mode=batch['stop_mode'], steps=batch['steps'], stop_pm_pred=stop_progress_pred)
        
        return actions, rnn_states_out, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum, progress_pred, stop_progress_pred
    
    def draw_multiple_actions(self, denoise_action_list, batch, episode_ids, save_dir=None, step=0):
        actions = []
        un_actions_nocumsum = []
        rnn_states_list = []

        # 创建图像（只创建一次）
        fix, ax = plt.subplots(1, 1, figsize=(8, 8))

        # 设置坐标轴
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # 存储所有轨迹的数据范围
        all_x = []
        all_y = []

        for i in range(batch['num_sample']):
            dp_output = denoise_action_list[i]
            actions_list = []
            un_actions_nocumsum_list = []
            
            # 获取动作并转换为numpy数组
            un_actions = get_action(dp_output, self.action_stats).cpu().detach().numpy()
            
            # 收集数据范围
            all_x.extend(un_actions[0][:, 0])
            all_y.extend(un_actions[0][:, 1])
            # 绘制轨迹
            ax.plot(un_actions[0][:, 0], un_actions[0][:, 1], 
                alpha=0.5, marker='o', label=f'Sample {i+1}')
            
            actions_list.append(un_actions)
            un_actions_nocumsum_list.append(un_actions)
            
            # 保存到总列表
            actions.append(actions_list)
            un_actions_nocumsum.append(un_actions_nocumsum_list)

        # 设置对称的显示范围
        max_range = max(
            abs(max(all_x)), abs(min(all_x)),
            abs(max(all_y)), abs(min(all_y))
        )
        ax.set_xlim(-max_range*1.2, max_range*1.2)
        ax.set_ylim(-max_range*1.2, max_range*1.2)
        
        # 添加原点和网格
        ax.plot(0, 0, 'ko', markersize=5)
        ax.grid(True)
        ax.axis('equal')
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 保存图像
        save_dir = 'logs/images/num_samples'
        os.makedirs(save_dir, exist_ok=True)
        save_file = f'logs/images/num_samples/EpisodeId_{episode_ids}_step_{step}.png'
        plt.savefig(save_file)
        print(f"Save image to {save_file}")
        
        plt.close()

    def forward(
        self, batch
    ) -> Tuple[Tensor, Tensor]:
        mode = batch['mode']
        analysis_time = batch['analysis_time'] if 'analysis_time' in batch else False
        
        if mode == "img_embedding":
            if 'depth_return_x_before_fc' not in batch:
                batch['depth_return_x_before_fc'] = False
            if 'need_img_extraction' not in batch:
                batch['need_img_extraction'] = True
            return self.img_embedding(batch['rgb_inputs'], batch['depth_inputs'], batch['img_mod'], batch['depth_return_x_before_fc'], batch['proj'], batch['process_images'], batch['need_img_extraction'])

        elif mode == "txt_embedding":
            text_embeds, txt_masks, text_cls_embeds = self.instruction_encoder(
                batch['instr_inputs'],
                use_qformer=False
            )
            return text_embeds

        elif mode == "update_rnn":
            return self.update_rnn_states(batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks'])
        
        elif mode == "pred_actions":
            device = batch['observations']['instruction'].device
            batch_size = batch['observations']['instruction'].shape[0]
            if 'num_sample' not in batch:
                batch['num_sample'] = 1
            if 'need_img_extraction' in batch and batch['need_img_extraction']:
                if self.model_config.IMAGE_ENCODER.use_stack:
                    input_rgb = batch['observations']['stack_rgb']
                    input_depth = batch['observations']['stack_depth']  
                else:
                    if 'rgb' in batch['observations'].keys():
                        input_rgb = batch['observations']['rgb']
                    else:
                        input_rgb = batch['observations']['rgb_features']
                    input_depth = batch['observations']['depth']
                
                if 'rgb_features' in batch['observations'].keys():
                    need_rgb_extraction = False
                else:
                    need_rgb_extraction = True
                
                if batch['train_cls_free_guidance'] and self.model_config.Diffusion_Policy.random_mask_rgb:
                    cls_free_mask = torch.rand(batch_size) < self.model_config.Diffusion_Policy.cls_mask_ratio
                    cls_free_mask = cls_free_mask.to(device)
                    input_rgb[cls_free_mask] = torch.zeros_like(input_rgb[cls_free_mask])
                    input_depth[cls_free_mask] = torch.zeros_like(input_depth[cls_free_mask])
                
                if analysis_time:
                    start_time = time.time()
                stack_rgb, stack_depth = self.img_embedding(input_rgb, input_depth, batch['img_mod'], batch['depth_return_x_before_fc'], batch['proj'], batch['process_images'], need_rgb_extraction)
                if analysis_time:
                    end_time = time.time()
                    print(f"MODEL img_embedding time: {end_time - start_time}")
                if len(stack_rgb.shape) == 2:
                    batch['observations']['stack_rgb'] = stack_rgb.unsqueeze(1)
                    batch['observations']['stack_depth'] = stack_depth.unsqueeze(1)
                else:
                    batch['observations']['stack_rgb'] = stack_rgb
                    batch['observations']['stack_depth'] = stack_depth
            else:
                if batch['train_cls_free_guidance'] and self.model_config.Diffusion_Policy.random_mask_rgb and self.model_config.Diffusion_Policy.cls_mask_method == 'mask_inputs':
                    cls_free_mask = torch.rand(batch_size) < self.model_config.Diffusion_Policy.cls_mask_ratio
                    cls_free_mask = cls_free_mask.to(device)
                    batch['observations']['stack_rgb'][cls_free_mask] = torch.zeros_like(batch['observations']['stack_rgb'][cls_free_mask])
                    batch['observations']['stack_depth'][cls_free_mask] = torch.zeros_like(batch['observations']['stack_depth'][cls_free_mask])

            return self.pred_actions(batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks'], batch['add_noise_to_action'], batch['denoise_action'], batch['num_sample'], batch['train_cls_free_guidance'], batch['sample_cls_free_guidance'], batch['need_txt_extraction'], analysis_time)

        elif mode == "act":
            return self.act(batch)
