from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
import copy
from transformers import PretrainedConfig
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import matplotlib.pyplot as plt

from habitat import Config
from copy import deepcopy
from habitat_baselines.common.baseline_registry import baseline_registry
# from habitat_baselines.rl.models.rnn_state_encoder import (
#     build_rnn_state_encoder,
# )
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion 

from vlnce_baselines.common.aux_losses import AuxLosses
# from vlnce_baselines.models.encoders import resnet_encoders
# from vlnce_baselines.models.encoders.instruction_encoder import (
#     InstructionEncoder,
# )
from vlnce_baselines.models.policy import ILPolicy

import vlnce_baselines.models.encoders as encoders

from vlnce_baselines.models.utils import get_delta, get_action, get_data_stats, normalize_data, unnormalize_data, action_reduce

action_spaces = {
    'stop': [0],
    'go_forward': [1],
    'turn_left': [2],
    'turn_right': [3],
    'wait': [4]
}

@baseline_registry.register_policy
class CMA_DP_Policy(ILPolicy):
    def __init__(
        self,
        config: Config,
        observation_space: Space,
        action_space: Space,
        action_stats: Dict,
    ) -> None:
        self.config = config
        self.model_config = config.MODEL
        self.action_stats = action_stats
        # action_num = action_space.n
        if self.model_config.learn_angle:
            action_num = 3
        else:
            action_num = 2
        super().__init__(
            CMA_DP_Net(
                observation_space=observation_space,
                model_config=self.model_config,
                num_actions=action_num,
                action_stats=action_stats
            ),
            action_num,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space, batch_size: int
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
            batch_size=batch_size
        )
    
    def forward(self, batch):
        if batch['mode'] in ['img_embedding', 'pred_actions', 'update_rnn']:
            return self.net(batch)
        elif batch['mode'] == 'act':
            return self.act(batch)

    def parse_action(self, diffusion_output, dist_pred, pm_pred=None, stop_mode='distance', steps=None):
        cumsum = False if self.config.EVAL.ACTION == 'descrete' else True
        if self.model_config.learn_angle:
            un_actions = get_action(diffusion_output, self.action_stats, cumsum=cumsum)
            un_actions_nocumsum = get_action(diffusion_output, self.action_stats, cumsum=False)
            actions_cumsum = get_action(diffusion_output, self.action_stats, cumsum=True)
        else:
            un_actions = diffusion_output
            un_actions_nocumsum, actions_cumsum = un_actions, None

        un_actions = un_actions.detach().cpu().numpy()

        if self.config.EVAL.ACTION == 'xyyaw':
            actions = []
            for idx in range(un_actions.shape[0]):
                # if dist_pred[idx].item() < 1e-1 or (un_actions[0] < 1e-1 and un_actions[1] < 1e-1):
                value_sum = 0
                for value in un_actions[idx][-1]:
                    value_sum += abs(value)
                if stop_mode == 'distance':
                    if dist_pred[idx].item() < self.config.EVAL.distance_threshold or\
                            (abs(un_actions[idx][0][0]) < 1e-1 and abs(un_actions[idx][0][1]) < 1e-1 and abs(un_actions[idx][step_idx][2]) < 1e-1):
                        # stop
                        actions.append({"action": "STOP"})
                        continue
                elif stop_mode == 'progress':
                    if pm_pred[idx].item() > self.config.EVAL.pm_threshold or\
                            (abs(un_actions[idx][0][0]) < 1e-1 and abs(un_actions[idx][0][1]) < 1e-1 and abs(un_actions[idx][0][2]) < 1e-1):
                        # stop
                        actions.append({"action": "STOP"})
                        continue
                actions.append(
                    {
                        "action": {
                            "action": "GO_TOWARD_XYYAW",
                            "action_args": {
                                "actions": un_actions[idx],
                            },
                        }
                    }
                )
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
                            # steps = None # !!! 
                            # if steps is not None:
                            #     stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                            #     (steps[bs_idx] > 5 and abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1 and abs(un_actions[bs_idx][step_idx][2]) < 1e-1)
                            # else:
                            # stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                            #     (abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1 and abs(un_actions[bs_idx][step_idx][2]) < 1e-1)
                            stop_flag = (pm_pred[bs_idx].item() > self.config.EVAL.pm_threshold) or\
                                (abs(diffusion_output[bs_idx][step_idx][0]) < 3e-1 and abs(diffusion_output[bs_idx][step_idx][1]) < 3e-1 and abs(diffusion_output[bs_idx][step_idx][2]) < 3e-1)

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

    def act(self, batch):
        observations = batch['observations']
        rnn_states = batch['rnn_states']
        prev_actions = batch['prev_actions']
        masks = batch['masks']
        add_noise_to_action = batch['add_noise_to_action']
        denoise_action = batch['denoise_action']
        batch['mode'] = 'pred_actions'
        
        batch_size = rnn_states.shape[0]
        vis = batch['vis']
        step = batch['step']
        episode_ids = batch['episode_ids']

        noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_pred = self.forward(batch)

        # prev_actions = diffusion_output[:,:self.model_config.len_traj_act]
        if batch['denoise_action'] and batch['num_sample'] > 1:         
            diffusion_output_split = torch.split(diffusion_output, batch['num_sample'], dim=0)
            if dist_pred is not None:
                dist_pred_split = torch.split(dist_pred, batch['num_sample'], dim=0)
            else:
                dist_pred_split = None
            rnn_states_split = torch.split(rnn_states_out, batch['num_sample'], dim=0)
            actions = []
            un_actions_nocumsum = []
            rnn_states_list = []
            for i in range(batch_size):
                dp_output = diffusion_output_split[i]
                fix, ax = plt.subplots(1, 1)
                actions_list = []
                un_actions_nocumsum_list = []
                for traj_id, traj in enumerate(dp_output):
                    traj = traj.unsqueeze(0)
                    cand_actions, cand_actions_cumsum, cand_un_actions_nocumsum = self.parse_action(traj, dist_pred_split[i][traj_id], pm_pred=progress_pred[i][traj_id], stop_mode=batch['stop_mode'], steps=batch['steps'])
                    actions_list.append(cand_actions)
                    un_actions_nocumsum_list.append(cand_un_actions_nocumsum)
                    if vis:
                        ax.plot(cand_actions_cumsum[:, 0].detach().cpu(), cand_un_actions_nocumsum[:, 1].detach().cpu(), alpha=0.1, marker='o')
                
                if vis:
                    # save images
                    save_file = f'data/images/num_samples/EpisodeId_{episode_ids[i]}_step_{step}.png'
                    plt.savefig(save_file)
                    print(f"Save image to {save_file}")
                    
                    plt.close()
                
                # randomly sample one from list
                actions.append(actions_list[np.random.randint(0, len(actions_list))][0])
                un_actions_nocumsum.append(un_actions_nocumsum_list[np.random.randint(0, len(un_actions_nocumsum_list))])
            
            rnn_states_out = torch.stack([x[0] for x in rnn_states_split], dim=0)
            
        else:
            actions, actions_cumsum, un_actions_nocumsum = self.parse_action(diffusion_output, dist_pred, pm_pred=progress_pred, stop_mode=batch['stop_mode'], steps=batch['steps'])
        
        return actions, rnn_states_out, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum, progress_pred

class CMA_DP_Net(Net):
    """An implementation of the cross-modal attention (CMA) network in
    https://arxiv.org/abs/2004.02857
    Modified by the Diffusion Policy
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int, 
        action_stats=None
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.num_actions = num_actions
        self.action_stats = action_stats
        
        model_config.defrost()
        model_config.TEXT_ENCODER.final_state_only = False
        # Note that I use TEXT_ENCODER to represent the instruction encoder rather than the original INSTRUCTION_ENCODER
        model_config.freeze()
        
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
        self.image_encoder = encoders.ImageEncoder(self.model_config, self.model_config.IMAGE_ENCODER, observation_space, self.model_config.LORA)
        
        # Init the cross-modal fusion network
        bert_config = PretrainedConfig.from_pretrained('roberta-base')
        cross_modal_config = copy.deepcopy(bert_config)
        for k,v in self.model_config.CROSS_MODAL_ENCODER.items():
            setattr(cross_modal_config, k, v)
        self.cross_modal_encoder = encoders.VisionLanguageEncoder(cross_modal_config)
        
        # Init the prev action embedding
        if model_config.IMAGE_ENCODER.use_stack:
            prev_action_encoder_size = model_config.IMAGE_ENCODER.RGB.feature_dim
        else:
            prev_action_encoder_size = model_config.PREV_ACTION_ENCODER.encoding_size
        self.prev_action_embedding = nn.Linear(num_actions, prev_action_encoder_size)
        self.prev_act_ln = nn.LayerNorm(self.model_config.PREV_ACTION_ENCODER.encoding_size)
        self.prev_action_pos_embedding = encoders.PositionalEncoding(self.model_config.PREV_ACTION_ENCODER.encoding_size, self.model_config.len_traj_act)
        
        # Init the step embedding
        if model_config.STEP_ENCODER.use:
            self.step_embeddings = nn.Embedding(self.model_config.STEP_ENCODER.max_steps, self.model_config.STEP_ENCODER.encoding_size)
            self.step_embeddings_dp = nn.Embedding(self.model_config.STEP_ENCODER.max_steps, model_config.STATE_ENCODER.hidden_size)

        # concat_size = model_config.IMAGE_ENCODER.RGB.feature_dim +\
        #     self.model_config.PREV_ACTION_ENCODER.encoding_size*model_config.len_traj_act
        
        if self.model_config.IMAGE_ENCODER.RGB.img_mod == 'cls':
            concat_size = model_config.IMAGE_ENCODER.RGB.projection_dim
        elif self.model_config.IMAGE_ENCODER.RGB.img_mod == 'multi_patches_avg_pooling':
            if self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'flat':
                concat_size = model_config.IMAGE_ENCODER.RGB.projection_dim * model_config.IMAGE_ENCODER.RGB.multi_patches_num
            elif self.model_config.STATE_ENCODER.rgb_depth_embed_method == 'first':
                concat_size = model_config.IMAGE_ENCODER.RGB.projection_dim
                
        # Init the IMU encoder
        if model_config.IMU_ENCODER.use:
            self.imu_linear = nn.Linear(model_config.IMU_ENCODER.input_size, model_config.IMU_ENCODER.encoding_size)
            
            concat_size += model_config.IMU_ENCODER.encoding_size
        
        if not model_config.IMAGE_ENCODER.use_stack:
            concat_size += self.model_config.PREV_ACTION_ENCODER.encoding_size*model_config.len_traj_act
            # concat_size += self.model_config.PREV_ACTION_ENCODER.encoding_size
        
        # self.state_compress_linear = nn.Linear(concat_size, model_config.STATE_ENCODER.hidden_size)
        
        # Init the GRU network
        self.state_encoder = encoders.build_rnn_state_encoder(
            input_size=concat_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=model_config.STATE_ENCODER.num_layers
        )
        
        # Init the diffusion policy network
        self.use_local_cond = model_config.Diffusion_Policy.use_local_cond
        local_cond_dim = model_config.STATE_ENCODER.hidden_size if self.use_local_cond else None

        self.global_cond_linear = nn.Linear(model_config.TEXT_ENCODER.hidden_size, model_config.Diffusion_Policy.encoding_size)
        
        self.dp_type = model_config.Diffusion_Policy.type
        if model_config.Diffusion_Policy.type == 'resnet_unet':
            self.action_dp_pred_net = ConditionalUnet1D(
                    input_dim=num_actions,
                    local_cond_dim=local_cond_dim,
                    global_cond_dim=model_config.Diffusion_Policy.encoding_size,
                    down_dims=model_config.Diffusion_Policy.down_dims,
                    diffusion_step_embed_dim=model_config.Diffusion_Policy.diffusion_step_embed_dim,
                    cond_predict_scale=model_config.Diffusion_Policy.cond_predict_scale
                )
        elif model_config.Diffusion_Policy.type == 'transformer':
            # define the length of conditions
            if model_config.IMAGE_ENCODER.use_stack:
                vis_length = self.model_config.len_traj_act
            else:
                vis_length = 1
            
            txt_length = self.model_config.TEXT_ENCODER.max_length
            rnn_length = 1

            if self.model_config.Diffusion_Policy.cond == 'rnn_instr_vis':
                n_obs_steps = rnn_length + 1 + vis_length
            elif self.model_config.Diffusion_Policy.cond == 'rnn_instr':
                n_obs_steps = rnn_length + 1
            elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr_vis':
                n_obs_steps = rnn_length + txt_length + vis_length
            elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr':
                n_obs_steps = rnn_length + txt_length

            self.action_dp_pred_net = TransformerForDiffusion(
                    input_dim=num_actions,
                    output_dim=num_actions,
                    horizon=model_config.Diffusion_Policy.len_traj_pred,
                    n_obs_steps=n_obs_steps,
                    n_emb=model_config.Diffusion_Policy.transformer_encoding_size,
                    cond_dim=model_config.Diffusion_Policy.transformer_encoding_size,
                    causal_attn=True,
                    time_as_cond=True,
                    n_head=self.model_config.Diffusion_Policy.transformer_n_head,
                    n_layer=self.model_config.Diffusion_Policy.transformer_n_layers,
                    n_cond_layers=self.model_config.Diffusion_Policy.transformer_n_cond_layers
                )
        
        if model_config.Diffusion_Policy.scheduler == 'DDPM':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=model_config.Diffusion_Policy.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type=model_config.Diffusion_Policy.pred_type
            )
        elif model_config.Diffusion_Policy.scheduler == 'DDIM':
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=model_config.Diffusion_Policy.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type=model_config.Diffusion_Policy.pred_type
            )
            self.noise_scheduler.set_timesteps(model_config.Diffusion_Policy.num_diffusion_iters)
       
        # Init the distance prediction network
        if self.model_config.DISTANCE_PREDICTOR.use:
            self.distance_pred_net = encoders.DistanceNetwork(
                embedding_dim=model_config.STATE_ENCODER.hidden_size, normalize=model_config.DISTANCE_PREDICTOR.normalize)

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
        if model_config.PROGRESS_MONITOR.use:
            self.progress_monitor = encoders.DistanceNetwork(
            embedding_dim=model_config.STATE_ENCODER.hidden_size, 
            normalize=True) # pm_pred 

            self._init_pm_layers()
        
        self._output_size = num_actions

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

    def _init_pm_layers(self) -> None:
        if self.model_config.PROGRESS_MONITOR.use:
            # nn.init.kaiming_normal_(
            #     self.progress_monitor.weight, nonlinearity="tanh"
            # )
            # nn.init.constant_(self.progress_monitor.bias, 0)
            for param in self.progress_monitor.parameters():
                if param.ndim == 2:  # Typically weights are 2D
                    nn.init.kaiming_normal_(param, nonlinearity="tanh")
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
    
    def pred_actions(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
        add_noise_to_action=True,
        denoise_action=False
    ):
        device = observations['instruction'].device
        batch_size = observations['instruction'].shape[0]
        
        '''1. Encoding text'''
        text_embeds, txt_masks, text_cls_embeds = self.instruction_encoder(
            observations['instruction']
        ) 
                
        '''2. Encoding previous actions'''
        prev_actions_masks = prev_actions.float() * masks.unsqueeze(-1).float()
        prev_action_embeds = self.prev_action_embedding(prev_actions_masks)
        if not self.model_config.IMAGE_ENCODER.use_stack:
            # prev_action_embeds = self.prev_action_pos_embedding(prev_action_embeds)
            # prev_action_embeds = prev_action_embeds[:,0,:]
            prev_action_embeds = prev_action_embeds.reshape(batch_size, -1) # use the stacked prev_action_embeds!
        
        '''3. Encoding images'''
        rgb_depth_embeds = self.image_encoder(observations['stack_rgb'], observations['stack_depth'], prev_action_embeds=prev_action_embeds, use_stack=self.model_config.IMAGE_ENCODER.use_stack)
        
        '''4. Encoding vision-and-language''' 
        if not self.model_config.IMAGE_ENCODER.use_stack:
            do_self_attn = False
        else:
            do_self_attn = True
        fused_cross_modal_embeds, attention_probs = self.cross_modal_encoder(rgb_depth_embeds, text_embeds, txt_masks, output_attentions=True,do_self_attn=do_self_attn)
        attention_probs = attention_probs[:,0,:]
        
        '''5. Encoding IMU'''
        if self.model_config.IMU_ENCODER.use:
            imu_embeds = self.imu_linear(observations['imu'])
            
            # 6. Concat features
            concat_embeds = torch.cat([fused_cross_modal_embeds, imu_embeds], dim=1)
        
        if not self.model_config.IMAGE_ENCODER.use_stack:
            concat_embeds = torch.cat([concat_embeds, prev_action_embeds], dim=1)

        '''7. Compute GRU features'''
        state, rnn_states_out = self.state_encoder(concat_embeds, rnn_states, masks.bool())
        
        '''8. Predict action distribution using diffusion policy'''
        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        noise = noise_pred = diffusion_output = None
        if denoise_action:
            # initialize action from Gaussian noise
            noisy_diffusion_output = torch.randn(
                (batch_size, self.model_config.Diffusion_Policy.len_traj_pred, self.num_actions), device=device)
            noise = deepcopy(noisy_diffusion_output)
            diffusion_output = noisy_diffusion_output
            for k in self.noise_scheduler.timesteps[:]:
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
                    noise_pred = self.action_dp_pred_net(
                        sample=diffusion_output, 
                        timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                        cond=lv_state.float())
                    
                elif self.dp_type == 'resnet_unet':
                    if self.use_local_cond:
                        # use image rnn as local_condition
                        local_cond = state.unsqueeze(1).expand(-1, self.model_config.Diffusion_Policy.len_traj_pred, -1).float()
                        # use text features as global_condition
                        # Here I use the image-text cross-attention to get the global_condition
                        global_cond = torch.mul(text_embeds, attention_probs.unsqueeze(-1)).sum(1)
                        global_cond = self.global_cond_linear(global_cond)

                        noise_pred = self.action_dp_pred_net(
                            sample=diffusion_output, 
                            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                            local_cond=local_cond,
                            global_cond=global_cond)
                    else:
                        noise_pred = self.action_dp_pred_net(
                            sample=diffusion_output, 
                            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                            global_cond=state.float())

                # inverse diffusion step (remove noise)
                diffusion_output = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=diffusion_output
                ).prev_sample
            
        else:
            if add_noise_to_action:
                # Add noise to the clean images according to the noise magnitude at each diffusion iterationd
                # Sample noise to add to actions
                # deltas = get_delta(observations['actions'])
                # naction = normalize_data(deltas, self.action_stats, device)
                # naction = from_numpy(ndeltas).to(device)            
                naction = observations['actions'] # which has been normalized
                noise = torch.randn(naction.shape, device=device)
                noisy_action = self.noise_scheduler.add_noise(
                    naction, noise, timesteps)
            else:
                noisy_action = observations['actions']
            
            if self.dp_type == 'transformer':
                # Predict the noise residual
                if self.model_config.Diffusion_Policy.cond == 'rnn_instr_vis':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1), rgb_depth_embeds), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_instr':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1)), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr':
                    lv_state = torch.cat((state.unsqueeze(1), text_cls_embeds.unsqueeze(1)), dim=1)
                elif self.model_config.Diffusion_Policy.cond == 'rnn_Fullinstr_vis':
                    lv_state = torch.cat((state.unsqueeze(1), text_embeds, rgb_depth_embeds), dim=1)
                noise_pred = self.action_dp_pred_net(
                    sample=noisy_action.float(), 
                    timestep=timesteps,
                    cond=lv_state.float())
                
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
            dist_pred = self.distance_pred_net(state)
        
        progress_pred = None
        # if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
        if self.model_config.PROGRESS_MONITOR.use:
            # progress_pred = torch.tanh(self.progress_monitor(state)) # pm_pred 
            progress_pred = self.progress_monitor(state)

        return noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_pred

    def update_rnn_states(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor
    ):
        device = observations['instruction'].device
        batch_size = observations['instruction'].shape[0]
        
        '''1. Encoding text'''
        text_embeds, txt_masks, text_cls_embeds = self.instruction_encoder(
            observations['instruction']
        ) 
                
        '''2. Encoding previous actions'''
        prev_actions_masks = prev_actions.float() * masks.unsqueeze(-1).float()
        prev_action_embeds = self.prev_action_embedding(prev_actions_masks)
        if not self.model_config.IMAGE_ENCODER.use_stack:
            # prev_action_embeds = self.prev_action_pos_embedding(prev_action_embeds)
            # prev_action_embeds = prev_action_embeds[:,0,:]
            prev_action_embeds = prev_action_embeds.reshape(batch_size, -1) # use the stacked prev_action_embeds!
        
        '''3. Encoding images'''
        rgb_depth_embeds = self.image_encoder(observations['stack_rgb'], observations['stack_depth'], prev_action_embeds=prev_action_embeds, use_stack=self.model_config.IMAGE_ENCODER.use_stack)
        
        '''4. Encoding vision-and-language''' 
        if not self.model_config.IMAGE_ENCODER.use_stack:
            do_self_attn = False
        else:
            do_self_attn = True
        fused_cross_modal_embeds, attention_probs = self.cross_modal_encoder(rgb_depth_embeds, text_embeds, txt_masks, output_attentions=True,do_self_attn=do_self_attn)
        attention_probs = attention_probs[:,0,:]
        
        '''5. Encoding IMU'''
        if self.model_config.IMU_ENCODER.use:
            imu_embeds = self.imu_linear(observations['imu'])
            
            # 6. Concat features
            concat_embeds = torch.cat([fused_cross_modal_embeds, imu_embeds], dim=1)
        
        if not self.model_config.IMAGE_ENCODER.use_stack:
            concat_embeds = torch.cat([concat_embeds, prev_action_embeds], dim=1)

        '''7. Compute GRU features'''
        state, rnn_states_out = self.state_encoder(concat_embeds, rnn_states, masks.bool())
        return state, rnn_states_out

    def img_embedding(self, rgb_inputs, depth_inputs, img_mod, depth_return_x_before_fc=False):
        rgb_embeds = self.image_encoder.embed_image(rgb_inputs,img_mod=img_mod).squeeze(1)
        depth_embeds = self.image_encoder.embed_depth(depth_inputs, return_x_before_fc=depth_return_x_before_fc).squeeze(1)
        return rgb_embeds, depth_embeds
            
    def forward(
        self, batch
    ) -> Tuple[Tensor, Tensor]:
        mode = batch['mode']
        if mode == "img_embedding":
            if 'depth_return_x_before_fc' not in batch:
                batch['depth_return_x_before_fc'] = False
            return self.img_embedding(batch['rgb_inputs'], batch['depth_inputs'], batch['img_mod'], batch['depth_return_x_before_fc'])
        
        elif mode == "pred_actions":   
            return self.pred_actions(batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks'], batch['add_noise_to_action'], batch['denoise_action'])
        
        elif mode == "update_rnn":
            return self.update_rnn_states(batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks'])