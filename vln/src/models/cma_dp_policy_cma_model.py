from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
import copy
from transformers import PretrainedConfig
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from habitat import Config
from copy import deepcopy
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion 

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

import vlnce_baselines.models.encoders as encoders

from vlnce_baselines.models.utils import get_delta, get_action, get_data_stats, normalize_data, unnormalize_data, action_reduce

action_spaces = {
    'stop': [0],
    'go_forward': [1],
    'turn_left': [2],
    'turn_right': [3]
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
        return self.net(batch)

    def act(self, batch):
        observations = batch['observations']
        rnn_states = batch['rnn_states']
        prev_actions = batch['prev_actions']
        masks = batch['masks']
        add_noise_to_action = batch['add_noise_to_action']
        denoise_action = batch['denoise_action']

        noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_hat = self.forward(batch)

        # prev_actions = diffusion_output[:,:self.model_config.len_traj_act]

        cumsum = False if self.config.EVAL.ACTION == 'descrete' else True
        un_actions = get_action(diffusion_output, self.action_stats, cumsum=cumsum)
        un_actions_nocumsum = get_action(diffusion_output, self.action_stats, cumsum=False)

        un_actions = un_actions.detach().cpu().numpy()

        if self.config.EVAL.ACTION == 'xyyaw':
            actions = []
            for idx in range(un_actions.shape[0]):
                # if dist_pred[idx].item() < 1e-1 or (un_actions[0] < 1e-1 and un_actions[1] < 1e-1):
                value_sum = 0
                for value in un_actions[idx][-1]:
                    value_sum += abs(value)
                if dist_pred[idx].item() < 10:
                    actions.append({"action": "STOP"})
                else:
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
                for step_idx in range(un_actions.shape[1]):
                    if dist_pred[bs_idx].item() < self.config.EVAL.distance_threshold:
                        # stop
                        actions[bs_idx].append(action_spaces['stop'])
                        continue
                    if abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1:
                        # turn left or turn right
                        if un_actions[bs_idx][step_idx][2] > 0:
                            actions[bs_idx].append(action_spaces['turn_right'])
                        elif un_actions[bs_idx][step_idx][2] < 0:
                            actions[bs_idx].append(action_spaces['turn_left'])
                    else:
                        # move forward
                        actions[bs_idx].append(action_spaces['go_forward'])
        
        return actions, rnn_states_out, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum

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
        
        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
            spatial_output=True,
        )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = model_config.DEPTH_ENCODER.output_size
        rnn_input_size += model_config.RGB_ENCODER.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self._output_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )
        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)
        
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
            self.action_dp_pred_net = TransformerForDiffusion(
                    input_dim=num_actions,
                    output_dim=num_actions,
                    horizon=model_config.Diffusion_Policy.len_traj_pred,
                    n_obs_steps=1, # !!!
                    n_emb=model_config.Diffusion_Policy.transformer_encoding_size,
                    cond_dim=model_config.Diffusion_Policy.transformer_encoding_size,
                    causal_attn=True,
                    time_as_cond=True,
                    n_layer=self.model_config.Diffusion_Policy.transformer_n_layers,
                    n_cond_layers=self.model_config.Diffusion_Policy.transformer_n_cond_layers
                )
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_config.Diffusion_Policy.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
       
        # Init the distance prediction network
        self.distance_pred_net = encoders.DistanceNetwork(
            embedding_dim=model_config.STATE_ENCODER.hidden_size, normalize=model_config.DISTANCE_PREDICTOR.normalize)

        # self._output_size = model_config.STATE_ENCODER.hidden_size
        if model_config.PROGRESS_MONITOR.use:
            self.progress_monitor = nn.Linear(self.output_size, 1)

            self._init_pm_layers()
        
        self._output_size = num_actions

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

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
        add_noise_to_action: True,
        denoise_action: False
    ):
        device = observations['instruction'].device
        batch_size = observations['instruction'].shape[0]
        
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state, # 512
                text_embedding, # 256
                rgb_embedding, # 256
                depth_embedding, # 128
                prev_actions, # 32
            ],
            dim=1,
        ) 
        x = self.second_state_compress(x) # 512
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x, # [B, 512]
            rnn_states[:, self.state_encoder.num_recurrent_layers :], # [5, 1, 512]
            masks, # [B, 512]
        )
        
        '''Predict action distribution using diffusion policy'''
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
                    noise_pred = self.action_dp_pred_net(
                        sample=diffusion_output, 
                        timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                        cond=x)
                    
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
                # Add noise to the clean images according to the noise magnitude at each diffusion iteration
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
                noise_pred = self.action_dp_pred_net(
                    sample=noisy_action.float(), 
                    timestep=timesteps,
                    cond=x)
                
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
        dist_pred = self.distance_pred_net(state)
        
        progress_hat = None
        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(state))
            # progress_loss = F.mse_loss(
            #     progress_hat.squeeze(1),
            #     observations["progress"],
            #     reduction="none",
            # )
            # AuxLosses.register_loss(
            #     "progress_monitor",
            #     progress_loss,
            #     self.model_config.PROGRESS_MONITOR.alpha,
            # )

        return noise_pred, dist_pred, rnn_states_out, noise, diffusion_output, progress_hat

    def img_embedding(self, rgb_inputs, depth_inputs):
        rgb_embeds = self.image_encoder.embed_image(rgb_inputs).squeeze(1)
        depth_embeds = self.image_encoder.embed_depth(depth_inputs).squeeze(1)
        return rgb_embeds, depth_embeds
            
    def forward(
        self, batch
    ) -> Tuple[Tensor, Tensor]:
        mode = batch['mode']
        if mode == "img_embedding":
            return self.img_embedding(batch['rgb_inputs'], batch['depth_inputs'])
        
        elif mode == "pred_actions":
            return self.pred_actions(batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks'], batch['add_noise_to_action'], batch['denoise_action'])