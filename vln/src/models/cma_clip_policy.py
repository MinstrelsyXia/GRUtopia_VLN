from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gym import Space
import copy

import vln.src.models.encoders as encoders

from vln.src.models.encoders import resnet_encoders

from vln.src.models.encoders.instruction_encoder import (
    InstructionEncoder,
)

from transformers import PretrainedConfig
    
class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)

class CustomFixedCategorical(torch.distributions.Categorical):
    """Same as the CustomFixedCategorical in hab-lab, but renames log_probs
    to log_prob. All the torch distributions use log_prob.
    """

    def sample(
        self, sample_shape=torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CMA_CLIP_Net(nn.Module):
    """An implementation of the cross-modal attention (CMA) network in
    https://arxiv.org/abs/2004.02857
    """

    def __init__(
        self, config, observation_space, action_stats=None, num_actions=4
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.model_config = config.MODEL

        # Init instruction encoder
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
        self.rgb_proj_linear = nn.Linear(self.model_config.IMAGE_ENCODER.RGB.feature_dim, self.model_config.IMAGE_ENCODER.RGB.projection_dim)
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.model_config.IMAGE_ENCODER.RGB.projection_dim,
                self.model_config.IMAGE_ENCODER.RGB.projection_dim,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod((192,4,4)),
                self.model_config.IMAGE_ENCODER.RGB.projection_dim,
            ),
            nn.ReLU(True),
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = self.model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        # Init the RNN state decoder
        rnn_input_size = self.model_config.IMAGE_ENCODER.RGB.projection_dim
        rnn_input_size += self.model_config.IMAGE_ENCODER.RGB.projection_dim
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = encoders.build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=self.model_config.STATE_ENCODER.hidden_size,
            rnn_type=self.model_config.STATE_ENCODER.rnn_type,
            num_layers=1
        )

        self._output_size = (
            self.model_config.STATE_ENCODER.hidden_size
            + self.model_config.STATE_ENCODER.hidden_size # RGB
            + self.model_config.STATE_ENCODER.hidden_size # DEPTH
            + self.model_config.TEXT_ENCODER.hidden_size # TEXT
        )

        # cross-attn for RGB, depth, and instruction
        bert_config = PretrainedConfig.from_pretrained('data/pretrained/roberta')
        cross_modal_config = copy.deepcopy(bert_config)
        for k,v in self.model_config.CROSS_MODAL_ENCODER.items():
            setattr(cross_modal_config, k, v)
        
        self.state_txt_cross_encoder = encoders.VisionLanguageEncoder(cross_modal_config)
        self.txt_rgb_cross_encoder = encoders.VisionLanguageEncoder(cross_modal_config)
        self.txt_depth_cross_encoder = encoders.VisionLanguageEncoder(cross_modal_config)
        self.depth_k_linear = nn.Linear(192, self.model_config.TEXT_ENCODER.hidden_size)
        
        # second rnn
        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )
        
        self.second_state_encoder = encoders.build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=self.model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )
        self._output_size = self.model_config.STATE_ENCODER.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()
        
        # Determine
        self.action_distribution = CategoricalNet(
            self._output_size, self.num_actions
        )

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

    def _init_layers(self) -> None:
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

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

    def _forward(
        self,
        batch,
        observations: Dict[str, Tensor],
        rnn_states: Tensor, # [bs, 2, 512]
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # instruction_embedding = self.instruction_encoder(observations)
        # depth_embedding = self.depth_encoder(observations)
        # depth_embedding = torch.flatten(depth_embedding, 2) # [bs, 192, 16]

        # rgb_embedding = self.rgb_encoder(observations)
        # rgb_embedding = torch.flatten(rgb_embedding, 2) # [bs, 2112, 16]
        
                
        # 1. instruction embedding
        instruction_embedding, txt_masks, text_cls_embeds = self.instruction_encoder(
            observations['instruction'], need_txt_extraction=True
        )
        
        # 2. rgb & depth embedding
        rgb_features, depth_features = self.img_embedding(observations['rgb'], observations['depth'], batch['img_mod'], batch['depth_return_x_before_fc'], batch['proj'], batch['process_images'], need_rgb_extraction=True)
        # rgb_features: [bs, 5, dim]
        # depth_features: [bs, 192, 4, 4]
        
        rgb_features = self.rgb_proj_linear(rgb_features) # 768 -> 512
        
        rgb_embedding = rgb_features.permute(0, 2, 1) # [bs, 5, dim] -> [bs, dim, 5]
        depth_embedding = torch.flatten(depth_features, 2) # [bs, 192, 4, 4] -> [bs, 192, 16]

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
        
        do_self_attn = True
        # 1. Q: state. KV: text.
        text_embedding, _ = self.state_txt_cross_encoder(state.unsqueeze(1), instruction_embedding, q_masks=masks, kv_masks=txt_masks, output_attentions=True,do_self_attn=do_self_attn)
        
        # 2. Q: text. KV: rgb.
        rgb_embedding, _ = self.txt_rgb_cross_encoder(text_embedding, rgb_features, q_masks=masks, kv_masks=None, output_attentions=True,do_self_attn=do_self_attn)
        rgb_embedding = rgb_embedding[:,0,:]
        
        # 3. Q: text. KV: depth.
        depth_k_embedding = self.depth_k_linear(depth_embedding.permute(0, 2, 1))
        depth_embedding, _ = self.txt_depth_cross_encoder(text_embedding, depth_k_embedding, q_masks=masks, kv_masks=None, output_attentions=True,do_self_attn=do_self_attn)
        depth_embedding = depth_embedding[:, 0, :]
        
        text_embedding = text_embedding.squeeze(1)

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

        progress_hat = None
        if self.model_config.PROGRESS_MONITOR.use:
            progress_hat = torch.tanh(self.progress_monitor(x))
            # progress_loss = F.mse_loss(
            #     progress_hat.squeeze(1),
            #     observations["progress"],
            #     reduction="none",
            # )

        return x, rnn_states_out, progress_hat

    def build_distribution(
        self, observations, rnn_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_states = self.forward(
            observations, rnn_states, prev_actions, masks
        )
        return self.action_distribution(features)
    
    def forward(self, batch):
        x, rnn_states_out, progress_hat = self._forward(batch, batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks'])
        # distribution = self.action_distribution(x) # This would meet the error when using DataParallel during training "TypeError: 'CustomFixedCategorical' object is not iterable"
        if batch['mode'] == 'train':
            outputs = self.action_distribution(x).logits
        elif batch['mode'] == 'inference':
            outputs = self.action_distribution(x).mode()
        return outputs, rnn_states_out, progress_hat

