import torch
import torch.nn as nn
import sys
import os
import math
import clip
import numpy as np
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig
from vlnce_baselines.config.default import get_config
import torch.nn.functional as F
from torchvision.transforms import Resize
from copy import deepcopy

from .lora import LinearWithLoRA, MultiheadAttnWithLoRA
from vlnce_baselines.models.LongCLIP.model import longclip
from functools import partial

from .bert_backbone import extend_neg_masks

class InstructionLongCLIPEncoder(nn.Module):
    def __init__(self, config, lora_config=None):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_text_encoder

        self.text_transformer, _ = longclip.load(config.model_path)
        
        # del visual part
        del self.text_transformer.visual
        
        if not self.update_lang_bert:
            for name, param in self.text_transformer.named_parameters():
                param.requires_grad = False
        
        if lora_config is not None and lora_config.add_for_instruction_encoder:
            assign_lora = partial(LinearWithLoRA, rank=lora_config.lora_r, alpha=lora_config.lora_alpha)
            assign_attn_lora = partial(MultiheadAttnWithLoRA, rank=lora_config.lora_r, alpha=lora_config.lora_alpha)
            lora_start_layer = lora_config.lora_start_layer if lora_config.lora_start_layer<len(self.text_transformer.transformer.resblocks) else 0
            for layer_idx, layer in enumerate(self.text_transformer.transformer.resblocks):
                if layer_idx >= lora_start_layer:
                    # if lora_config.lora_query:
                    #     # since CLIP use the nn.MultiheadAttention, we need to assign LoRA to the in_proj_weight (for all query, key, and values)
                    #     layer.attn = assign_attn_lora(layer.attn) # TODO cannot directly wrap the attn
                    if lora_config.lora_mlp:
                        layer.mlp = assign_lora(layer.mlp)
            
            print("Add LoRA for instruction encoder")

    def forward(self, txt_inputs, txt_masks=None):
        txt_inputs = txt_inputs.long()
        # padding the length of text to 248
        if txt_inputs.size(1) < 248:
            txt_inputs = F.pad(txt_inputs, (0, 248 - txt_inputs.size(1)), value=0)
        if txt_masks is None:
            txt_masks = (txt_inputs != 0).to(txt_inputs.device)
        txt_cls_embeds, txt_full_embeds = self.text_transformer.encode_text(txt_inputs, return_full=True)
        txt_cls_embeds = txt_cls_embeds.type(torch.float32)
        txt_full_embeds = txt_full_embeds.type(torch.float32)
        return txt_full_embeds, txt_masks, txt_cls_embeds