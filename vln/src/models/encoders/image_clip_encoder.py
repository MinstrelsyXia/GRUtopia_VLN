import torch
import torch.nn as nn
import sys
import os
import math
import clip
import numpy as np
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig
import torch.nn.functional as F
from torchvision.transforms import Resize, ToPILImage
from copy import deepcopy
from PIL import Image

from vln.src.models.encoders import resnet_encoders

from .bert_backbone import PositionalEncoding
from .lora import LinearWithLoRA
from vln.src.models.LongCLIP.model import longclip
from functools import partial

class WrapModule(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        res = self.model(pixel_values)
        if len(res.shape) == 2:
            res = res.unsqueeze(1)
        return {"last_hidden_state": res}


class ImageEncoder(torch.nn.Module):
    def __init__(self, full_config, config, observation_space, lora_config=None, test=False):
        super().__init__()

        self.config = config

        # RGB image model
        self.is_clip_long = False
        if config.RGB.model_name == 'clip-long':
            self.image_transformer, self.image_processor = longclip.load(config.RGB.model_path)
            # del text part
            del self.image_transformer.token_embedding
            del self.image_transformer.transformer
            del self.image_transformer.positional_embedding
            del self.image_transformer.ln_final
            self.is_clip_long = True
            self.to_pil = ToPILImage()
            
        else:
            self.image_transformer_config = CLIPVisionConfig.from_pretrained(
                config.RGB.model_path
            )
            self.image_transformer = CLIPVisionModel(self.image_transformer_config)
                
            self.image_processor = CLIPImageProcessor.from_pretrained(
                config.RGB.model_path
            )
        self.image_feature_dim = config.RGB.feature_dim
        self.image_projection_dim = config.RGB.projection_dim
        self.image_fc = torch.nn.Linear(
            self.image_feature_dim, self.image_projection_dim, bias=False
        )

        # Depth model
        if config.DEPTH.bottleneck == "TAC":
            depth_config = CLIPVisionConfig()
            self.depth_transformer = CLIPVisionModel(config=depth_config)
            self.depth_processor = CLIPImageProcessor.from_pretrained(
                config.DEPTH.model_name
            )
            # Normalize
            self.depth_mean = torch.nn.Parameter(
                torch.tensor([0.48145466, 0.4578275, 0.40821073]), requires_grad=False
            )
            self.depth_std = torch.nn.Parameter(
                torch.tensor([0.26862954, 0.26130258, 0.27577711]), requires_grad=False
            )
            self.depth_mean_numpy = self.depth_mean.detach().numpy()
            self.depth_std_numpy = self.depth_std.detach().numpy()
            
            self.resize_trans = Resize(224)
        
        elif config.DEPTH.bottleneck == 'resnet':
            self.depth_encoder = getattr(
                resnet_encoders, config.DEPTH.cnn_type
            )(
                observation_space,
                output_size=config.DEPTH.output_size,
                checkpoint=config.DEPTH.ddppo_checkpoint,
                backbone=config.DEPTH.backbone,
                trainable=config.DEPTH.update_depth_encoder,
                spatial_output=True,
            )
            self.depth_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.depth_encoder.output_shape),
                    config.DEPTH.feature_dim,
                ),
                nn.ReLU(True),
            )
        
        # position embedding
        self.pos_embedding = PositionalEncoding(config.RGB.projection_dim, max_seq_len=config.img_stack_nums)
        
        self.layernorm = nn.LayerNorm(config.RGB.projection_dim)

        # image & depth linear
        self.img_learnable_linear = nn.Linear(config.RGB.feature_dim, config.RGB.projection_dim)
        self.img_ln = nn.LayerNorm(config.RGB.projection_dim)
        self.depth_learnable_linear = nn.Linear(config.DEPTH.feature_dim, config.DEPTH.projection_dim)
        self.depth_ln = nn.LayerNorm(config.DEPTH.projection_dim)

        # Dropout layers
        self.env_drop = nn.Dropout(config.env_drop)
        self.dropout = nn.Dropout(config.dropout)

        # Set trainable
        for param in self.image_transformer.parameters():
            param.requires_grad_(config.RGB.update_rgb_encoder)
        if config.DEPTH.bottleneck == "TAC":
            for param in self.depth_transformer.parameters():
                param.requires_grad_(config.DEPTH.update_depth_encoder)
        
        # add lora
        self.lora_config = lora_config
        if lora_config is not None and lora_config.add_for_rgb_encoder:
            self.add_lora(self.image_transformer)
            print("Add LoRA for RGB encoder")
        if config.DEPTH.bottleneck == "TAC":
            if lora_config is not None and lora_config.add_for_depth_encoder:
                self.add_lora(self.depth_transformer)
                print("Add LoRA for depth encoder")

        # Init params
        self.init_param()
    
    def process_image(self, image_inputs):
        if len(image_inputs.shape) == 5:
            # bs, stack_num, 224, 224, 3
            image_size = image_inputs.shape[2]
            # image_inputs = image_inputs.reshape(-1, 3, image_size, image_size)
            image_inputs = image_inputs.reshape(-1,image_inputs.shape[-3],image_inputs.shape[-2],image_inputs.shape[-1])
        if self.is_clip_long:
            if len(image_inputs.shape)==3 and image_inputs.shape[-1] == 3:
                # convert H,W,C to C,H,W
                image_inputs = image_inputs.permute(2,0,1)
                image_Image = self.to_pil(image_inputs)
                image_feat = np.array(self.image_processor(image_Image))
                image_feat = torch.from_numpy(np.array(image_feat)).to(image_inputs.device)
            
            elif len(image_inputs.shape) == 4 and image_inputs.shape[-1] == 3:
                # convert B,H,W,C to B,C,H,W
                image_inputs = image_inputs.permute(0,3,1,2)
                image_feat = []
                for image in image_inputs:
                    image = self.to_pil(image.cpu())
                    image_feat.append(np.array(self.image_processor(image)))
                image_feat = np.array(image_feat)
                image_feat = torch.from_numpy(image_feat).to(image_inputs.device)

        else:
            image_feat = self.image_processor(image_inputs, do_resize=False, do_center_crop=False, return_tensors="pt").pixel_values
        
        if len(image_inputs.shape) == 5:
            image_feat = image_feat.reshape(image_inputs.shape[0], image_inputs.shape[1], -1)
        return image_feat
    
    def _normalize(self, depth_batch):
        """Simplified process function"""
        if isinstance(depth_batch, np.ndarray):
            # Handle NumPy array
            depth_mean = self.depth_mean_numpy
            depth_std = self.depth_std_numpy
      
            if depth_batch.shape[-1] == 1:
                depth_batch = np.repeat(depth_batch, repeats=3, axis=-1)  # Expand to last dimension with 3 copies
            depth_batch = (depth_batch - depth_mean) / depth_std
            
            if len(depth_batch.shape) == 4:
                depth_batch = np.transpose(depth_batch, (0, 3, 1, 2))  # Permute dimensions for NumPy
            elif len(depth_batch.shape) == 3:
                depth_batch = np.transpose(depth_batch, (2,0,1))  # Permute dimensions for NumPy 
            
        elif isinstance(depth_batch, torch.Tensor):
            # Handle PyTorch tensor
            device = depth_batch.device
            depth_mean = self.depth_mean.to(device)
            depth_std = self.depth_std.to(device)
        
            if len(depth_batch.shape) == 4:
                if depth_batch.shape[-1] == 1:
                    depth_batch = depth_batch.expand(-1, -1, -1, 3)  # Expand to last dimension with 3
                depth_batch = (depth_batch - depth_mean) / depth_std
                depth_batch = depth_batch.permute(0, 3, 1, 2)  # Permute dimensions for PyTorch
            elif len(depth_batch.shape) == 3:
                if depth_batch.shape[-1] == 1:
                    depth_batch = depth_batch.expand(-1, -1, 3)  # Expand to last dimension with 3
                depth_batch = (depth_batch - depth_mean) / depth_std
                depth_batch = depth_batch.permute(2,0,1)  # Permute dimensions for PyTorch
        
        assert depth_batch.shape[-1] == 224
        # depth_batch = self.resize_trans(depth_batch) # Here is a bug that needs the Image class to handle..
        
        return depth_batch

    
    def process_depth(self, depth_inputs):
        if len(depth_inputs.shape) == 2:
            depth_inputs = np.expand_dims(depth_inputs, axis=2).repeat(3, axis=2)
        # depth_feat = self.depth_processor(depth_inputs, return_tensors="pt").pixel_values
        depth_feat = self._normalize(depth_inputs)
        return depth_feat
        
    def embed_image(self, image_batch, fc=False, max_batch_size=400, img_mod='cls', proj=True):
        """Embed a batch of image."""
        if len(image_batch.shape) == 3:
            image_batch = image_batch.unsqueeze(0)
        
        BS = image_batch.shape[0]
        if len(image_batch.shape) == 5:
            image_batch = image_batch.reshape(-1, 3, image_batch.shape[3], image_batch.shape[4]) # [BS, T, 224, 224, 3] -> [BS*T, 3, 224, 224]
        
        embeddings = []
        # Process in chunks if the batch size exceeds the limit
        for i in range(0, image_batch.shape[0], max_batch_size):
            batch_subset = image_batch[i:i + max_batch_size]
            if self.is_clip_long:
                if img_mod == 'cls':
                    # return [bs, 768]
                    outputs = self.image_transformer.encode_image(batch_subset, proj=proj)
                elif img_mod == 'multi_patches_avg_pooling':
                    # return [bs, 5, 768]. 0 is CLS token, and the last 4 are average pooling tokens.
                    outputs = self.image_transformer.encode_image_multi_patches(batch_subset)
            else:
                outputs = self.image_transformer(pixel_values=batch_subset).pooler_output
            embeddings.append(outputs)
        
        # Concatenate outputs from all the chunks
        if img_mod == 'cls':
            outputs = torch.cat(embeddings, dim=0).reshape(BS, -1, outputs.shape[-1])
        elif img_mod == 'multi_patches_avg_pooling':
            outputs = torch.cat(embeddings, dim=0).float() # convert float16 -> 32

        if fc:
            outputs = self.image_fc(outputs)
        return outputs
    
    def embed_depth(self, input, return_x_before_fc=False):
        if self.config.DEPTH.bottleneck == 'resnet':
            outputs = self.embed_depth_resnet(input, return_x_before_fc=return_x_before_fc)
            if return_x_before_fc:
                outputs = outputs[0] # [bs, 128, 4, 4]
        elif self.config.DEPTH.bottleneck == 'TAC':
            outputs = self.embed_depth_TAC(input, fc=False)
        return outputs
    
    def embed_depth_resnet(self, depth, return_x_before_fc=False):
        # set return_x_before_fc to be True when collect dataset (the same as CMA)
        BS = depth.shape[0]
        reshape_flag = False
        if len(depth.shape) == 5:
            # stack depth: [BS, T, 224, 224, 1]
            depth = depth.flatten(0,1)
            reshape_flag = True
        batch = {'depth': depth}
        outputs = self.depth_encoder(batch, return_x_before_fc=return_x_before_fc)
        if reshape_flag:
            new_outputs = []
            for output in outputs:
                new_outputs.append(output.reshape(BS, -1, *output.shape[1:]))
            outputs = new_outputs
        return outputs

    def embed_depth_TAC(self, depth_batch, fc=False, max_batch_size=500):
        """Embed a batch of depth."""
        BS = depth_batch.shape[0]
        if len(depth_batch.shape) == 5:
            depth_batch = depth_batch.reshape(-1, 3, depth_batch.shape[3], depth_batch.shape[4])
        
        if BS > max_batch_size:
            embeddings = []
            # Process in chunks if the batch size exceeds the limit
            for i in range(0, BS, max_batch_size):
                batch_subset = depth_batch[i:i + max_batch_size]
                # outputs = self.depth_transformer(pixel_values=batch_subset).pooler_output
                outputs = self.depth_transformer(pixel_values=batch_subset, output_hidden_states=True)
                outputs = outputs['hidden_states'][-2][:,0,:]
                embeddings.append(outputs)
            
            # Concatenate outputs from all the chunks
            outputs = torch.cat(embeddings, dim=0).reshape(BS, -1, outputs.shape[-1])
        else:
            outputs = self.depth_transformer(pixel_values=depth_batch, output_hidden_states=True)
            outputs = outputs['hidden_states'][-2][:,0,:]
        
        if fc:
            outputs = self.depth_fc(outputs)
        return outputs
    
    def add_lora(self, model):
        assign_lora = partial(LinearWithLoRA, rank=self.lora_config.lora_r, alpha=self.lora_config.lora_alpha)
        for layer_idx, layer in enumerate(model.vision_model.encoder.layers):
            if layer_idx >= self.lora_config.lora_start_layer:
                if self.lora_config.lora_start_layer:
                    layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
                if self.lora_config.lora_key:
                    layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
                if self.lora_config.lora_value:
                    layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
                if self.lora_config.lora_projection:
                    layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
                if self.lora_config.lora_mlp:
                    layer.mlp.fc1 = assign_lora(layer.mlp.fc1)
                    layer.mlp.fc2 = assign_lora(layer.mlp.fc2)

    def init_param(self):
        pass

    def clamp_param(self):
        self.temperature.data.clamp_(-2, 5)
        self.time_scale.data.clamp_(1 / 20, 1)
    
    def forward(self, rgb_inputs, depth_inputs, fc=False, 
                do_process=False, do_embeds=False,
                prev_action_embeds=None,
                use_stack=False,
                img_mod='cls'):
        batch_size = rgb_inputs.shape[0]
        if do_process:
            rgb_inputs = self.process_image(rgb_inputs)
            depth_inputs = self.process_depth(depth_inputs)
        if do_embeds:
            image_embeddings = self.embed_image(rgb_inputs, fc=fc)
            depth_embeddings = self.embed_depth(depth_inputs, fc=fc)
        else:
            image_embeddings = rgb_inputs
            depth_embeddings = depth_inputs
        
        if not use_stack and len(image_embeddings.shape) == 4:
            # This will meet at the update_dataset time
            image_embeddings = image_embeddings[:,0,:]
            depth_embeddings = depth_embeddings[:,0,:]
        
        if self.config.use_env_drop:
            # directly use dropout on the raw features
            image_embeddings = self.env_drop(image_embeddings)
            # depth_embeddings = self.env_drop(depth_embeddings)
        
        if self.config.DEPTH.bottleneck == 'resnet':
            if use_stack:
                stack_lens = depth_embeddings.shape[1]
                depth_embeddings = depth_embeddings.reshape(-1, 128, 4, 4)
            depth_resnet_inputs = {'depth_features': depth_embeddings}
            depth_embeds = self.depth_encoder(depth_resnet_inputs) # [bs,128,4,4]
            depth_embeds = torch.flatten(depth_embeds, 2) # [bs, 192, 16]
            depth_embeddings = self.depth_linear(depth_embeds)
            if use_stack:
                depth_embeddings = depth_embeddings.reshape(batch_size, stack_lens, -1)

        image_embeddings = self.dropout(self.img_learnable_linear(image_embeddings))
        depth_embeddings = self.dropout(self.depth_learnable_linear(depth_embeddings))
        
        if img_mod == 'cls':
            img_depth_embeds = image_embeddings + depth_embeddings
            if prev_action_embeds is not None and use_stack:
                img_depth_embeds = img_depth_embeds + prev_action_embeds
                            
            img_depth_embeds = self.layernorm(img_depth_embeds)
            
        elif img_mod == 'multi_patches_avg_pooling':
            # 20241025: combine the depth with the full rgb embeds at the 0-pth location.
            ## 0-th location: full depth+img. 2~5: semantic rgb.
            if use_stack:
                image_embeddings[:,:,0,:] = image_embeddings[:,:,0,:] + depth_embeddings[:,:,]
            else:
                image_embeddings[:,0,:] = image_embeddings[:,0,:] + depth_embeddings
            img_depth_embeds = image_embeddings     
        
        if use_stack:
            img_depth_pos_embeds = self.pos_embedding(img_depth_embeds)
            return img_depth_pos_embeds
        else:
            if img_mod == 'cls':
                return img_depth_embeds.unsqueeze(1)
            elif img_mod == 'multi_patches_avg_pooling':
                return img_depth_embeds

    
def convert_weights_float(model: nn.Module):
    """Convert applicable model parameters back to fp32"""

    def _convert_weights_to_fp32(md):
        if isinstance(md, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            md.weight.data = md.weight.data.float()
            if md.bias is not None:
                md.bias.data = md.bias.data.float()

        if isinstance(md, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(md, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(md, name):
                attr = getattr(md, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)

class CLIPResEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        rgb_level: int = -1,
    ) -> None:
        super().__init__()
        self.model, self.preprocessor = clip.load(model_name)
        self.rgb_embedding_seq = None
        self.model.visual.attnpool.register_forward_hook(self._vit_hook)
        self.sub_embedding_seq = None
        convert_weights_float(self.model)
        self.model.train()

    def _vit_hook(self, m, i, o):
        self.rgb_embedding_seq = o.float()

    def encode_image(self, pixel_values):
        rgb_observations = pixel_values
        _ = self.model.encode_image(rgb_observations).float()
        # LND -> NLD
        rgb_embedding_seq = self.rgb_embedding_seq.float().unsqueeze(1)
        return {"last_hidden_state": rgb_embedding_seq}

    def forward(self, pixel_values):
        return self.encode_image(pixel_values=pixel_values)

if __name__ == '__main__':
    config = get_config("/ssd/wangliuyi/code/VLN-CE/vlnce_baselines/config/r2r_baselines/cma_dp_w61.yaml")
    model_config = config.MODEL.IMAGE_ENCODER

    model = TACEncoder(model_config)
    ckpt = torch.load("/ssd/wangliuyi/code/VLN-CE/data/pretrained/TAC/best.pth")
    model.load_state_dict(ckpt["state_dict"], strict=False)


    depth_encoder = model.depth_transformer
    torch.save(depth_encoder.state_dict(), "tac_depth_encoder.pth")

    from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig
    config = CLIPVisionConfig()
    depth_encoder = CLIPVisionModel(config=model_config)
    ckpt = torch.load("tac_depth_encoder.pth")
    depth_encoder.load_state_dict(ckpt)
    depth_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")