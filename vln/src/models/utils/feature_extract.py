from typing import Any, Dict, List
import torch


def extract_instruction_tokens(
    observations: List[Dict],
    bert_tokenizer = None,
    is_clip_long=False,
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    for i in range(len(observations)):
        if bert_tokenizer is None:# TODO
            observations[i]['instruction'] = observations[i]['instruction']["tokens"]
        else:
            # use bert tokenizer
            if is_clip_long:
                tokens = bert_tokenizer(observations[i]['instruction'])[0].tolist()
            else:
                tokens = bert_tokenizer.text_token(observations[i]['instruction'])['input_ids'][0].tolist()
            observations[i]['instruction'] = tokens
    return observations

def extract_image_features(policy, batch, img_mod, len_traj_act=4, world_size=1, stack_rgb=None, stack_depth=None, depth_encoder_type='TAC', save_img_raw=False,
                           batch_stack_rgb_length=None, proj=True, net_device=None):
    """Extracts image features from observations using the policy's image feature extractor."""
    device = batch['globalgps'].device
    bs = batch['globalgps'].shape[0]
    
    if device.type == 'cpu' and net_device is not None:
        device = net_device

    if world_size > 1:
        net = policy.module
    else:
        net = policy

    if stack_rgb is None and stack_depth is None:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        
        rgb_feat = net.image_encoder.process_image(rgb).float().to(device)
        if depth_encoder_type == 'TAC':
            depth_feat = net.image_encoder.process_depth(depth).float().to(device)
        elif depth_encoder_type == 'resnet':
            depth_feat = depth.type(torch.float32)
        
        batch_inputs = {
            'mode': 'img_embedding',
            'rgb_inputs': rgb_feat,
            'depth_inputs': depth_feat,
            'depth_return_x_before_fc': True, # For ResNet depth encoder,
            'img_mod': img_mod,
            'proj': proj,
            'process_images': False
        }
        rgb_features, depth_features = policy(batch_inputs)
        rgb_features = rgb_features.type(torch.float32)
        depth_features = depth_features.type(torch.float32)

        # if img_mod == 'cls':
        #     batch['stack_rgb'] = rgb_features.unsqueeze(1)
        #     batch['stack_depth'] = depth_features.unsqueeze(1)
        # elif img_mod == 'multi_patches_avg_pooling':
        #     batch['stack_rgb'] = rgb_features
        #     batch['stack_depth'] = depth_features
        batch['stack_rgb'] = rgb_features
        batch['stack_depth'] = depth_features
    else:
        rgb = stack_rgb # shape: [bs, len_traj_act, 224, 224, 3]
        # depth = stack_depth.repeat(1,1,1,1,3)
        depth = stack_depth

        # rgb_length = rgb.shape[1]

        rgb = rgb.view(-1, rgb.shape[-2], rgb.shape[-3], rgb.shape[-1])
        depth = depth.view(-1, depth.shape[-2], depth.shape[-3], depth.shape[-1])

        rgb_feat = net.image_encoder.process_image(rgb).float().to(device)
        if depth_encoder_type == 'TAC':
            depth_feat = net.image_encoder.process_depth(depth).float().to(device)
        elif depth_encoder_type == 'resnet':
            depth_feat = depth.type(torch.float32)
        
        batch_inputs = {
            'mode': 'img_embedding',
            'rgb_inputs': rgb_feat,
            'depth_inputs': depth_feat,
            'depth_return_x_before_fc': True, # For ResNet depth encoder
            'img_mod': img_mod,
            'proj': proj
        }
        rgb_features, depth_features = policy(batch_inputs)
        rgb_features = rgb_features.type(torch.float32)
        depth_features = depth_features.type(torch.float32)

        # padding to the len_traj_act
        if img_mod == 'cls':
            dim = 512 if proj else 768
            rgb_features = rgb_features.reshape(bs, len_traj_act, dim)
        elif img_mod == 'multi_patches_avg_pooling':
            multi_patch_num = rgb_features.shape[-2]
            rgb_features = rgb_features.reshape(bs, len_traj_act, multi_patch_num,rgb_features.shape[-1])
        depth_features = depth_features.reshape(bs, len_traj_act, depth_features.shape[-3], depth_features.shape[-2], depth_features.shape[-1])

        if batch_stack_rgb_length is None:
            batch['stack_rgb'] = rgb_features
            if depth_encoder_type == 'TAC':
                batch['stack_depth'] = depth_features.reshape(bs, -1, depth_features.shape[-1])
            elif depth_encoder_type == 'resnet':
                batch['stack_depth'] = depth_features.reshape(bs, -1, depth_features.shape[-3], depth_features.shape[-2], depth_features.shape[-1])
        
        else:
            for env_idx in range(bs):
                pad_len = len_traj_act - batch_stack_rgb_length[env_idx]
                if img_mod == 'cls':
                    pad_tensor = torch.zeros(pad_len, rgb_features.shape[-1]).to(device)
                elif img_mod == 'multi_patches_avg_pooling':
                    pad_tensor = torch.zeros(pad_len, rgb_features.shape[-2],rgb_features.shape[-1]).to(device)
                rgb_features[env_idx, batch_stack_rgb_length[env_idx]:] = pad_tensor

                if depth_encoder_type == 'TAC':
                    pad_tensor = torch.zeros(pad_len, depth_features.shape[-1]).to(device)
                    depth_features[env_idx, batch_stack_rgb_length[env_idx]:] = pad_tensor
                elif depth_encoder_type == 'resnet':
                    pad_tensor = torch.zeros(pad_len, depth_features.shape[-3], depth_features.shape[-2], depth_features.shape[-1]).to(device)
                    depth_features[env_idx, batch_stack_rgb_length[env_idx]:] = pad_tensor
            batch['stack_rgb'] = rgb_features
            batch['stack_depth'] = depth_features   

    if not save_img_raw:
        if 'rgb' in batch.keys():
            del batch['rgb']
        if 'depth' in batch.keys():
            del batch['depth']

    return batch