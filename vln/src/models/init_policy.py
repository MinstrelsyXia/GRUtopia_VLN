import os,sys
import torch
import numpy as np
from gym import spaces

from vln.src.models.misc import set_random_seed, set_dropout, set_cuda, wrap_model
from vln.src.models.save import save_training_meta, load_checkpoint

def get_policy(policy_name):
    if policy_name == 'CMA_DP_Policy':
        # TODO
        from vln.src.models.cma_dp_policy import CMA_DP_Net
        return CMA_DP_Net
    elif policy_name == 'CMA_DP_ImgMultiPatch_Policy':
        from vln.src.models.cma_dp_policy_ImgMultiPatch import CMA_DP_Net
        return CMA_DP_Net
    elif policy_name == "CMA_Policy":
        from vln.src.models.cma_policy import CMANet
        return CMANet
    elif policy_name == 'DP_noRNN_Policy':
        from vln.src.models.dp_policy_noRNN import CMA_DP_noRNN_Net
        return CMA_DP_noRNN_Net
    else:
        raise ValueError(f"Policy {policy_name} not found")

def initialize_policy(
        config,
        logger,
        load_from_ckpt,
        device,
        load_from_pretrain: bool = False,
        action_stats = None,
    ) -> None:
        default_gpu, n_gpu, device = set_cuda(config, device)
        if default_gpu:
            logger.info(
                'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(
                    device, n_gpu, bool(config.local_rank != -1), config.fp16
                )
            )
        
        seed = config.seed
        if config.local_rank != -1:
            seed += config.rank
        set_random_seed(seed)

        # if default_gpu:
        #     save_training_meta(config)
        
        observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(256,256,1),
                dtype=np.float32,
            )
        
        policy = get_policy(config.MODEL.policy_name)

        self_policy = policy(
            config=config,
            observation_space=observation_space,
            action_stats=action_stats,
        )
            
        optimizer = torch.optim.Adam(
            self_policy.parameters(), lr=float(config.IL.lr)
        )
        if load_from_pretrain:
            new_ckpt_weights = {}
            model_config = config.MODEL
            if model_config.TEXT_ENCODER.model_name == 'meter':
                tmp = torch.load(model_config.TEXT_ENCODER.model_path)
                tmp = tmp['state_dict']
                
                for param_name, param in tmp.items():
                    if 'text_transformer.embeddings' in param_name:
                        param_name = param_name.replace('text_transformer', 'net.instruction_encoder')
                        new_ckpt_weights[param_name] = param
                    elif 'text_transformer.encoder' in param_name:
                        param_name = param_name.replace('text_transformer.encoder', 'net.instruction_encoder')
                        new_ckpt_weights[param_name] = param
                    elif 'cross_modal_image_layers' in param_name:
                        if model_config.CROSS_MODAL_ENCODER.load_model:
                            param_name = param_name.replace('cross_modal_image_layers', 'net.cross_modal_encoder.cross_modal_encoder.crossattention')
                            new_ckpt_weights[param_name] = param
                    else:
                        new_ckpt_weights[param_name] = param
                del tmp
            
            if model_config.IMAGE_ENCODER.DEPTH.bottleneck == 'TAC':
                tmp_depth = torch.load(model_config.IMAGE_ENCODER.DEPTH.model_path)
                for param_name, param in tmp_depth.items():
                    if 'vision_model' in param_name:
                        param_name = param_name.replace('vision_model', 'net.image_encoder.depth_transformer.vision_model')
                        new_ckpt_weights[param_name] = param
                
                del tmp_depth
            
            if model_config.IMAGE_ENCODER.RGB.model_name == 'clip-vit-base-patch32':
                tmp_rgb = torch.load(os.path.join(model_config.IMAGE_ENCODER.RGB.model_path, 'pytorch_model.bin'))
                for param_name, param in tmp_rgb.items():
                    if 'vision_model' in param_name:
                        param_name = param_name.replace('vision_model', 'net.image_encoder.image_transformer.vision_model')
                        new_ckpt_weights[param_name] = param
                
                del tmp_rgb
            
            self_policy.load_state_dict(new_ckpt_weights, strict=False)       
        
        start_epoch = 0
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = load_checkpoint(ckpt_path, map_location="cpu")
            state_dict = ckpt_dict['state_dict']
            new_state_dict = {}
            # Iterate through the state dictionary items
            for k, v in state_dict.items():
                # Check if the key includes 'module.'
                if 'module.' in k:
                    # Handle the key by stripping 'module.' if necessary or perform any required operation
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v
            del state_dict[k]  # Remove the old key with 'module.'
                    
            incompatible_keys, _= self_policy.load_state_dict(new_state_dict,
                                        strict=False)
            if len(incompatible_keys) > 0:
                logger.warning(f"Incompatible keys: {incompatible_keys}")
            # if config.IL.is_requeue:
            #     optimizer.load_state_dict(ckpt_dict["optim_state"])
            #     start_epoch = start_epoch = ckpt_dict["epoch"] + 1
            #     step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self_policy.parameters())
        params_t = sum(
            p.numel() for p in self_policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params / 1e6:.2f}M. Trainable: {params_t / 1e6:.2f}M")
        logger.info("Finished setting up policy.")
        
        if len(config.TORCH_GPU_IDS) == 1:
            config.DDP.use = False
        if config.DDP.use:
            if config.DDP.use_dp:
                # Data parallel
                self_policy = wrap_model(
                    self_policy,
                    config.TORCH_GPU_IDS,
                    config.local_rank,
                    logger,
                    config.world_size,
                    use_dp=config.DDP.use_dp,
                ) 
            else:
                # Distributed data parallel
                self_policy = wrap_model(
                    self_policy,
                    torch.device(f"cuda:{config.local_rank}"),
                    config.local_rank,
                    logger,
                    config.world_size,
                    use_dp=config.DDP.use_dp,
                )
        else:
            self_policy.to(device)
        
        return self_policy, optimizer
    