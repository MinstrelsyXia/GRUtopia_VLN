def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        load_from_pretrain: bool = False,
    ) -> None:
        default_gpu, n_gpu, device = set_cuda(self.config)
        if default_gpu:
            logger.info(
                'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(
                    device, n_gpu, bool(self.config.local_rank != -1), self.config.fp16
                )
            )
        
        seed = self.config.seed
        if self.config.local_rank != -1:
            seed += self.config.rank
        set_random_seed(seed)
        
        if hasattr(self.config.MODEL, 'Diffusion_Policy'):
            self.action_stats = {}
            action_stats = self.config.MODEL.Diffusion_Policy.action_stats
            for key in action_stats:
                self.action_stats[key] = torch.from_numpy(np.array(action_stats[key])).to(device)
        
        if default_gpu:
            save_training_meta(self.config)
        
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)

        self.policy = policy(
            config=self.config,
            observation_space=observation_space,
            action_space=action_space,
            action_stats=self.action_stats,
        )
            
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.IL.lr
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
            
            self.policy.load_state_dict(new_ckpt_weights, strict=False)       
        
        start_epoch = 0
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
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
                    
            incompatible_keys, _= self.policy.load_state_dict(new_state_dict,
                                        strict=False)
            if len(incompatible_keys) > 0:
                logger.warning(f"Incompatible keys: {incompatible_keys}")
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params / 1e6:.2f}M. Trainable: {params_t / 1e6:.2f}M")
        logger.info("Finished setting up policy.")
        
        if len(self.config.TORCH_GPU_IDS) == 1:
            self.config.defrost()
            self.config.DDP.use = False
            self.config.freeze()
        if self.config.DDP.use:
            if self.config.local_rank != -1:
                self.policy = wrap_model(self.policy, self.config.TORCH_GPU_IDS[0], self.config.local_rank, self.config.world_size)
            else:
                self.policy = wrap_model(self.policy, self.config.TORCH_GPU_IDS, self.config.local_rank, self.config.world_size)
        else:
            self.policy.to(self.device)
        
        return start_epoch
    