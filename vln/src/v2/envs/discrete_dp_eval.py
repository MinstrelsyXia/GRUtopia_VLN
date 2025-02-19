import os, sys
from .base import BaseSingleScanEnv
from grutopia.core.config import SimulatorConfig
from vln.src.v2.dataloader.eval import EvalPathKeyDataloader
from vln.src.v2.util.common_log_util import common_logger as log
from vln.src.v2.util import progress_log_util
from vln.src.v2.util.eval import(
    get_obs,
    Statistic_Info,
    ActionExecutor,
)
from vln.src.models.utils.feature_extract import extract_instruction_tokens
import torch
from vln.src.utils.utils import batch_obs
import numpy as np
from vln.src.v2.util.stuck_checker import StuckChecker
from vln.src.models.init_policy import initialize_policy
from vln.src.v2.util.data_collector import DataCollector

from vln.src.models.LongCLIP.model import longclip
from vln.src.models.utils.bert_token import BertTokenizer

from vln.src.models.utils.feature_extract import extract_image_features
from vln.src.utils.utils import extract_best_eval_results, load_dataset, action_reduce, get_checkpoint_id, poll_checkpoint_folder, is_slurm_batch_job, batch_obs, FixedLengthStack, _compute_actions, get_delta, normalize_data, map_action_to_2d, save_video, get_action, to_local_coords

class DiscreteDPEvalSingleScanEnv(BaseSingleScanEnv):
    
    def __init__(
            self,
            robot_name,
            sim_config:SimulatorConfig,
            scene_asset_path,
            start_position,
            start_rotation,
            headless,
            dataloader:EvalPathKeyDataloader,
            eval_config,
            lmdb_path,
            ckpt_name,
        ):
        super().__init__(
            robot_name=robot_name,
            sim_config=sim_config,
            scene_asset_path=scene_asset_path,
            start_position=start_position,
            start_rotation=start_rotation,
            headless=headless,
        )
        self.dataloader=dataloader
        #TODO:
        self.device = torch.device("cuda", 0)
        self.eval_config = eval_config
        #TODO:
        self.per_action_max_step=1500
        self.max_step=25000
        self.lmdb_path = lmdb_path
        self.ckpt_name = ckpt_name
        
        # Init the action stats
        self.action_stats = None
        if hasattr(self.eval_config.MODEL, 'Diffusion_Policy'):
            self.action_stats = {
                'min': torch.Tensor(np.asarray(self.eval_config.MODEL.Diffusion_Policy.action_stats.min)).to(self.device),
                'max': torch.Tensor(np.asarray(self.eval_config.MODEL.Diffusion_Policy.action_stats.max)).to(self.device)
            }

        policy, _, _, _ = initialize_policy(
            self.eval_config,
            log,
            load_from_ckpt=True,
            device=self.device,
            load_from_pretrain=False,
            action_stats=self.action_stats,
        )
        self.policy = policy

        if self.eval_config.MODEL.policy_name in ["CMA_DP_ImgMultiPatch_Policy"]:
            self.use_clip_encoders = True
        else:
            self.use_clip_encoders = False
        
        self.use_bert = False
        self.bert_tokenizer = None
        self.is_clip_long = False

        if self.use_clip_encoders:
            if self.eval_config.MODEL.TEXT_ENCODER.type == 'roberta':
                self.bert_tokenizer = BertTokenizer(
                    max_length=self.eval_config.MODEL.INSTRUCTION_ENCODER.max_length,
                    load_model=self.eval_config.MODEL.INSTRUCTION_ENCODER.load_model,
                    device=self.device
                )
                self.use_bert = True
            elif self.eval_config.MODEL.TEXT_ENCODER.type == 'clip-long':
                self.bert_tokenizer = longclip.tokenize
                self.use_bert = True
                self.is_clip_long = True
        
        # other model settings
        self.action_dim = 3
        
        # mkdir for eval_dir
        self.EP_DIR = os.path.join('logs', self.eval_config.NAME)
        os.makedirs(self.EP_DIR, exist_ok=True)

    def topdown_snapshot(self):
        map_info = self.get_global_map(
            robot_height=self.robot_height,
        )
        camera_pose = self.topdown_global_map_camera.get_world_pose()[0] - self.task._offset
        height, width = self.topdown_global_map_camera._camera._resolution
        snapshot={
            "map_info":map_info,
            "camera_pose":camera_pose,
            #TODO:
            "aperture":500,
            "width":width,
            "height":height,
        }
        return snapshot

    def eval(self, test_verbose=False):
        
        self.load_scan_and_robot()
        eval_path_key_list = self.dataloader.eval_path_key_list
        path_key_data = self.dataloader.path_key_data
        path_key_split = self.dataloader.path_key_split
        scan = self.dataloader.target_scan
        progress_log_util.init(scan, len(eval_path_key_list), rank=self.dataloader.rank)
        progress_log_util.progress_logger.info(f"start eval scan: {scan}, total_path:{len(eval_path_key_list)}")
        robot_ankle_height = self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
        data_collector = DataCollector(self.dataloader.lmdb_path,self.dataloader.rank)

        self.policy.eval()
        env_num = 1
        for path_key in eval_path_key_list:
            split = path_key_split[path_key]
            data = path_key_data[path_key]
            log.info(f"split: {split}")
            log.info(f"scan: {scan}")
            log.info(f"trajectory_id_episode_id: {path_key}")
            log.info(f"data: {data}")
            progress_log_util.trace_start(
                trajectory_id = path_key,
                step_count=0,
            )
            start_position = data['start_position']
            start_rotation = data['start_rotation']
            self.reset_robot(start_position, start_rotation)
            self.warm_up(240)

            stuck_checker = StuckChecker(self.task._offset,self.isaac_robot)
            # map_info = self.topdown_snapshot()
            robot_position, robot_rotation = self.task.get_robot_poses_without_offset()

            if 'sub_instruction' in data:
                sub_instr = data['sub_instruction']
                sub_instr_tokens = data['sub_instruction_tokens']
                max_instr_len = 100
            else:
                sub_instr = None
                sub_instr_tokens = None
                max_instr_len = 248
            observations = get_obs(self.env, data['instruction'],robot_position,robot_rotation, sub_instr, sub_instr_tokens, robot_name=self.robot_name)
            start_positions = [x['globalgps'][[0,1]] for x in observations]
            start_positions = torch.from_numpy(np.stack(start_positions, axis=0)).to(self.device)
            start_yaws = [x['globalyaw'] for x in observations]
            start_yaws = torch.from_numpy(np.stack(start_yaws, axis=0)).to(self.device)

            observations = extract_instruction_tokens(
                observations, 
                #TODO:
                bert_tokenizer=self.bert_tokenizer,
                is_clip_long=self.is_clip_long,
                max_instr_len=max_instr_len
            )

            batch = batch_obs(observations, self.device)

            if self.eval_config.MODEL.IMAGE_ENCODER.use_stack:
                batch_stack_rgb_length = [1 for _ in range(len(batch))]
                h, w, c = batch['rgb'].shape[1:]
                batch_stack_rgb = torch.zeros(len(batch), self.eval_config.MODEL.IMAGE_ENCODER.img_stack_nums, h, w, c, device=self.device)
                batch_stack_rgb[:, 0, :, :, :] = batch['rgb']

                h, w, c = batch['depth'].shape[1:]
                batch_stack_depth = torch.zeros(len(batch), self.eval_config.MODEL.IMAGE_ENCODER.img_stack_nums, h, w, c, device=self.device)
                batch_stack_depth[:, 0, :, :, :] = batch['depth']

            else:
                batch_stack_rgb, batch_stack_depth, batch_stack_rgb_length = None, None, None

            classifier_free_mask_depth = self.eval_config.MODEL.Diffusion_Policy.use_cls_free_guidance and self.eval_config.MODEL.IMAGE_ENCODER.DEPTH.update_depth_encoder

            batch = extract_image_features(
                self.policy, batch, 
                img_mod=self.eval_config.MODEL.IMAGE_ENCODER.RGB.img_mod,
                len_traj_act=self.eval_config.MODEL.IMAGE_ENCODER.img_stack_nums,
                world_size=1,
                depth_encoder_type=self.eval_config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck,
                stack_rgb = batch_stack_rgb,
                stack_depth = batch_stack_depth,
                batch_stack_rgb_length = batch_stack_rgb_length,
                proj=self.eval_config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
                need_rgb_extraction=True,
                classifier_free_mask_depth=classifier_free_mask_depth,
                )
            
            if self.eval_config.MODEL.IMU_ENCODER.use:
                imu = torch.zeros(env_num, self.eval_config.MODEL.IMU_ENCODER.input_size, device=self.device)
                batch["imu"] = imu.float()

            batch["steps"] = torch.from_numpy(np.array([0])).to(self.device)
            env_nums = 1
            rnn_states = torch.zeros(
                env_nums,
                self.policy.num_recurrent_layers,
                self.eval_config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
            prev_actions = torch.zeros(
                env_nums, self.eval_config.MODEL.len_traj_act, self.action_dim, device=self.device, dtype=torch.long
            )
            not_done_masks = torch.zeros(
                env_nums,
                1, dtype=torch.uint8, device=self.device
            )

            # init fix_length_stack
            stack_rgb_length = self.eval_config.MODEL.IMAGE_ENCODER.img_stack_nums if self.eval_config.MODEL.IMAGE_ENCODER.use_stack else 1
            # stack_rgb_length = self.config.MODEL.len_traj_act
            stack_rgb = [FixedLengthStack(stack_rgb_length) for _ in range(env_num)]
            stack_depth = [FixedLengthStack(stack_rgb_length) for _ in range(env_num)]
            prev_globalgps = [FixedLengthStack(self.eval_config.MODEL.len_traj_act+1) for _ in range(env_num)] # TODO !!! act length
            prev_globalyaw = [FixedLengthStack(self.eval_config.MODEL.len_traj_act+1) for _ in range(env_num)]
            
            env_idx = 0
            # record the current position before action
            stack_rgb[env_idx].push(observations[env_idx]["rgb"])
            stack_depth[env_idx].push(observations[env_idx]["depth"])
            prev_globalgps[env_idx].push(batch[env_idx]['globalgps'].detach().cpu().numpy())
            prev_globalyaw[env_idx].push(batch[env_idx]['global_rotation'][-1].detach().cpu().item())

            statistic_info = Statistic_Info(
                env=self.env,
                path_data=data,
                shortest_path_length=data['info']['geodesic_distance'],
                shortest_to_goal_distance=999,
                step_interval=self.eval_config.EVAL.step_interval,
                success_distance=self.eval_config.EVAL.success_distance,
            )
            spl_dict = {}
            stats_episodes = {}

            steps = [0]
            stop_flag = False
            while True:
                if stop_flag:
                    break
                if len(spl_dict) > 0 and np.mean(list(spl_dict.values())) < 0.02 and len(stats_episodes) > 100:
                    # this ckpt is too bad to continue
                    log.info(f"Break. This ckpt is too bad to continue with average SPL {np.mean(list(spl_dict.values())):.3f}")
                    return
                if statistic_info.sim_step % 1000 == 0:
                    log.info(f"[split:{split}][scan:{scan}][trajectory_id_episode_id: {path_key}][step:{statistic_info.sim_step}]")
                
                batch_settings = {
                    'mode': 'act',
                    'observations': batch,
                    'rnn_states': rnn_states,
                    'prev_actions': prev_actions,
                    'masks': not_done_masks,
                    'add_noise_to_action': False,
                    'denoise_action': True,
                    'num_sample': self.eval_config.EVAL.num_sample,
                    'step': statistic_info.sim_step,
                    'episode_ids': path_key,
                    'stop_mode': self.eval_config.EVAL.stop_mode,
                    'steps': steps,
                    'predicted_actions_save_dir': self.EP_DIR,
                    'num_sample': self.eval_config.EVAL.num_sample,
                    'train_cls_free_guidance': False,
                    'sample_cls_free_guidance': self.eval_config.MODEL.Diffusion_Policy.use_cls_free_guidance,
                    'need_txt_extraction': True,
                    'vis': test_verbose,
                }

                with torch.no_grad():
                    actions, rnn_states, noise_pred, dist_pred, noise, diffusion_output, un_actions_nocumsum, pm_pred, stop_progress_pred = self.policy(batch_settings)

                # prev_actions.copy_(actions)
                actions = actions[env_idx]
                for exe_step_i in range(self.eval_config.EVAL.len_traj_act):
                    a = actions[exe_step_i]
                    if a[0] == 0:
                        log.info(f"[split:{split}][scan:{scan}][trajectory_id_episode_id: {path_key}][stop!!!]")
                        action = [
                            {self.robot_name: {'stop': ['stop']}}
                        ]
                    else:
                        action = [
                            {self.robot_name: {'move_by_descrete': a}}
                        ]
                    executor = ActionExecutor(
                        env=self.env, 
                        task=self.task, 
                        stuck_checker=stuck_checker,
                        robot=self.robot,

                        per_action_max_step=self.per_action_max_step,
                        total_max_step=self.max_step,
                        robot_ankle_height=robot_ankle_height,

                        statistic_info=statistic_info,
                        context=self,
                        
                        robot_name=self.robot_name,
                        fall_height_threshold=self.sim_config.config_dict['tasks'][0]['robots'][0]['fall_height_threshold']
                    )
                    outputs = executor.env_step(actions = action)
                    outputs_dict = outputs['outputs_dict']
                    dones = outputs['dones']
                    info = outputs['infos'][0]
                    reason = outputs['reason']
                    statistic_info = executor.statistic_info
                    statistic_info.policy_step +=1

                    if dones[0]:
                        stop_flag = True

                    if test_verbose:
                        import matplotlib.pyplot as plt
                        plt.imsave('logs/test.png', outputs_dict[0]['rgb'])

                    if 'sub_instruction' in data: # MLANet
                        max_instr_len = 100
                    else:
                        max_instr_len = 248

                    outputs_dict = extract_instruction_tokens(
                        outputs_dict,
                        #TODO: 
                        bert_tokenizer=self.bert_tokenizer,
                        is_clip_long=self.is_clip_long,
                        max_instr_len=max_instr_len
                    )
                    outputs_dict[0]['sub_instruction'] = sub_instr_tokens
                    batch = batch_obs(outputs_dict, self.device)

                    # update prev_actions
                    stack_rgb[env_idx].push(outputs_dict[0]["rgb"])
                    stack_depth[env_idx].push(outputs_dict[0]["depth"])
                    prev_globalgps[env_idx].push(batch[env_idx]['globalgps'].detach().cpu().numpy())
                    prev_globalyaw[env_idx].push(batch[env_idx]['global_rotation'][-1].detach().cpu().item())

                    for idx in range(env_num):
                        # reverse to make the latest frame to be 0 position
                        prev_globalgps_numpy = np.array(prev_globalgps[idx].get_stack(reverse=True))
                        prev_globalyaw_numpy = np.array(prev_globalyaw[idx].get_stack(reverse=True))
                        prev_act = _compute_actions( 
                            prev_globalgps_numpy, prev_globalyaw_numpy,
                            curr_time=0, fill_mode="constant",
                            len_traj_pred=self.eval_config.MODEL.len_traj_act,
                            waypoint_spacing=self.eval_config.MODEL.Diffusion_Policy.waypoint_spacing,
                            learn_angle=self.eval_config.MODEL.learn_angle,
                            metric_waypoint_spacing=self.eval_config.MODEL.Diffusion_Policy.metric_waypoint_spacing,
                            num_action_params=self.action_dim,
                            normalize=False)
                        action_deltas = get_delta(prev_act)
                        if self.eval_config.MODEL.learn_angle: 
                            # [x,y,yaw]
                            prev_act_delta = torch.from_numpy(action_deltas).to(self.device)
                            prev_act_delta_norm = normalize_data(prev_act_delta, self.action_stats)
                            prev_actions[idx] = prev_act_delta_norm
                        else:
                            # [forward, rotation]
                            prev_act_delta = torch.from_numpy(map_action_to_2d(action_deltas)).to(self.device)
                            prev_actions[idx] = prev_act_delta    

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

                    if not self.eval_config.MODEL.IMAGE_ENCODER.use_stack:
                        batch['rgb'] = batch_stack_rgb.squeeze(1)
                        batch['depth'] = batch_stack_depth.squeeze(1)
                        batch_stack_rgb, batch_stack_depth = None, None

                    batch = extract_image_features(
                        self.policy, batch, 
                        img_mod=self.eval_config.MODEL.IMAGE_ENCODER.RGB.img_mod,
                        len_traj_act=1,
                        world_size=1,
                        depth_encoder_type=self.eval_config.MODEL.IMAGE_ENCODER.DEPTH.bottleneck,
                        stack_rgb = batch_stack_rgb,
                        stack_depth = batch_stack_depth,
                        batch_stack_rgb_length = batch_stack_rgb_length,
                        proj=self.eval_config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
                        need_rgb_extraction=True,
                        classifier_free_mask_depth=classifier_free_mask_depth,
                        )

                    batch["steps"] = torch.from_numpy(np.array(steps)).to(self.device)
                    
                    # IMU
                    if self.eval_config.MODEL.IMU_ENCODER.use:
                        # initialize_imu
                        batch["imu"] = torch.zeros(batch["globalgps"].shape[0], self.eval_config.MODEL.IMU_ENCODER.input_size).to(self.device)
                        if self.eval_config.MODEL.IMU_ENCODER.to_local_coords:
                            batch["imu"][:, :2] = to_local_coords(batch["globalgps"][:, [0,1]].float(), start_positions, start_yaws)
                        else:
                            batch["imu"][:, :2] = batch["globalgps"][:, [0,1]] - start_positions
                        if self.eval_config.MODEL.IMU_ENCODER.input_size == 3:
                            batch["imu"][:, 2] = batch["globalyaw"] - start_yaws
                    
                    # update rnn states
                    with torch.no_grad():
                        batch_settings = {
                            'mode': 'update_rnn',
                            'observations': batch,
                            'rnn_states': rnn_states,
                            'prev_actions': prev_actions,
                            'masks': not_done_masks,
                        }
                        
                        _, update_rnn_states= self.policy(batch_settings)
                        rnn_states = update_rnn_states

                    not_done_masks = torch.tensor(
                        [[0] if done else [1] for done in dones],
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    if dones[0]:
                        stop_flag = True
                        break
                    
                if dones[0]:
                    result = reason
                    if result == '':
                        if info['success'] > 0:
                            result='success'
                        else:
                            info['fail_reason']='not_reach_goal'
                            result='not_reach_goal'
                    progress_log_util.trace_end(
                        trajectory_id = path_key,
                        step_count=statistic_info.sim_step,
                        result = result,
                    )

                    # info['ext_info']=map_info
                    data_collector.save_eval_result(
                        ckpt_name=self.ckpt_name, 
                        path_key=path_key, 
                        info=info
                    )
                    stats_episodes[path_key] = info
                    spl_dict[path_key] = float(stats_episodes[path_key]["spl"])
                    mean_spl = np.mean(list(spl_dict.values()))
                    log.info(f"Average SPL: {mean_spl}, result:{result}")
                    break
        
        progress_log_util.report()