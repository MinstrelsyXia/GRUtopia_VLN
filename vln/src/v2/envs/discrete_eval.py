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

class DiscreteEvalSingleScanEnv(BaseSingleScanEnv):
    
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
        policy, _, _, _ = initialize_policy(
            self.eval_config,
            log,
            load_from_ckpt=True,
            device=self.device,
            load_from_pretrain=False,
            action_stats=None,
        )
        self.policy = policy

        if self.eval_config.MODEL.policy_name == "CMA_CLIP_Policy":
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
        
        if hasattr(self.eval_config.MODEL, 'TEXT_ENCODER'):
            # use instr clip-long
            self.bert_tokenizer = longclip.tokenize
            self.use_bert = True
            self.is_clip_long = True


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

    def eval(self):
        
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
                max_instr_len = 200
            observations = get_obs(self.env, data['instruction'],robot_position,robot_rotation, sub_instr, sub_instr_tokens, robot_name=self.robot_name)
            observations = extract_instruction_tokens(
                observations, 
                #TODO:
                bert_tokenizer=self.bert_tokenizer,
                is_clip_long=self.is_clip_long,
                max_instr_len=max_instr_len
            )

            observations = batch_obs(observations, self.device)
            observations["steps"] = torch.from_numpy(np.array([0])).to(self.device)
            env_nums = 1
            rnn_states = torch.zeros(
                env_nums,
                self.policy.num_recurrent_layers,
                self.eval_config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
            prev_actions = torch.zeros(
                env_nums,
                1, device=self.device, dtype=torch.long
            )
            not_done_masks = torch.zeros(
                env_nums,
                1, dtype=torch.uint8, device=self.device
            )
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
            while True:
                if len(spl_dict) > 0 and np.mean(list(spl_dict.values())) < 0.02 and len(stats_episodes) > 100:
                    # this ckpt is too bad to continue
                    log.info(f"Break. This ckpt is too bad to continue with average SPL {np.mean(list(spl_dict.values())):.3f}")
                    return
                if statistic_info.sim_step % 1000 == 0:
                    log.info(f"[split:{split}][scan:{scan}][trajectory_id_episode_id: {path_key}][step:{statistic_info.sim_step}]")
                
                batch = {
                    'mode': 'inference',
                    'observations': observations,
                    'rnn_states': rnn_states,
                    'prev_actions': prev_actions,
                    'masks': not_done_masks
                }

                if self.use_clip_encoders:
                    batch.update({
                        "need_img_extraction": True,
                        "img_mod": self.eval_config.MODEL.IMAGE_ENCODER.RGB.img_mod,
                        'proj': self.eval_config.MODEL.IMAGE_ENCODER.RGB.rgb_proj,
                        'process_images': True,
                        'need_txt_extraction': True,
                        "depth_return_x_before_fc": False
                    })

                with torch.no_grad():
                    if self.use_clip_encoders:
                        actions, rnn_states, _ = self.policy(batch)
                    else:
                        actions, rnn_states, _ = self.policy(batch)
                prev_actions.copy_(actions)
                if self.eval_config.EVAL.ACTION == 'descrete':
                    for bs_i, a in enumerate(actions):
                        if a == 0:
                            log.info(f"[split:{split}][scan:{scan}][trajectory_id_episode_id: {path_key}][stop!!!]")
                            action = [
                                {self.robot_name: {'stop': ['stop']}}
                            ]
                        else:
                            action = [
                                {self.robot_name: {'move_by_descrete': [a.item()]}}
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

                if 'sub_instruction' in data: # MLANet
                    max_instr_len = 100
                else:
                    max_instr_len = 200
                outputs_dict = extract_instruction_tokens(
                    outputs_dict,
                    #TODO: 
                    bert_tokenizer=self.bert_tokenizer,
                    is_clip_long=self.is_clip_long,
                    max_instr_len=max_instr_len
                )
                outputs_dict[0]['sub_instruction'] = sub_instr_tokens
                observations = batch_obs(outputs_dict, self.device)

                observations["steps"] = torch.from_numpy(np.array([0])).to(self.device)
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )
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