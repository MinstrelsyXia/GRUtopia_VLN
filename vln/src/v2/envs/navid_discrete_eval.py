import os, sys
import re
import random
import cv2
from .base import BaseSingleScanEnv
from grutopia.core.config import SimulatorConfig
from vln.src.v2.dataloader.eval import EvalPathKeyDataloader
from grutopia.core.util.log import log
from vln.src.v2.util import progress_log_util
from vln.src.v2.util.eval import(
    get_obs,
    Statistic_Info,
    ActionExecutor,
    generate_eval_key
)
from vln.src.models.utils.feature_extract import extract_instruction_tokens
import torch
from vln.src.utils.utils import batch_obs
import numpy as np
from vln.src.v2.util.stuck_checker import StuckChecker
import time
import sys
import lmdb
import msgpack_numpy
from vln.src.models.init_policy import initialize_policy

from vln.src.models.navid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vln.src.models.navid.conversation import conv_templates, SeparatorStyle
from vln.src.models.navid.model.builder import load_pretrained_model
from vln.src.models.navid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class NavidDiscreteEvalSingleScanEnv(BaseSingleScanEnv):
    
    def __init__(
            self,
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
        self.timestamp = time.time()
        self.lmdb_path = lmdb_path
        self.ckpt_name = ckpt_name
        self.ckpt_path = eval_config.IL.ckpt_to_load
        
        '''Init the policy'''
        self.conv_mode = "vicuna_v1"
        self.model_name = get_model_name_from_path(self.ckpt_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.ckpt_path, None, get_model_name_from_path(self.ckpt_path))
        
        log.info("Initialization Complete")
        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."
        
        self.rgb_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.reset()

    def update_timestamp(self):
        self.timestamp = time.time()
        sys.stdout.flush()

    def topdown_snapshot(self):
        map_info = self.get_global_map(
            robot_height=1.55,
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
            map_info = self.topdown_snapshot()
            

            # observations = batch_obs(observations, self.device)
            # observations["steps"] = torch.from_numpy(np.array([0])).to(self.device)
            env_nums = 1

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
                
                robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
                observations = get_obs(self.env, data['instruction'],robot_position,robot_rotation)
                action = self.act(observations[0]) # only one env
                actions = [action]
                
                if self.eval_config.EVAL.ACTION == 'descrete':
                    for bs_i, a in enumerate(actions):
                        cur_a = a['action']
                        if cur_a == 0:
                            log.info(f"[split:{split}][scan:{scan}][trajectory_id_episode_id: {path_key}][stop!!!]")
                            action = [
                                {'h1': {'stop': ['stop']}}
                            ]
                        else:
                            action = [
                                {'h1': {'move_by_descrete': [cur_a]}}
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
                )
                outputs = executor.env_step(actions = action)
                outputs_dict = outputs['outputs_dict']
                dones = outputs['dones']
                info = outputs['infos'][0]
                reason = outputs['reason']
                statistic_info = executor.statistic_info
                statistic_info.policy_step +=1

                if dones[0]:
                    result = reason
                    if result == '':
                        if info['success'] > 0:
                            result='success'
                        else:
                            info['fail_reason']='not_reach_goal'
                            result='not_reach_goal'
                    
                    self.reset()
                    
                    progress_log_util.trace_end(
                        trajectory_id = path_key,
                        step_count=statistic_info.sim_step,
                        result = result,
                    )

                    info['ext_info']=map_info
        
                    database_write = lmdb.open(f"{self.lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
                    with database_write.begin(write=True) as txn:
                        key_write = generate_eval_key(ckpt_name=self.ckpt_name,path_key=path_key).encode()
                        value_write = msgpack_numpy.packb(info, use_bin_type=True)
                        txn.put(key_write, value_write)
                    database_write.close()
                    stats_episodes[path_key] = info
                    spl_dict[path_key] = float(stats_episodes[path_key]["spl"])
                    mean_spl = np.mean(list(spl_dict.values()))
                    log.info(f"Average SPL: {mean_spl}")
                    break
        
        progress_log_util.report()
    
    #### Copy from navid
    def reset(self):
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.first_forward = False

    def act(self, observations):
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            return {"action": temp_action}

        navigation_qs = self.promt_template.format(observations["instruction"])
        navigation = self.predict_inference(navigation_qs)
        # if self.config.test_verbose:
        #     self.eval_logger.info(f"Navigation Output: {navigation}")
        
        
        action_index, num = self.extract_result(navigation[:-1])
        # log.info(f"Navigation Output: {navigation}")
        # log.info(f"action_index:{action_index} num:{num}")

        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1:
            for _ in range(min(3, int(num/25))):
                self.pending_action_list.append(1)

        elif action_index == 2:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(2)

        elif action_index == 3:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(3)
        
        if action_index is None or len(self.pending_action_list)==0:
            self.pending_action_list.append(random.randint(1, 3))
            # Primarily unused, intended to complete the pipeline logic.

        return {"action": self.pending_action_list.pop(0)}

    def process_images(self, rgb_list):
        batch_image = np.asarray(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        return [video]


    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)


        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs



    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)

        return None, None



    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]



        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line


        if line:

            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)


        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image