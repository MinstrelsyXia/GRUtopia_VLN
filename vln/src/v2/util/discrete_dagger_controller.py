import random
from vln.src.v2.util.eval import get_obs
from vln.src.models.utils.feature_extract import extract_instruction_tokens
from vln.src.utils.utils import batch_obs
from vln.src.v2.util.path_plan import plan_and_get_actions_discrete
from vln.src.v2.util.common import check_is_on_track
from vln.src.v2.util.common_log_util import common_logger as log
import numpy as np
import torch
import time

class DiscreteDaggerController:
    def __init__(
        self,
        data,
        context,
    ):
        self.data = data
        self.instruction = self.data['instruction']
        self.nav_path = data['reference_path']
        self.context = context
        self.path_planner = self.context.path_planner
        self.policy = self.context.policy
        self.policy_probability = self.context.policy_probability
        self.rnn_states = self.context.rnn_states
        self.task = self.context.task
        self.device = self.context.device
        self.env = self.context.env
        self.prev_action = 1

        # 1 表示 planner, 0 表示 policy
        if self.random_for_policy():
            self.mode = 0
        else:
            self.mode = 1
        self.set_action_count()
        log.info(f"controller init mode [ {self.mode} ][action_count:{self.action_count - 1}]")
        self.total_policy_count=0
        self.total_planner_count=0
        self.planner_action_index=-1
        self.planner_action_list=[]
        self.planner_real_point_list=[]
        self.current_point_index=0

    def set_action_count(self):
        action_count = 1
        max_action_count = 40
        if self.mode == 0:
            while self.random_for_policy():
                action_count += 1
                if action_count > max_action_count:
                    break
            self.action_count = action_count
        else:
            while not self.random_for_policy():
                action_count += 1
                if action_count > max_action_count:
                    break
            self.action_count = action_count

    def random_for_policy(self):
        random.seed(time.time())
        random_number = random.uniform(0, 1)
        return random_number < self.policy_probability

    def do_change_mode(self):
        old_mode = self.mode
        if self.mode == 0:
            new_mode = 1
            self.planner_action_index=-1
            self.planner_action_list=[]
            self.planner_real_point_list=[]
        else:
            new_mode = 0
        self.mode = new_mode
        self.set_action_count()
        log.info(f"controller change mode [ {old_mode} -> {new_mode} ][action_count:{self.action_count}]")
    
    def change_mode_if_needed(self):
        self.action_count -=1
        if self.action_count > 0:
            return
        self.do_change_mode()
    
    def get_next_action_by_policy(
        self,
    ):
        self.policy.eval()
        robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
        observations = get_obs(self.env, self.instruction ,robot_position,robot_rotation)
        rgb = observations[0]['rgb']
        depth = observations[0]['depth'].squeeze()
        observations = extract_instruction_tokens(
            observations, 
            bert_tokenizer=None,
            is_clip_long=False
        )
        observations = batch_obs(observations, self.device)
        observations["steps"] = torch.from_numpy(np.array([0])).to(self.device)
        not_done_masks = torch.zeros(1, 1, dtype=torch.uint8, device=self.device)
        prev_actions = torch.tensor([self.prev_action] , dtype=torch.uint8, device=self.device)
        batch = {
            'mode': 'inference',
            'observations': observations,
            'rnn_states': self.rnn_states,
            'prev_actions': prev_actions,
            'masks': not_done_masks
        }
        with torch.no_grad():
            actions, rnn_states = self.policy(batch)
        self.rnn_states = rnn_states
        return actions[0].item(), rgb, depth

    def if_need_path_plan(self):
        if len(self.planner_action_list) == 0:
            return True
        if self.planner_action_index == len(self.planner_action_list) - 1:
            return True
        robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
        is_on_track = check_is_on_track(
            robot_position=robot_position,
            robot_rotation=robot_rotation,
            action=self.prev_action,
            action_index=self.planner_action_index,
            real_points=self.planner_real_point_list,
        )
        if not is_on_track:
            return True
        return False

    def get_next_action_by_planner(
        self,
        current_point_index,
    ):
        need_path_plan = self.if_need_path_plan()
        if not need_path_plan:
            self.planner_action_index += 1
            return self.planner_action_list[self.planner_action_index], None
        goal = self.nav_path[current_point_index + 1]
        map_info = self.context.get_global_map(robot_height=1.55, dilation_iterations=2)
        camera_pose = self.context.topdown_global_map_camera.get_world_pose()[0] - self.task._offset
        robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
        height, width = self.context.topdown_global_map_camera._camera._resolution
        action_list, real_points, find_flag, reason = plan_and_get_actions_discrete(
            map_info=map_info,
            robot_position=robot_position,
            robot_rotation=robot_rotation,
            goal = goal,
            camera_pose = camera_pose,
            aperture=self.context.aperture,
            width=width,
            height=height,
            path_planner=self.path_planner,
        )
        if not find_flag or len(action_list) == 0:
            return None, reason
        self.planner_action_list = action_list
        self.planner_real_point_list = real_points
        self.planner_action_index = 0
        
        return self.planner_action_list[0], reason

    def get_next_action(
        self,
        current_point_index,
    ):  
        if current_point_index != self.current_point_index:
            self.current_point_index = current_point_index
            self.do_change_mode()
        else:
            self.change_mode_if_needed()
        policy_action, rgb, depth = self.get_next_action_by_policy()
        if self.mode == 1:
            planner_action, reason = self.get_next_action_by_planner(current_point_index)
            self.prev_action = planner_action
            self.total_planner_count +=1
            return planner_action,'planner', rgb, depth, reason
        else:
            self.prev_action = policy_action
            self.total_policy_count +=1
            return policy_action, 'policy', rgb, depth, None
    
    def report(self):
        total = self.total_policy_count + self.total_planner_count
        policy_percentage = self.total_policy_count / total
        policy_percentage = round(policy_percentage,4)  * 100
        log.info(f"controller report: policy_percentage = [ {self.total_policy_count} / {total} ] = {policy_percentage}%")