import numpy as np
import importlib
from grutopia.core.env import BaseEnv
from vln.src.v2.util.stuck_checker import StuckChecker
from vln.src.v2.util.common import (
    check_robot_fall, 
    describe_action, 
    get_action_state,
    get_new_position_and_rotation,
    norm_depth,
)
from vln.src.v2.util.common_log_util import common_logger as log

class Statistic_Info:
    def __init__(
            self,
            env,
            path_data,
            shortest_path_length,
            shortest_to_goal_distance,
            step_interval,
            success_distance,
        ):
        self.env = env
        # policy 执行次数
        self.policy_step = 0
        # sim step 执行次数
        self.sim_step = 0
        # rotation 连续旋转次数
        self.continuous_rotation_count = 0
        # 目前已经移动的长度
        self.current_path_length = 0
        self.step_interval = step_interval
        # 摄像头采样
        self.stack_rgb = [[]]
        self.stack_depth = [[]]
        # 机器人位置采样
        self.prev_globalgps = [[]]
        self.prev_globalyaw = [[]]
        # 地图采样
        self.total_rgb_list = [[]]
        self.total_topdown_rgb_list = [[]]
        self.pred_traj_list = [[]]
        # 统计结果需要的信息
        self.path_data = path_data
        self.shortest_path_length = shortest_path_length
        self.shortest_to_goal_distance = shortest_to_goal_distance
        self.success_distance = success_distance
    
    def compute_metrics(
        self, 
        robot_position, 
        fail_reason=''
    ):
        """计算VLN任务的评估指标
        
        Args:
            infos: 当前episode的信息,包含当前位置、目标位置等
            
        Returns:
            metrics: 包含各项指标的字典
        """
        metrics = {}
        current_position = robot_position
        goal_position = self.path_data['reference_path'][-1]

        # 计算Navigation Error (NE) - 当前位置到目标的欧氏距离
        ne = np.linalg.norm(current_position[:2] - goal_position[:2])
        metrics['reference_path'] = self.path_data['reference_path']
        metrics['shortest_path_length'] = self.shortest_path_length
        metrics['pred_traj_list'] = self.pred_traj_list[0]
        metrics['NE'] = ne 
        
        # 计算Success Rate (SR) - 是否到达目标点
          # 成功阈值通常设为3米
        success = ne < self.success_distance
        metrics['success'] = float(success)
        
        # 计算Oracle Success Rate (OSR) - 轨迹中是否有点达到目标
        if ne < self.shortest_to_goal_distance:
            self.shortest_to_goal_distance = ne
        metrics['osr'] = float(self.shortest_to_goal_distance < self.success_distance)
        
        # 计算Trajectory Length (TL) - 轨迹总长度
        metrics['TL'] = self.current_path_length

        # 计算SPL (Success weighted by Path Length)
        if metrics['TL'] > 0:
            spl = metrics['success'] * self.shortest_path_length / max(
                metrics['TL'], self.shortest_path_length
            )
        else:
            spl = 0
        metrics['spl'] = spl
        
        # 计算NDTW (Normalized Dynamic Time Warping)
        # 计算当前轨迹与参考轨迹之间的DTW距离
        # DTW参数
        dtw_threshold = self.success_distance  # 通常设置为success_distance
        # 计算路径间的累积DTW距离
        dtw_distance = 0.0
        trajectory = []
        if len(self.pred_traj_list[0]) > 0:
            trajectory = np.array(self.pred_traj_list[0])[:,:2] # 只取x,y坐标
            reference_path = np.array(self.path_data['reference_path'])[:,:2]

            for point in trajectory:
                # 找到参考路径上最近的点
                min_dist = float('inf')
                for ref_point in reference_path:
                    dist = np.linalg.norm(point - ref_point)
                    min_dist = min(min_dist, dist)
                # 累加DTW距离,使用高斯函数进行归一化
                dtw_distance += np.exp(-min_dist**2 / (2 * dtw_threshold**2))
            
        # 归一化DTW得分
        ndtw = dtw_distance / len(trajectory) if len(trajectory) > 0 else 0.0
        metrics['ndtw'] = ndtw
        
        # 计算SDTW (Success weighted Dynamic Time Warping)
        # metrics['sdtw'] = metrics['success'] * metrics['ndtw']
        
        # 其他可能的指标
        metrics['steps'] = self.sim_step  # 步数
        metrics['episode_id'] = self.path_data['episode_id']  # episode ID
        metrics['trajectory_id'] = self.path_data['trajectory_id']  # 轨迹 ID
        metrics['fail_reason'] = fail_reason
        
        return [metrics] # batch size = 1

class ActionExecutor:
    def __init__(
        self, 
        env:BaseEnv, 
        task, 
        stuck_checker:StuckChecker,
        robot,

        per_action_max_step,
        total_max_step,
        robot_ankle_height,

        statistic_info:Statistic_Info,
        context,
        robot_name='h1',
        fall_height_threshold=0.5,
        max_rotation_count=-1
    ):
        # 执行 step 用到的工具类
        self.env=env
        self.task=task
        self.stuck_checker = stuck_checker
        self.robot = robot
        self.isaac_robot = self.robot.isaac_robot

        # 执行 step 需要的配置信息
        self.per_action_max_step=per_action_max_step
        self.total_max_step = total_max_step
        self.robot_ankle_height = robot_ankle_height
        self.max_rotation_count = max_rotation_count
        # 统计信息
        self.statistic_info = statistic_info
        # 可以优化掉的变量
        self.instruction = self.statistic_info.path_data['instruction']
        # 进程 stuck 检查使用
        self.context = context
        
        # robot name
        self.robot_name = robot_name
        self.fall_height_threshold = fall_height_threshold

    def _check_max_steps(self, step):
        if step > self.per_action_max_step:
            return True, 'exceed_per_action_max_step'
        if self.statistic_info.sim_step > self.total_max_step:
            return True, 'exceed_total_max_step'
        return False, ''

    def _check_fall_and_stuck(self,robot_position,robot_rotation,step):
        is_stuck = self.stuck_checker.check_robot_stuck(robot_position, robot_rotation, cur_iter=step, max_iter=2500, threshold=0.2)
        robot_bottom_z = self.robot.get_ankle_height() - self.robot_ankle_height
        is_fall = check_robot_fall(robot_position, robot_rotation, robot_bottom_z, height_threshold=self.fall_height_threshold)

        if is_stuck or is_fall:
            reason = 'fall' if is_fall else 'stuck'
            log.warning(f"Current action has been interrupted by {reason}.")
            return [True], reason
        return [False], ''
    
    def _check_max_rotation_count(self, action):
        is_action_rotate = (action == 2 or action == 3)
        if is_action_rotate:
            self.statistic_info.continuous_rotation_count += 1
        else:
            self.statistic_info.continuous_rotation_count = 0
        if self.statistic_info.continuous_rotation_count > self.max_rotation_count:
            return True, 'exceed_max_rotation_count'
        return False, ''

    def _execute_action(
        self,
        action, 
        action_name, 
        check_fall_and_stuck,
        check_max_rotation_count=False,
    ):
        finish_state = False
        step = 0
        dones = [False]
        reason = ''
        prev_position, _ = self.task.get_robot_poses_without_offset()
        if check_max_rotation_count:
            over_max_rotation_count, desc = self._check_max_rotation_count(action[0]['h1'][action_name][0])
            if over_max_rotation_count:
                reason = desc
                log.warning(f"Current action has been interrupted by {reason}.")
                return [True], reason
        
        while not finish_state:
            self.context.update_timestamp()
            obs = self.env.step(actions=action, add_rgb_subframes=False, render=False)
            robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
            
            self.statistic_info.current_path_length += np.linalg.norm(robot_position[:2] - prev_position[:2])
            prev_position = robot_position

            finish_state = get_action_state(obs, action_name)
            step += 1
            self.statistic_info.sim_step += 1
            over_max_step, desc = self._check_max_steps(step)
            if over_max_step:
                dones = [True]
                reason = desc
                break

            if check_fall_and_stuck and step % 20 == 0:
                fall_or_stuck, desc = self._check_fall_and_stuck(robot_position, robot_rotation, self.statistic_info.sim_step)
                if fall_or_stuck[0]:
                    dones = [True]
                    reason = desc
                    log.warning(f"Current action has been interrupted by {reason}.")
                    break
        return dones, reason

    def env_step(
        self,
        actions, 
    ):
        '''step in isaac-sim until the action has finished'''
        dones = [False]
        reason = ''
        action_name = list(actions[0][self.robot_name].keys())[0]
        if action_name == 'stop' or len(actions) == 0:
            dones = [True]
        else:
            dones,reason = self._execute_action(
                action = actions, 
                action_name=action_name,  
                check_fall_and_stuck=True,
                check_max_rotation_count=False
            )
        
        robot_position, robot_rotation = self.isaac_robot.get_world_pose()
        outputs_dict = get_obs(self.env,self.instruction,robot_position,robot_rotation, robot_name=self.robot_name)
        self.statistic_info.pred_traj_list[0].append(robot_position)
        infos = self.statistic_info.compute_metrics(robot_position=robot_position,fail_reason=reason)
        
        if action_name == 'move_by_descrete':
            action_list = actions[0][self.robot_name]['move_by_descrete']
            for one_action in action_list:
                log.info(f"[descrete][step:{self.statistic_info.sim_step}] 完成动作:{describe_action(one_action)},距离目标 {round(infos[0]['NE'],2)} 米")

        return {
            "outputs_dict": outputs_dict,
            "dones": dones,
            "infos": infos,
            "reason": reason,
        }

class FlashActionExecutor:
    def __init__(
        self, 
        env:BaseEnv, 
        task, 
        statistic_info:Statistic_Info,
        context,
        total_max_step,
    ):
        # 执行 step 用到的工具类
        self.env=env
        self.task=task
        self.total_max_step = total_max_step
        # 统计信息
        self.statistic_info = statistic_info
        # 可以优化掉的变量
        self.instruction = self.statistic_info.path_data['instruction']
        # 进程 stuck 检查使用
        self.context = context

    def _execute_action(
        self,
        action,
    ):
        self.context.update_timestamp()
        robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
        new_robot_position, new_robot_rotation = get_new_position_and_rotation(robot_position, robot_rotation,action)
        self.context.reset_robot(new_robot_position,new_robot_rotation)
        self.env.step(actions=[{self.robot_name:{'stand_still': []}}], render=True)
        self.statistic_info.sim_step += 1
        self.statistic_info.current_path_length += np.linalg.norm(new_robot_position[:2] - robot_position[:2])

    def env_step(
        self,
        action, 
    ):
        '''step in isaac-sim until the action has finished'''
        dones = [False]
        reason = ''
        if action == 0:
            dones = [True]
        else:
            self._execute_action(action)
            if self.statistic_info.sim_step > self.total_max_step:
                dones = [True]
                reason = 'exceed_total_max_step'
        
        robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
        outputs_dict = get_obs(self.env,self.instruction,robot_position,robot_rotation, robot_name=self.robot_name)
        self.statistic_info.pred_traj_list[0].append(robot_position)
        infos = self.statistic_info.compute_metrics(robot_position=robot_position,fail_reason=reason)
        log.info(f"[descrete][step:{self.statistic_info.sim_step}] 完成动作:{describe_action(action)},距离目标 {round(infos[0]['NE'],2)} 米")
        return {
            "outputs_dict": outputs_dict,
            "dones": dones,
            "infos": infos,
            "reason": reason,
        }

def get_obs(env, instruction,robot_position,robot_rotation, sub_instr=None, sub_instr_tokens=None, robot_name='h1'):
    obs = env.get_observations(add_rgb_subframes=True)
    obs_data = {}
    obs_data['globalgps'] = None
    obs_data['global_rotation'] = None
    obs_data['globalyaw'] = None
    obs_data['rgb'] = None
    obs_data['depth'] = None
    obs_data['instruction'] = instruction['instruction_text']
    if "instruction_tokens" in instruction:
        obs_data['instruction_tokens'] = instruction['instruction_tokens']
    
    # For MLANet
    if sub_instr is not None:
        # obs_data['sub_instruction'] = sub_instr
        obs_data['sub_instruction'] = sub_instr_tokens

    obs_data['step'] = 0
    cur_obs = obs['vln_0'][f"{robot_name}_0"]['pano_camera_0']
    rgb_info = cur_obs['rgba'][..., :3]
    depth_info = norm_depth(cur_obs['depth'])
    obs_data['rgb'] = rgb_info
    obs_data['depth'] = depth_info[..., np.newaxis]
    rotations_utils = importlib.import_module("omni.isaac.core.utils.rotations")
    quat_to_euler_angles = rotations_utils.quat_to_euler_angles
    _,_, yaw = quat_to_euler_angles(robot_rotation)
    obs_data['globalgps'] = np.array(robot_position)
    obs_data['global_rotation'] = np.array(robot_rotation)
    obs_data['globalyaw'] = yaw
    return [obs_data]

def generate_eval_key(ckpt_name, path_key):
    return f"eval_{ckpt_name}_{path_key}"