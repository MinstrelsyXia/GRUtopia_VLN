# Author: w61
# Date: 2024.11.19
# the task env file.

import os,sys
import yaml
import time
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt

from vln.src.dataset.data_utils_multi_env import VLNDataLoader, load_scene_usd
from vln.src.utils.utils import to_global_coords

class TaskEnv(VLNDataLoader):
    def __init__(self, config, splits, eval_logger, filter_same_trajectory=False, policy_eval=True,):
        self.config = config
        vln_config = config.vln_config
        sim_config = config.sim_config
        self.eval_logger = eval_logger
        
        vln_config.camera_list = vln_config.settings.camera_list
        super().__init__(vln_config, sim_config, splits, filter_same_trajectory, policy_eval=policy_eval, eval_logger=eval_logger)
        
        # self.args -> vln_config
        # self.config -> eval_config
        
        self.current_split = ''
        self.finish_splits = []
        
        # warm up
        self.warm_up_steps = 240 if self.args.headless else 2500
        self.max_step = self.args.settings.max_step
        self.per_action_max_step = self.args.settings.per_action_max_step

        # actions
        self.action_name = self.args.settings.action
        self.success_distance = self.config.EVAL.success_distance
        
        # env_idx (for now, only support one env!)
        self.env_idx = 0
        self.env_nums = 1
        
        # observations
        self.camera_list = self.args.settings.camera_list

        self.is_app_up = False
    
    def manage_eval_data(self, reset_split=False, step_time=0):
        ''' Manage the data for eval process
        '''
        if reset_split:
            # reset for different splits
            if len(self.current_split) == 0:
                self.current_split = self.splits[0]
            else:
                self.current_split = self.splits[1]
            self.eval_logger.info(f"Start to evaluate on {self.current_split} split")
            
            self.finish_scans = []
            self.current_episode_idx = -1 # this is not episode_id in data. but the location in data.
            self.current_scan_data = self.data[self.current_split]
            self.number_of_episodes = [len(self.current_scan_data[scan]) for scan in self.current_scan_data.keys()]
            self.current_scan_list = list(self.current_scan_data.keys())
            self.current_scan_idx = 0
            self.current_scan = self.current_scan_list[self.current_scan_idx]
            
            self.start_step_list = [0]
            self.current_step_list = [0]

            self.shortest_to_goal_distance = 999 # 记录当前导航路径中距离目标的最近距离
            self.prev_position = None
            self.current_path_length = 0
            self.pred_traj_list = [[] for _ in range(self.env_nums)]

        if self.current_episode_idx < len(self.current_scan_data)-1:
            # new episode
            reset_scene = False
            self.current_episode_idx += 1
            self.start_step_list[self.env_idx] = step_time
            self.current_step_list[self.env_idx] = step_time
            
        else:
            # finish this scan
            self.finish_scans.append(self.current_scan)
            self.eval_logger.info(f"Finish the scan {self.current_scan}")
            
            # check weather all data in this split has been evaluated
            if len(self.finish_scans) == len(self.current_scan_list):
                self.eval_logger.info(f"Finish the split {self.current_split}")
                self.finish_splits.append(self.current_split)
                return None, None, None
            
            # reset to next scan
            reset_scene = True
            self.current_scan_idx += 1
            self.current_scan = self.current_scan_list[self.current_scan_idx]
            self.current_episode_idx = 0
            
        self.data_item = self.current_scan_data[self.current_scan][self.current_episode_idx]

        self.eval_logger.info(f"Current scan: {self.current_scan}, trajectory_id: {self.data_item['trajectory_id']}")
        self.eval_logger.info(f"Instruction: {self.data_item['instruction']['instruction_text']}")
        self.eval_logger.info(f"Start position: {self.data_item['start_position']}, Start rotation: {self.data_item['start_rotation']}")
        
        return self.current_scan, self.data_item, reset_scene
    
    def construct_env(self, init_omni_env=False, split=None, path_id_list=None, step_time=0):
        reset_split = True if len(self.current_split) == 0 else False
        scan, item, reset_scene = self.manage_eval_data(reset_split, step_time)
        if scan is None:
            scan, item, reset_scene = self.manage_eval_data(reset_split=True, step_time=step_time)
            if scan is None:
                # two splits have been evaluated.
                return None
        
        '''init or reset isaac-sim env'''
        if path_id_list is not None and split is not None:
            # load assigned split and path_id
            idx_list = []
            for path_id in path_id_list:
                for idx, item in enumerate(self.data[split][scan]):
                    if item['trajectory_id'] == path_id:
                        idx_list.append(idx)
                        break
        
        env_i = 0 # For now, we only use one env.
        scene_usd_path = load_scene_usd(self.args, scan)
        self.sim_config.config.tasks[env_i].scene_asset_path = scene_usd_path
        self.sim_config.config.tasks[env_i].robots[0].position = item["start_position"] # only one robot
        self.sim_config.config.tasks[env_i].robots[0].orientation = item["start_rotation"]  

        if reset_scene:
            # reset scene without restart app
            start_time = time.time()
            self.env._runner._world.clear()
            self.env._runner.add_tasks(self.sim_config.config.tasks)
            self.eval_logger.info(f"Reset scene {scan} without restarting app for using {((time.time()-start_time)/60):.2f} minutes.")
        elif init_omni_env:
            # start app.
            # should only be called at the first time.
            self.init_env(self.sim_config, headless=self.args.headless)
            self.init_omni_env()
            self.init_env_manager()
            self.is_app_up = True

        self.init_robots()
        
        if init_omni_env:
            warm_up_steps = self.warm_up_steps
        else:
            warm_up_steps = 50

        # wait for the agent to be ready.
        warm_up_step = 0
        env_actions = [{'h1':{self.action_name:[[item["start_position"]]]}}]
        start_time = time.time()
        while self.env.simulation_app.is_running() and warm_up_step < warm_up_steps:
            self.env.step(actions=env_actions)
            warm_up_step += 1
        end_time = time.time()
        fps = warm_up_step / (end_time - start_time)
        self.eval_logger.info(f"Warm up for {warm_up_step} steps. FPS: {fps:.2f}")

        # get_shortest_path
        self.prev_position = self.get_robot_poses()[self.env_idx][0]
        self.gt_exe_path, self.shortest_path_length = self.get_shortest_path(self.current_scan)
        self.eval_logger.info(f"The shortest path length is {self.shortest_path_length:.2f}")

        # obtain the observations
        obs = self.get_obs()
        
        return obs
    
    def get_shortest_path(self, scan):
        # Init the topdown map
        self.topdown_map = self.GlobalTopdownMap(self.args, scan)
        self.freemap, self.camera_pose = self.get_global_free_map_single(self.env_idx, verbose=False)
        self.topdown_map.update_map(self.freemap, self.camera_pose, verbose=False, env_idx=self.env_idx)
        self.eval_logger.info(f"The shortest path has been initialized for Scan {scan}, Path_id {self.data_item['trajectory_id']}")  

        # Compute the shortest path
        exe_path = self.topdown_map.navigate_p2p(self.data_item['reference_path'][0], self.data_item['reference_path'][-1], step_time=0, verbose=False, save_dir=self.config.GT_PATH_DIR)
        # exe_path = self.topdown_map.navigate_p2p(self.data_item['reference_path'][0], self.data_item['reference_path'][-1], step_time=0, verbose=True, save_dir=self.config.GT_PATH_DIR, all_paths=self.data_item['reference_path']) # DEBUG 

        # compute the length
        # 计算路径总长度
        shortest_path_length = 0
        for i in range(len(exe_path)-1):
            # 计算相邻两点之间的欧氏距离
            shortest_path_length += np.linalg.norm(np.array(exe_path[i+1]) - np.array(exe_path[i]))
        
        return exe_path, shortest_path_length
    
    def norm_depth(self, depth_info, min_depth=0, max_depth=10):
        depth_info[depth_info > max_depth] = max_depth
        depth_info = (depth_info - min_depth) / (max_depth - min_depth)
        return depth_info
    
    def get_obs(self):
        obs = self.env.get_observations(add_rgb_subframes=True)
        
        camera_pose_dict = self.get_camera_pose()
        robot_pose_dict = self.get_robot_poses()
        
        for env_idx, (task_name, task) in enumerate(obs.items()):
            # 只支持一个环境
            for robot_name, robot in task.items():
                # 假设每个环境中没有多个机器人
                obs_data = {}
                obs_data['globalgps'] = None
                obs_data['global_rotation'] = None
                obs_data['globalyaw'] = None
                obs_data['rgb'] = None
                obs_data['depth'] = None
                obs_data['instruction'] = self.data_item['instruction']['instruction_text']
                obs_data['step'] = self.current_step_list[env_idx] - self.start_step_list[env_idx]
                
                for camera in self.camera_list:
                    cur_obs = obs[task_name][robot_name][camera]
                    camera_pose = camera_pose_dict[task_name][camera]
                    pos, quat = camera_pose[0], camera_pose[1]
                    _,_, yaw = self.quat_to_euler_angles(quat) #TODO: 检查！

                    rgb_info = cur_obs['rgba'][..., :3]
                    depth_info = self.norm_depth(cur_obs['depth'])

                    if camera == self.config.IL.camera_name:
                        obs_data['rgb'] = rgb_info
                        obs_data['depth'] = depth_info[..., np.newaxis]

                pos, quat = robot_pose_dict[env_idx][0], robot_pose_dict[env_idx][1]
                _,_, yaw = self.quat_to_euler_angles(quat)

                # 更新所需的键
                obs_data['globalgps'] = np.array(pos)
                obs_data['global_rotation'] = np.array(quat)
                obs_data['globalyaw'] = yaw
    
        return [obs_data] # 批量大小为1
    
    def step(self, actions, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, verbose=False, check_fall_and_stuck=True):
        '''step in isaac-sim until the action has finished'''
        # TODO: This actually depends on the eval strategy:
        # 1. predict the next action until the action has finished. (now)
        # 2. predict the next action every interval
        finish_state = False
        start_step = 0
        dones = [False]
        infos = [] # TODO: compute the metrics!
        action_name = list(actions[0]['h1'].keys())[0]
        reason = ''
        
        current_position = self.get_robot_poses()[self.env_idx][0]
        self.eval_logger.info(f"========== Current position: {current_position}")

        if action_name == 'stop':
            dones = [True]
        else:
            while not finish_state:
                obs = self.env.step(actions=actions, add_rgb_subframes=False, render=False)
                # update the path length
                current_position = self.get_robot_poses()[self.env_idx][0]
                self.current_path_length = self.current_path_length + np.linalg.norm(current_position - self.prev_position)
                self.prev_position = current_position

                for env_idx, (task_name, task) in enumerate(obs.items()):
                    for robot_name, robot in task.items():
                        action_state = robot[action_name]
                        finish_state = action_state['finished']
                start_step += 1
                self.current_step_list[env_idx] += 1

                if self.current_step_list[env_idx] > self.max_step:
                    finish_state = False
                    self.eval_logger.error(f"Step has surpass the maximum steps. Break!")
                    dones[env_idx] = True
                    reason = 'exceed_max_step'
                    break

                if start_step > self.per_action_max_step:
                    finish_state = False
                    self.eval_logger.warning(f"Current action has exceeded the maximum steps ({self.per_action_max_step}) for each action. Breaking...")
                    reason = 'single_action_exceed_max_step'
                    break

                if start_step % self.config.EVAL.step_interval == 0:
                    # self.eval_logger.info(f"Step {start_step} for the action {actions}.")
                    self.eval_logger.info(f"Current position: {current_position}")
                    
                    # Update the states
                    outputs_dict = self.get_obs()
                    for idx in range(len(outputs_dict)):
                        stack_rgb[idx].push(outputs_dict[idx]["rgb"])
                        stack_depth[idx].push(outputs_dict[idx]["depth"])
                        
                        prev_globalgps[idx].push(outputs_dict[idx]["globalgps"])
                        prev_globalyaw[idx].push(outputs_dict[idx]["global_rotation"][-1])

                        if self.config.VIDEO_OPTION != -1:
                            total_rgb_list.append(outputs_dict[idx]["rgb"])
                
                if check_fall_and_stuck and start_step % 20 == 0:
                    status_abnormal_list, fall_list, stuck_list = self.check_and_reset_robot(cur_iter=self.current_step_list[self.env_idx], update_freemap=False, verbose=verbose)
                    for status_idx, status in enumerate(status_abnormal_list):
                        if self.warm_up_list[status_idx] == 0:
                            if status:
                                if fall_list[status_idx]:
                                    reason = 'fall'
                                elif stuck_list[status_idx]:
                                    reason = 'stuck'
                                self.episode_end_setting(self.current_split, self.current_scan, status_idx, reason)
                                self.eval_logger.warning(f"Current action has been interrupted by {reason}.")
                            dones[env_idx] = True
        
        outputs_dict = self.get_obs()
        if action_name == 'move_to_point':
            self.pred_traj_list[self.env_idx].extend(actions[0]['h1'][action_name])
        infos = self.compute_metrics(fail_reason=reason)
        
        return outputs_dict, dones, infos, self.current_step_list, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list
    
    def predicted_action_to_global(self, predicted_action, step_i,verbose=False):
        """
        将预测的单个动作转换为全局坐标系下的位置
        
        Args:
            predicted_action (torch.Tensor): 预测的动作 shape: [3] (dx, dy, dyaw)
            step_i (int): if step_i == -1, 则返回所有预测动作对应的全局位置和朝向
        
        Returns:
            global_position (np.ndarray): 全局坐标系下的位置 [x, y]
            global_quat (np.ndarray): 全局坐标系下的四元数朝向 [w, x, y, z]
        """
        # 获取当前机器人的位置和朝向
        current_position, current_rot = self.get_robot_poses()[self.env_idx]
        _, _, current_yaw = self.quat_to_euler_angles(current_rot)

        # 先将current_yaw归一化到[-π, π]区间
        original_yaw = copy.copy(current_yaw)
        current_yaw = np.arctan2(np.sin(current_yaw), np.cos(current_yaw))
        
        global_positions, global_yaws = to_global_coords(predicted_action, current_position, current_yaw)
        N = len(global_yaws)
        euler_angles = np.zeros((N, 3))  # [N, 3] array of [roll, pitch, yaw]
        euler_angles[:, 2] = global_yaws  # Set yaw values, keeping roll and pitch as 0
        
        global_quats = []
        for i in range(N):
            global_quats.append(self.euler_angles_to_quat(euler_angles[i]))  # Now expects [N, 3] input

        if verbose:
            self.topdown_map.draw_point(predicted_world_poses=global_positions, color=[1,0,0], current_world_pose=current_position, target_world_pose=self.data_item['reference_path'][-1], img_save_path=self.config.GT_PATH_DIR, step=self.current_step_list[self.env_idx], logger=self.eval_logger)
            
            self.draw_prediction(predicted_action, step_i)
    
        return global_positions, global_quats
    
    def draw_prediction(self, un_actions, step_i):
        cussum_actions = np.cumsum(un_actions, axis=1)
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        
        # Plot predicted actions with arrows
        plt.scatter(cussum_actions[:, 0], cussum_actions[:, 1], label='pred_actions', color='blue', alpha=0.5)
        for i in range(cussum_actions.shape[0]):
            # Calculate arrow direction components using yaw angle
            arrow_length = 0.2  # Adjust this value to change arrow length
            dx = arrow_length * np.cos(cussum_actions[i, 2])
            dy = arrow_length * np.sin(cussum_actions[i, 2])
            
            # Draw arrow
            plt.arrow(cussum_actions[i, 0], 
                    cussum_actions[i, 1], 
                    dx, dy, 
                    head_width=0.05, 
                    head_length=0.1, 
                    fc='blue', 
                    ec='blue',
                    alpha=0.5)
            
            # Add point index
            plt.text(cussum_actions[i, 0], cussum_actions[i, 1], 
                    i, fontsize=9, color='blue', ha='left')
        
        plt.savefig(os.path.join(self.config.GT_PATH_DIR, f'predicted_actions_{self.current_step_list[self.env_idx]}_step{step_i}.png'))
        self.eval_logger.info(f"Saved predicted actions to {os.path.join(self.config.GT_PATH_DIR, f'predicted_actions_{self.current_step_list[self.env_idx]}_step{step_i}.png')}")
        
    def compute_metrics(self, fail_reason=''):
        """计算VLN任务的评估指标
        
        Args:
            infos: 当前episode的信息,包含当前位置、目标位置等
            
        Returns:
            metrics: 包含各项指标的字典
        """
        metrics = {}
        
        # 获取当前位置和目标位置
        current_position = self.get_robot_poses()[self.env_idx][0]
        goal_position = self.data_item['reference_path'][-1]

        # 计算Navigation Error (NE) - 当前位置到目标的欧氏距离
        ne = np.linalg.norm(current_position[:2] - goal_position[:2])
        metrics['NE'] = ne 
        
        # 计算Success Rate (SR) - 是否到达目标点
          # 成功阈值通常设为3米
        success = ne < self.success_distance
        metrics['success'] = float(success)
        
        # 计算Oracle Success Rate (OSR) - 轨迹中是否有点达到目标
        min_distance = ne if ne < self.shortest_to_goal_distance else self.shortest_to_goal_distance # 如果需要轨迹中最小距离,需要在step中记录
        metrics['osr'] = float(min_distance < self.success_distance)
        
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
        if len(self.pred_traj_list[self.env_idx]) > 0:
            trajectory = np.array(self.pred_traj_list[self.env_idx])[:,:2] # 只取x,y坐标
            reference_path = np.array(self.data_item['reference_path'])[:,:2]

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
        metrics['steps'] = self.current_step_list[self.env_idx]  # 步数
        metrics['episode_id'] = self.data_item['episode_id']  # episode ID
        metrics['trajectory_id'] = self.data_item['trajectory_id']  # 轨迹 ID

        metrics['fail_reason'] = fail_reason
        
        return [metrics] # batch size = 1
