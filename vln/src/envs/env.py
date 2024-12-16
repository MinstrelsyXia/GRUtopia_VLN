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
import json

from vln.src.dataset.data_utils_multi_env import VLNDataLoader, load_scene_usd
from vln.src.utils.utils import to_global_coords

class TaskEnv(VLNDataLoader):
    def __init__(self, config, sim_config, splits, eval_logger, filter_same_trajectory=False, policy_eval=True):
        self.config = config
        vln_config = config.vln_config
        self.sim_config = sim_config
        self.eval_logger = eval_logger
        
        vln_config.camera_list = vln_config.settings.camera_list
        super().__init__(vln_config, sim_config, splits, filter_same_trajectory, policy_eval=policy_eval, eval_logger=eval_logger, load_eval=self.config.EVAL.load_eval_subset)
        
        # self.args -> vln_config
        # self.config -> eval_config
        
        self.current_split = ''
        self.finish_splits = []
        
        # warm up
        self.warm_up_steps = 160 if self.args.headless else 1200
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
    
    def manage_eval_data(self, reset_split=False, step_time=0, result_json_path=None):
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
            self.current_episode_idx = -1 # this is not episode_id in data. but the location in data. # !!! DEBUG. should be -1
            loaded_finished_scans = False
            if result_json_path is not None:
                self.result_json_path = result_json_path
                # jump the existing episode_id in result_json_path
                with open(self.result_json_path, 'r') as f:
                    loaded_results = json.load(f)
                    if len(loaded_results) > 0:
                        loaded_results = loaded_results[self.current_split]
                        if "finished_scans" in loaded_results:
                            # jump the existing finished scans
                            self.finished_scans = loaded_results["finished_scans"]
                            self.current_scan_idx = len(self.finished_scans)
                            loaded_finished_scans = True
            
            self.current_scan_data = self.data[self.current_split]
            self.number_of_episodes = [len(self.current_scan_data[scan]) for scan in self.current_scan_data.keys()]
            self.current_scan_list = list(self.current_scan_data.keys())
            self.current_scan_idx = 0 if not loaded_finished_scans else self.current_scan_idx
            self.current_scan = self.current_scan_list[self.current_scan_idx]

            if len(loaded_results) > 0:
                if self.current_scan not in loaded_results['episodes']:
                    # case 1: the scan has not been evaluated.
                    self.current_episode_idx = -1
                else:
                    # case 2: the scan has been partly evaluated.
                    self.current_episode_idx = len(loaded_results["episodes"][self.current_scan]) - 1
            
            self.start_step_list = [0]
            self.current_step_list = [0]

        if self.current_episode_idx < self.number_of_episodes[self.current_scan_idx] - 1:
            # new episode
            reset_scene = False
            self.current_episode_idx += 1
            self.start_step_list[self.env_idx] = step_time
            self.current_step_list[self.env_idx] = step_time
            
            self.shortest_to_goal_distance = 999 # 记录当前导航路径中距离目标的最近距离
            self.prev_position = None
            self.current_path_length = 0
            self.pred_traj_list = [[] for _ in range(self.env_nums)]
            
        else:
            # finish this scan
            self.finish_scans.append(self.current_scan)
            self.eval_logger.info(f"********Finish the scan {self.current_scan}")

            if self.result_json_path is not None:
                # record the finished scan in result_json_path
                with open(self.result_json_path, 'r') as f:
                    loaded_results = json.load(f)

                if "finished_scans" not in loaded_results[self.current_split]:
                    loaded_results[self.current_split]["finished_scans"] = []
                loaded_results[self.current_split]["finished_scans"].append(self.current_scan)
                with open(self.result_json_path, 'w') as f:
                    json.dump(loaded_results, f, indent=2)
            
            # check weather all data in this split has been evaluated
            if len(self.finish_scans) == len(self.current_scan_list):
                self.eval_logger.info(f"******** Finish the split {self.current_split}")
                self.finish_splits.append(self.current_split)
                return None, None, None
            
            # reset to next scan
            # TODO: cannot directly reset scene.
            reset_scene = True
            self.current_scan_idx += 1
            self.current_scan = self.current_scan_list[self.current_scan_idx]
            self.current_episode_idx = 0
            
        self.data_item = self.current_scan_data[self.current_scan][self.current_episode_idx]

        self.eval_logger.info(f"Current scan: {self.current_scan}, episode_id: {self.data_item['episode_id']}")
        self.eval_logger.info(f"Instruction: {self.data_item['instruction']['instruction_text']}")
        self.eval_logger.info(f"Start position: {self.data_item['start_position']}, Start rotation: {self.data_item['start_rotation']}")
        self.EP_DIR = os.path.join(self.config.GT_PATH_DIR, f"{self.current_split}_{self.current_scan}_{self.data_item['episode_id']}")
        os.makedirs(self.EP_DIR, exist_ok=True)
        
        return self.current_scan, self.data_item, reset_scene
    
    def construct_env(self, init_omni_env=False, split=None, path_id_list=None, step_time=0, result_json_path=None):
        reset_split = True if len(self.current_split) == 0 else False
        scan, item, reset_scene = self.manage_eval_data(reset_split, step_time, result_json_path=result_json_path)
        if scan is None:
            scan, item, reset_scene = self.manage_eval_data(reset_split=True, step_time=step_time, result_json_path=result_json_path)
            if scan is None:
                # two splits have been evaluated.
                self.eval_logger.info(f"All data in {self.current_split} and {split} have been evaluated.")
                return 'all_data_evaluated'
        
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
        # env_actions = [{'h1':{self.action_name:[[item["start_position"]]]}}]
        env_actions = [{'h1': {'stand_still': []}}]
        start_time = time.time()
        while self.env.simulation_app.is_running() and warm_up_step < warm_up_steps:
            self.env.step(actions=env_actions)
            warm_up_step += 1
            if self.config.show_topdown_window and warm_up_step % self.config.EVAL.step_interval == 0:
                self.save_topdown_map()
        end_time = time.time()
        fps = warm_up_step / (end_time - start_time)
        self.eval_logger.info(f"Warm up for {warm_up_step} steps. FPS: {fps:.2f}")

        # get_shortest_path
        self.prev_position = self.get_robot_poses()[self.env_idx][0]
        self.gt_exe_path, self.shortest_path_length = self.get_shortest_path(self.current_scan, verbose=self.config.test_verbose)
        # np.save(os.path.join(self.EP_DIR, 'gt_exe_path.npy'), self.gt_exe_path) # !!!
        self.eval_logger.info(f"The shortest path length is {self.shortest_path_length:.2f}")
        if self.shortest_path_length == 0:
            self.eval_logger.error(f"The shortest path planning for {self.current_scan} has failed. Please check the data.")
            return 'shortest_path_planning_failed'

        # obtain the observations
        obs = self.get_obs()
        
        return obs
    
    def get_shortest_path(self, scan, verbose=False):
        # Init the topdown map
        self.topdown_map = self.GlobalTopdownMap(self.args, scan, vis_verbose=verbose)
        self.freemap, self.camera_pose = self.get_global_free_map_single(self.env_idx, verbose=False)
        self.topdown_map.update_map(self.freemap, self.camera_pose, verbose=False, env_idx=self.env_idx)
        self.eval_logger.info(f"The shortest path has been initialized for Scan {scan}, Episode_id {self.data_item['episode_id']}")  

        # Compute the shortest path
        exe_path = self.topdown_map.navigate_p2p(self.data_item['reference_path'][0], self.data_item['reference_path'][-1], step_time=0, verbose=verbose, save_dir=self.EP_DIR)
        # exe_path = self.topdown_map.navigate_p2p(self.data_item['reference_path'][0], self.data_item['reference_path'][-1], step_time=0, verbose=True, save_dir=self.config.GT_PATH_DIR, all_paths=self.data_item['reference_path']) # DEBUG 

        # compute the length
        # 计算路径总长度
        if exe_path is not None:
            shortest_path_length = 0
            for i in range(len(exe_path)-1):
                # 计算相邻两点之间的欧氏距离
                shortest_path_length += np.linalg.norm(np.array(exe_path[i+1]) - np.array(exe_path[i]))
        else:
            shortest_path_length = 0
        
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
                if "instruction_tokens" in self.data_item['instruction']:
                    # This is for cma from habitat.
                    # It seems that vlnce-cma uses the Glove to encode the instruction.
                    obs_data['instruction_tokens'] = self.data_item['instruction']['instruction_tokens']
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
                    elif 'topdown' in camera:
                        obs_data['topdown_rgb'] = rgb_info

                pos, quat = robot_pose_dict[env_idx][0], robot_pose_dict[env_idx][1]
                _,_, yaw = self.quat_to_euler_angles(quat)

                # 更新所需的键
                obs_data['globalgps'] = np.array(pos)
                obs_data['global_rotation'] = np.array(quat)
                obs_data['globalyaw'] = yaw
    
        return [obs_data] # 批量大小为1
    
    def step(self, actions, stack_rgb=None, stack_depth=None, prev_globalgps=None, prev_globalyaw=None, total_rgb_list=None, total_topdown_rgb_list=None, rot_action=None, verbose=False, check_fall_and_stuck=True):
        '''step in isaac-sim until the action has finished'''
        dones = [False]
        reason = ''
        action_name = list(actions[0]['h1'].keys())[0]
        
        current_position = self.get_robot_poses()[self.env_idx][0]
        if verbose:
            self.eval_logger.info(f"========== Current position: {current_position}")

        if action_name == 'stop':
            dones = [True]
        else:
            if len(actions) > 0:
                dones, reason, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list = self._execute_action(actions, action_name, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list, verbose, check_fall_and_stuck)
            
            if rot_action is not None:
                dones, reason, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list = self._execute_action(rot_action, action_name, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list, verbose, check_fall_and_stuck)

        outputs_dict = self.get_obs()
        if action_name == 'move_to_point':
            self.pred_traj_list[self.env_idx].extend(actions[0]['h1'][action_name])
        infos = self.compute_metrics(fail_reason=reason)

        current_position = self.get_robot_poses()[self.env_idx][0]
        stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list = self._update_states(current_position, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list, verbose=verbose)
        
        return {"outputs_dict": outputs_dict,
                "dones": dones,
                "infos": infos,
                "current_step_list": self.current_step_list,
                "stack_rgb": stack_rgb,
                "stack_depth": stack_depth,
                "prev_globalgps": prev_globalgps,
                "prev_globalyaw": prev_globalyaw,
                "total_rgb_list": total_rgb_list,
                "total_topdown_rgb_list": total_topdown_rgb_list
                }
    
    def _execute_action(self, action, action_name, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list, verbose, check_fall_and_stuck):
        finish_state = False
        start_step = 0
        dones = [False]
        reason = ''
        
        while not finish_state:
            obs = self.env.step(actions=action, add_rgb_subframes=False, render=False)
            current_position = self.get_robot_poses()[self.env_idx][0]
            self.current_path_length += np.linalg.norm(current_position - self.prev_position)
            self.prev_position = current_position

            finish_state = self._get_action_state(obs, action_name)
            start_step += 1
            self.current_step_list[self.env_idx] += 1

            if self._check_max_steps(start_step):
                dones[self.env_idx] = True
                reason = 'exceed_max_step' if self.current_step_list[self.env_idx] > self.max_step else 'single_action_exceed_max_step'
                break

            if start_step % self.config.EVAL.step_interval == 0:
                stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list =self._update_states(current_position, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list)
                if self.config.show_topdown_window:
                    self.save_topdown_map()
            
            if check_fall_and_stuck and start_step % 20 == 0:
                fall_or_stuck, reason = self._check_fall_and_stuck(verbose)
                if fall_or_stuck[self.env_idx]:
                    dones[self.env_idx] = True
                    self.eval_logger.warning(f"Current action has been interrupted by {reason}.")
                    break

        return dones, reason, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list
    
    def save_topdown_map(self):
        # 获取俯视相机的观察结果
        obs = self.get_obs()
        topdown_rgb = obs[0]['topdown_rgb']
        # save_path = os.path.join(self.config.GT_PATH_DIR, f'topdown_{self.current_step_list[self.env_idx]}.png')
        save_path = os.path.join(self.EP_DIR, f'topdown_view.png')
        plt.imsave(save_path, topdown_rgb)
        print(f"Saved topdown view to {save_path}")
    
    def _get_action_state(self, obs, action_name):
        for env_idx, (task_name, task) in enumerate(obs.items()):
            for robot_name, robot in task.items():
                action_state = robot[action_name]
                return action_state['finished']
        return False

    def _check_max_steps(self, start_step):
        if self.current_step_list[self.env_idx] > self.max_step:
            self.eval_logger.error(f"Step has surpass the maximum steps. Break!")
            return True
        if start_step > self.per_action_max_step:
            self.eval_logger.warning(f"Current action has exceeded the maximum steps ({self.per_action_max_step}) for each action. Breaking...")
            return True
        return False

    def _update_states(self, current_position, stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list, verbose=False):
        if verbose: 
            self.eval_logger.info(f"Current position: {current_position}")
        outputs_dict = self.get_obs()
        for idx in range(len(outputs_dict)):
            if stack_rgb is not None:
                stack_rgb[idx].push(outputs_dict[idx]["rgb"])
                stack_depth[idx].push(outputs_dict[idx]["depth"])
                prev_globalgps[idx].push(outputs_dict[idx]["globalgps"])
                prev_globalyaw[idx].push(outputs_dict[idx]["global_rotation"][-1])
            if self.config.VIDEO_OPTION != -1:
                total_rgb_list.append(outputs_dict[idx]["rgb"])
                total_topdown_rgb_list.append(outputs_dict[idx]["topdown_rgb"])
            self.pred_traj_list[idx].append(current_position)
        
        return stack_rgb, stack_depth, prev_globalgps, prev_globalyaw, total_rgb_list, total_topdown_rgb_list   

    def _check_fall_and_stuck(self, verbose):
        status_abnormal_list, fall_list, stuck_list = self.check_and_reset_robot(cur_iter=self.current_step_list[self.env_idx], update_freemap=False, verbose=verbose)
        for status_idx, status in enumerate(status_abnormal_list):
            # if self.warm_up_list[status_idx] == 0 and status:
            if status:
                reason = 'fall' if fall_list[status_idx] else 'stuck'
                # self.episode_end_setting(self.current_split, self.current_scan, status_idx, reason)
                self.eval_logger.warning(f"Current action has been interrupted by {reason}.")
                return [True], reason
        return [False], ''
    
    def predicted_action_to_global(self, predicted_action, step_i, len_traj_act=None, verbose=False):
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

        not_stop_idx = 0
        for action in predicted_action:
            if isinstance(action, str) and action == 'STOP':
                break
            else:
                not_stop_idx += 1
        predicted_action = np.array(predicted_action[:not_stop_idx])
        
        if isinstance(predicted_action, list):
            predicted_action = np.array(predicted_action)
        global_positions, global_yaws = to_global_coords(predicted_action, current_position, current_yaw)
        N = len(global_yaws)
        euler_angles = np.zeros((N, 3))  # [N, 3] array of [roll, pitch, yaw]
        euler_angles[:, 2] = global_yaws  # Set yaw values, keeping roll and pitch as 0
        
        global_quats = []
        for i in range(N):
            global_quats.append(self.euler_angles_to_quat(euler_angles[i]))  # Now expects [N, 3] input

        if verbose:
            self.topdown_map.draw_point(predicted_world_poses=global_positions, color=[1,0,0], current_world_pose=current_position, target_world_pose=self.data_item['reference_path'][-1], img_save_path=self.EP_DIR, step=self.current_step_list[self.env_idx], logger=self.eval_logger)
            
            # self.draw_prediction(current_yaw, predicted_action, global_yaws, step_i)
        
        exe_actions = self.convert_xyyaw_actions(global_positions, global_quats, len_traj_act)
    
        return global_positions, global_quats, exe_actions

    def convert_xyyaw_actions(self, global_positions, global_quats, len_traj_act):
        '''Further adjust target points and orientations based on distance thresholds
        
        Args:
            global_positions (np.ndarray): Array of global position coordinates [[x,y,z], ...]
            global_quats (list): List of quaternion orientations [[w,x,y,z], ...]
            
        Returns:
            exe_actions (list): List of selected positions and orientations based on distance threshold
        '''
        # Initialize variables
        exe_actions = []
        cumulative_distance = 0
        distance_threshold = 0.25  # Threshold for adding new waypoint (in meters)
        last_pos = global_positions[0]
        
        if len_traj_act is None:
            len_traj_act = len(global_positions)
        
        # Iterate through positions to find waypoints based on cumulative distance
        last_idx = 0
        for i in range(1, len(global_positions)):
            current_pos = global_positions[i]
            # Calculate distance from last added position
            distance = np.linalg.norm(current_pos - last_pos)
            cumulative_distance += distance
            last_idx = i
            
            # If cumulative distance exceeds threshold, add new waypoint
            if cumulative_distance >= distance_threshold:
                exe_actions.append(current_pos)
                # Reset cumulative distance and update last position
                cumulative_distance = 0
                last_pos = current_pos
                
                if len(exe_actions) == len_traj_act:
                    break
        
        # Always add final orientation in exe_actions
        exe_actions.append(global_quats[last_idx])
        
        return exe_actions

    def get_speed_actions(self, predicted_actions, len_traj_act, verbose=False):
        """将预测的位置变化序列转换为速度控制指令序列
        
        Args:
            predicted_actions: 预测的动作序列,每个动作包含(delta_x, delta_y, delta_yaw)
            
        Returns:
            speed_actions: 包含多个[forward_speed, lateral_speed, rotation_speed]的速度控制指令序列
        """
        speed_actions = []
        not_stop_idx = 0
        for action in predicted_actions:
            if isinstance(action, str) and action == 'STOP':
                break
            else:
                not_stop_idx += 1
        predicted_actions = predicted_actions[:not_stop_idx]
        
        ''' v1: step-by-step '''
        # convert cumsum to delta
        # delta_predicted_actions = np.diff(predicted_actions, axis=0)
        # predicted_actions = np.concatenate([predicted_actions[0:1], delta_predicted_actions], axis=0)

        # predicted_actions = predicted_actions[:len_traj_act]
        # max_distance = 0.5
        # for idx, action in enumerate(predicted_actions):
        #     [forward_speed, lateral_speed, rotation_speed], only_rotation = self.action_to_speed(action, max_distance)
        #     speed_actions.append([forward_speed, lateral_speed, rotation_speed])

        # add the final rotation action
        # if not only_rotation:
        #     if abs(action[2]) > self.yaw_threshold:
        #         rotation_speed = self.max_rotation_speed * action[2]
        #     speed_actions.append([0.0, 0.0, rotation_speed])
        
        ''' v2: go to the middle point '''
        # len_traj_act = len(predicted_actions)
        # max_distance = 0.3
        # middle_point = predicted_actions[len_traj_act//2]
        # speed_actions, only_rotation = self.action_to_speed(middle_point, max_distance, speed_actions)
        
        ''' v3: adaptive to choose the keypoints '''
        speed_actions = self.adaptive_action_to_speed(predicted_actions, len_traj_act, verbose)
        
        if verbose:
            for i, action in enumerate(speed_actions):
                self.eval_logger.info(f"Action {i}: forward={action[0]:.3f}, lateral={action[1]:.3f}, rotation={action[2]:.3f}")
        
        return speed_actions

    def action_to_speed(self, action, max_distance=0.5, speed_actions=[], add_final_rotation=True):
        # 设置阈值参数
        position_threshold = float(self.config.EVAL.rotation_threshold)  # 位置变化阈值,小于此值认为不需要移动
        self.yaw_threshold = 0.1  # 朝向变化阈值,小于此值认为不需要转向
        max_forward_speed = 1.0  # 最大前进速度
        max_lateral_speed = 1.0  # 最大横向速度
        self.max_rotation_speed = 2.0  # 最大旋转速度
        
        # 速度映射参数
        min_distance = 0  # 最小距离阈值
        max_distance = max_distance  # 最大距离阈值
        max_distance_per_forward = 0.5  # 每步前进的最大距离
        min_speed = 0.1  # 最小速度

        delta_x, delta_y, delta_yaw = action[0], action[1], action[2]

        only_rotation = False
        
        # 计算位置变化的距离
        distance = np.sqrt(delta_x**2 + delta_y**2)
        
        # 初始化速度指令
        forward_speed = 0.0
        lateral_speed = 0.0
        rotation_speed = self.max_rotation_speed
        
        if distance < position_threshold:
            # 如果位置变化很小,只进行旋转
            only_rotation = True
            if abs(delta_yaw) > self.yaw_threshold:
                # 根据旋转角度大小动态调整旋转速度
                rotation_speed *= delta_yaw
            xy_delta_yaw = 0.0
        else:
            forward_speed = max_forward_speed
            xy_delta_yaw = np.arctan2(delta_y, delta_x)
            delta_degree = np.degrees(xy_delta_yaw)

            angle_factor = (1 - (abs(xy_delta_yaw) * 2 / np.pi))**3
            distance_factor = min(1.0, distance / max_distance + min_speed)
            forward_speed *= angle_factor * distance_factor

            # 旋转速度
            if abs(xy_delta_yaw) > self.yaw_threshold:
                rotation_speed *= xy_delta_yaw
            else:
                rotation_speed = 0.0 
        
        # limit the rotation speed to max_rotation_speed (considered the sign)
        rotation_speed = np.clip(rotation_speed, -self.max_rotation_speed, self.max_rotation_speed)

        speed_actions.append([forward_speed, lateral_speed, rotation_speed])
        left_distance = distance - max_distance_per_forward

        while left_distance > 0:
            per_dis = min(left_distance, max_distance_per_forward)
            left_distance -= per_dis
            distance_factor = min(1.0, left_distance / max_distance_per_forward + min_speed)
            forward_speed *= distance_factor
            speed_actions.append([forward_speed, lateral_speed, 0.0])
        
        # rotate to the predicted_yaw action at the terminal
        if add_final_rotation:
            last_rotation_delta = delta_yaw - xy_delta_yaw
            if abs(last_rotation_delta) > self.yaw_threshold:
                rotation_speed *= last_rotation_delta
                rotation_speed = np.clip(rotation_speed, -self.max_rotation_speed, self.max_rotation_speed)
                speed_actions.append([0, 0, rotation_speed])

        return speed_actions, only_rotation
    
    def adaptive_action_to_speed(self, predicted_actions, len_traj_act, verbose=False):
        '''Further adjust target points and orientations based on distance thresholds
        
        Args:
            global_positions (np.ndarray): Array of global position coordinates [[x,y,z], ...]
            global_quats (list): List of quaternion orientations [[w,x,y,z], ...]
            
        Returns:
            speed_actions (list): List of selected positions and orientations based on distance threshold
        '''
        # Initialize variables
        speed_actions = []
        cumulative_distance = 0
        distance_threshold = 0.15  # Threshold for adding new waypoint (in meters)
        last_pos = predicted_actions[0]
        
        if len_traj_act is None:
            len_traj_act = len(predicted_actions)
        
        # Iterate through positions to find waypoints based on cumulative distance
        for i in range(1, len(predicted_actions)):
            current_pos = predicted_actions[i]
            # Calculate distance from last added position
            distance = np.linalg.norm(current_pos[:2] - last_pos[:2])
            cumulative_distance += distance
            
            # If cumulative distance exceeds threshold, add new waypoint
            if cumulative_distance >= distance_threshold:
                cur_speed, only_rotation = self.action_to_speed(current_pos, max_distance=0.3, speed_actions=[], add_final_rotation=True)
                speed_actions.extend(cur_speed)
                # Reset cumulative distance and update last position
                cumulative_distance = 0
                last_pos = current_pos
                
                if len(speed_actions) >= len_traj_act:
                    break     
        
        if len(speed_actions) == 0:
            cur_speed, only_rotation = self.action_to_speed(predicted_actions[-1], max_distance=0.3, speed_actions=[], add_final_rotation=True)
            speed_actions.extend(cur_speed)
        
        return speed_actions
    
    def draw_prediction(self, current_yaw, un_actions, global_yaws, step_i):
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        current_position = np.array([0,0])
        
        # Set axes through origin
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        # Plot predicted actions points and arrows
        ax.scatter(un_actions[:, 0], un_actions[:, 1], 
                label='Predicted Actions', color='blue', alpha=0.5)
    
        # Plot current position
        ax.scatter(current_position[0], current_position[1], 
        label='Current Position', color='red', marker='*', s=200)

        # Add current orientation arrow
        arrow_length = 0.3  # 使当前朝向箭头稍长一些以便区分
        dx = -arrow_length * np.sin(current_yaw)
        dy = arrow_length * np.cos(current_yaw)
        ax.arrow(current_position[0], current_position[1],
                dx, dy,
                head_width=0.08,  # 使当前朝向箭头稍粗一些
                head_length=0.15,
                fc='red',
                ec='red',
                label='Current Orientation')
        
        # Add arrows to show orientation
        arrow_length = 0.2
        for i in range(un_actions.shape[0]):
            dx = -arrow_length * np.sin(global_yaws[i])  # Swap sin/cos to match coordinate system
            dy = arrow_length * np.cos(global_yaws[i])
            
            ax.arrow(un_actions[i, 0], un_actions[i, 1],
                    dx, dy,
                    head_width=0.05,
                    head_length=0.1,
                    fc='blue',
                    ec='blue',
                    alpha=0.5)
            
            # Add index labels
            ax.text(un_actions[i, 0], un_actions[i, 1],
                    f' {i}', fontsize=9, color='blue')
        
        # Set axis labels
        ax.set_xlabel('Y-axis (meters)', x=1, ha='left')
        ax.set_ylabel('X-axis (meters)', y=1, ha='left')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set equal axis ratio
        ax.set_aspect('equal')
        
        # Auto-adjust display range and ensure origin at center
        max_range = max(abs(un_actions[:, :2]).max() * 1.2, 1.0)  # Add 20% margin, minimum range 1m
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Save figure
        save_path = os.path.join(self.EP_DIR, 
                                f'env_predicted_actions_{self.current_step_list[self.env_idx]}_step{step_i}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.eval_logger.info(f"Saved predicted actions plot to {save_path}")
        
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

    def draw_visited_map(self):
        '''Draw the visited map of the current episode'''
        freemap, camera_pose = self.get_global_free_map_single(self.env_idx, verbose=False)
        self.topdown_map.update_map(freemap, camera_pose, verbose=False, env_idx=self.env_idx)
        start_pixel = self.topdown_map.world_to_pixel(self.data_item['reference_path'][0])
        goal_pixel = self.topdown_map.world_to_pixel(self.data_item['reference_path'][-1])
        visited_path = [self.topdown_map.world_to_pixel(x) for x in self.pred_traj_list[self.env_idx]]
        save_path = os.path.join(self.EP_DIR, "visited_path_"+str(self.current_step_list[self.env_idx])+".jpg")
        self.topdown_map.vis_nav_path(start_pixel, goal_pixel, visited_path, freemap, img_save_path=save_path)
        self.eval_logger.info(f"Saved visited path plot to {save_path}")
        

