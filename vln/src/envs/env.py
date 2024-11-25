# Author: w61
# Date: 2024.11.19
# the task env file.

import os,sys
import yaml
import time
import numpy as np

from vln.src.dataset.data_utils_multi_env import VLNDataLoader, load_scene_usd

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
        self.warm_up_steps = 240 if self.args.headless else 2000
        self.max_step = self.args.settings.max_step
        
        # actions
        self.action_name = self.args.settings.action
        
        # env_idx (for now, only support one env!)
        self.env_idx = 0
        self.env_nums = 1
        
        # observations
        self.camera_list = self.args.settings.camera_list
    
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
            self.current_step_list = [self.warm_up_steps]
        
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
    
    def construct_env(self, init_omni_env=False, split=None, path_id_list=None):
        reset_split = True if len(self.current_split) == 0 else False
        scan, item, reset_scene = self.manage_eval_data(reset_split)
        if scan is None:
            scan, item, reset_scene = self.manage_eval_data(reset_split=True)
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

        self.init_robots()
        
        if init_omni_env:
            warm_up_steps = self.warm_up_steps
        else:
            warm_up_steps = 20

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
    
    def step(self, actions, verbose=False):
        '''step in isaac-sim until the action has finished'''
        # TODO: This actually depends on the eval strategy:
        # 1. predict the next action until the action has finished. (now)
        # 2. predict the next action every interval
        finish_state = False
        start_step = 0
        dones = [False]
        infos = [] # TODO: compute the metrics!
        action_name = list(actions[0]['h1'].keys())[0]
        if action_name == 'stop':
            dones = [True]

        else:
            while not finish_state:
                obs = self.env.step(actions=actions, add_rgb_subframes=False, render=False)
                for env_idx, (task_name, task) in enumerate(obs.items()):
                    for robot_name, robot in task.items():
                        action_state = robot[action_name]
                        finish_state = action_state['finished']
                start_step += 1
                self.current_step_list[env_idx] += 1

                if self.current_step_list[env_idx] > self.max_step:
                    finish_state = False
                    self.eval_logger.error(f"Step has surpass the maximum steps. Break!")
                    break

                if verbose and start_step % 50 == 0:
                    self.eval_logger.info(f"Step {start_step} in this action.")
        
        outputs_dict = self.get_obs()
        
        return outputs_dict, dones, infos
    
    def predicted_action_to_global(self, predicted_action):
        """
        将预测的单个动作转换为全局坐标系下的位置
        
        Args:
            predicted_action (torch.Tensor): 预测的动作 shape: [3] (dx, dy, dyaw)
        
        Returns:
            global_position (np.ndarray): 全局坐标系下的位置 [x, y]
            global_quat (np.ndarray): 全局坐标系下的四元数朝向 [w, x, y, z]
        """
        # 获取当前机器人的位置和朝向
        current_position, current_rot = self.get_robot_poses()[self.env_idx]
        _, _, current_yaw = self.quat_to_euler_angles(current_rot)
        
        # 计算当前朝向的旋转矩阵
        cos_theta = np.cos(current_yaw)
        sin_theta = np.sin(current_yaw)
        R = np.array([[cos_theta, -sin_theta],
                    [sin_theta, cos_theta]])
        
        # 将局部坐标变化转换到全局坐标系
        local_dxy = predicted_action[:2] # [dx, dy]
        global_dxy = np.dot(R, local_dxy)
        
        # 计算全局位置
        global_position = np.array([
            current_position[0] + global_dxy[0],
            current_position[1] + global_dxy[1],
            current_position[2]  # 保持原始z坐标
        ])
        
        # 计算全局朝向的欧拉角（roll=0, pitch=0）
        global_yaw = current_yaw + predicted_action[2]
        global_euler = np.array([0.0, 0.0, global_yaw])
        
        # 将欧拉角转换为四元数
        global_quat = self.euler_angles_to_quat(global_euler)
        
        return global_position, global_quat