from vln.src.dataset.data_utils_multi_env import load_scene_usd
from grutopia.core.env import BaseEnv
import numpy as np
from grutopia.core.util.log import log
import os
import math
from vln.src.utils import progress_log_util
from vln.src.local_nav.path_planner import AStarPlanner
from vlmaps.application_my.discrete.utils import (
    StuckChecker,
    AStarDiscretePlanner,
    SampleDataCache,
    check_robot_fall,
    plan_and_get_actions_discrete,
    plan_and_get_actions_continuous,
    get_env_actions_aggregation,
    is_one_action_finished,
    describe_action,
    is_halfway_sample_needed,
    is_action_finished,
)
# from vln.src.discrete.fall_helper import fall_by_problematic_data
from copy import deepcopy

class VLNEnv:
    def __init__(self, vln_config, scan, path_list, sim_config):
        self.vln_config = vln_config
        self.scan = scan
        self.path_list = path_list
        self.splits = vln_config.datasets.splits
        path_zero = self.path_list[0]
        scene_asset_path = load_scene_usd(vln_config, scan)
        sim_config.config.tasks[0].scene_asset_path = scene_asset_path
        start_position = np.array(path_zero["start_position"])
        start_rotation = np.array(path_zero["start_rotation"])
        sim_config.config.tasks[0].robots[0].position = start_position
        sim_config.config.tasks[0].robots[0].orientation = start_rotation
        self.sim_config = sim_config
        self.camera_list = [x.name for x in sim_config.config.tasks[0].robots[0].sensor_params if x.enable]

    def setup_env_and_robot(self):
        env = BaseEnv(self.sim_config, headless=self.vln_config.headless, webrtc=False)
        self.env = env
        task = env._runner.current_tasks[list(env._runner.current_tasks.keys())[0]]
        self.task = task
        the_robot = task.robots[list(task.robots.keys())[0]]
        self.robot = the_robot
        the_isaac_robot = the_robot.isaac_robot
        self.isaac_robot = the_isaac_robot
        path_zero = self.path_list[0]
        start_position = np.array(path_zero["start_position"])
        start_rotation = np.array(path_zero["start_rotation"])
        task.set_robot_poses_without_offset(start_position, start_rotation)
        the_isaac_robot.set_world_velocity(np.zeros(6))
        the_isaac_robot.set_joint_velocities(np.zeros(len(the_isaac_robot.dof_names)))
        the_isaac_robot.set_joint_positions(np.zeros(len(the_isaac_robot.dof_names)))
    
    def init_sample_utils_discrete(self):
        from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
        from vln.src.local_nav.global_topdown_map import GlobalTopdownMap
        self.topdown_map = GlobalTopdownMap(self.vln_config,self.scan)
        self.occupancy_map = CamOccupancyMap(self.vln_config, self.robot.sensors['topdown_camera_500'])
        self.stuck_checker = StuckChecker(self.task._offset,self.isaac_robot)
        self.path_planner = AStarDiscretePlanner(
            map_width = self.vln_config.maps.global_topdown_config.width,
            map_height= self.vln_config.maps.global_topdown_config.height,
            aperture = self.vln_config.maps.global_topdown_config.aperture,
            step_unit_meter = 0.25,
            angle_unit=15,
            max_step=50000,
        )
        progress_log_util.init(self.scan, len(self.path_list))
        progress_log_util.progress_logger.info(f"start sampling split: {self.splits[0]} scan: {self.scan}, total_path:{len(self.path_list)}")

    def init_sample_utils_continuous(self):
        from vln.src.local_nav.camera_occupancy_map import CamOccupancyMap
        from vln.src.local_nav.global_topdown_map import GlobalTopdownMap
        self.topdown_map = GlobalTopdownMap(self.vln_config,self.scan)
        self.occupancy_map = CamOccupancyMap(self.vln_config, self.robot.sensors['topdown_camera_500'])
        self.stuck_checker = StuckChecker(self.task._offset,self.isaac_robot)
        self.path_planner = AStarPlanner(
            args = self.vln_config,
            map_width = self.vln_config.maps.global_topdown_config.width,
            map_height= self.vln_config.maps.global_topdown_config.height,
            max_step=self.vln_config.planners.a_star_max_iter,
            windows_head=False,
            for_llm=False,
            verbose=False)
        progress_log_util.init(self.scan, len(self.path_list))
        progress_log_util.progress_logger.info(f"start sampling split: {self.splits[0]} scan: {self.scan}, total_path:{len(self.path_list)}")


    def warm_up(self, warm_step):
        step = 0
        warm_up_actions =[{'h1':{'stand_still': []}}]
        while(self.env.simulation_app.is_running()):
            step = step + 1
            '''(1) warm up process'''
            if step < warm_step:
                self.env.step(actions=warm_up_actions)
                continue
            elif step == warm_step:
                self.env.step(actions=warm_up_actions, add_rgb_subframes=True, render=True)
                break

    def stop(self):
        progress_log_util.report()
        if(hasattr(self.env, 'simulation_app')):
            self.env.simulation_app.close()

    def reset_single_robot(self,position, orientation):
        ''' Reset a single robot's pose
        '''
        self.task.set_single_robot_poses_without_offset(position, orientation)
        self.isaac_robot.set_world_velocity(np.zeros(6))
        self.isaac_robot.set_joint_velocities(np.zeros(len(self.isaac_robot.dof_names)))
        self.isaac_robot.set_joint_positions(np.zeros(len(self.isaac_robot.dof_names)))
        self.isaac_robot.set_joint_efforts(np.zeros(len(self.isaac_robot.dof_names)))
        robot_pos = self.task.get_robot_poses_without_offset()[0]
        self.occupancy_map.set_world_pose(robot_pos)

    def sample_one_path_discrete(self, path_index):
        step = 0
        env = self.env
        the_scan = self.scan
        the_data = self.path_list[path_index]
        the_path_id = the_data['trajectory_id']
        the_isaac_robot = self.isaac_robot
        the_robot = self.robot
        nav_path = the_data['reference_path']
        max_step = self.vln_config.settings.max_step
        
        finish = False
        finish_result = None
        current_point_index = 0
        need_path_plan = True
        action_list = []
        action_list_index = 0

        action_aggregation = []
        action_start_position = None
        action_start_rotation = None
        halfway_sample_count=0

        finish_one_action = True
        lmdb_path = os.path.join(self.vln_config.sample_episode_dir, self.vln_config.name)
        data_cache = SampleDataCache(lmdb_path = lmdb_path, path_id=the_path_id, instruction=the_data['instruction']['instruction_text'])
        self.reset_single_robot(the_data['start_position'], the_data['start_rotation'])

        # while(env.simulation_app.is_running()):
        #     env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
        while(env.simulation_app.is_running()):
            step = step + 1
            
            if finish:
                distance_str = "-"
                if finish_result == "success":
                    robot_position, _ = the_isaac_robot.get_world_pose()
                    x1,x2 = nav_path[-1][0],robot_position[0]
                    y1,y2 = nav_path[-1][1],robot_position[1]
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    distance_str = f"{round(distance, 2)}"

                log.info(f"[scan:{the_scan}][path:{the_path_id}] finish[step:{step}] result:{finish_result}, distance:{distance_str} m")
                if len(action_list) > 0 and (action_list_index + 1) < len(action_list):
                    data_cache.collect_action(action_list[action_list_index + 1])
                else:
                    data_cache.collect_action(0)
                data_cache.save_data(finish_result)
                progress_log_util.trace_end(
                    trajectory_id = the_path_id,
                    step_count=step,
                    result = finish_result,
                )
                break

            if step == 1:
                if fall_by_problematic_data(the_path_id):
                    for _ in range(100):
                        env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
                obs = env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=True, render=True)
                process = current_point_index / len(nav_path)
                camera_pose = self.task.get_camera_poses_without_offset('pano_camera_0')
                robot_pose = self.task.get_robot_poses_without_offset()
                data_cache.collect_observation(
                    obs = obs,
                    step = step,
                    process = process,
                    camera_pose = camera_pose,
                    robot_pose = robot_pose,
                )
                progress_log_util.trace_start(
                    trajectory_id = the_path_id,
                    step_count=step,
                )
                continue

            ''' (0) check the maximum step'''
            if step > max_step:
                log.error(f"[scan:{the_scan}][path:{the_path_id}]. Exceed the maximum steps: {max_step}")
                finish = True
                finish_result = 'max step'
                continue

            ''' (2) Check for the robot weather falls or stucks'''
            if step % 20 == 0:
                robot_position, robot_rotation = the_isaac_robot.get_world_pose()
                robot_bottom_z = the_robot.get_ankle_height() - self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
                is_fall = check_robot_fall(robot_position, robot_rotation, robot_bottom_z)
                if is_fall:
                    finish = True
                    finish_result = 'fall'
                    continue
                is_stuck = self.stuck_checker.check_robot_stuck(robot_position, robot_rotation, cur_iter=step, max_iter=2500, threshold=0.2)
                if is_stuck:
                    finish = True
                    finish_result = 'stuck'
                    continue
                
            '''(3) check for action finish status and update navigation'''
            if need_path_plan:
                if current_point_index == 0:
                    log.info(f"[scan:{the_scan}][path:{the_path_id}] robot starts navigating")
                if current_point_index < len(nav_path)-1:
                    
                    robot_position, robot_rotation = the_isaac_robot.get_world_pose()
                    action_list, real_points, find_flag = plan_and_get_actions_discrete(
                        goal = nav_path[current_point_index + 1],
                        occupancy_map = self.occupancy_map,
                        topdown_map = self.topdown_map,
                        robot_position=robot_position, 
                        robot_rotation = robot_rotation,
                        offset=self.task._offset,
                        aperture = self.vln_config.maps.global_topdown_config.aperture,
                        width = self.vln_config.maps.global_topdown_config.width,
                        height= self.vln_config.maps.global_topdown_config.height,
                        path_planner = self.path_planner,
                    )
                    if not find_flag or action_list is None or len(action_list) == 0:
                        if step > 20:
                            finish = True
                            finish_result = 'path planning'
                        continue
                    log.info(f"[scan:{the_scan}][path:{the_path_id}] robot is navigating to the {current_point_index + 1}-th target place.")
                    action_list_index = 0

                elif current_point_index == len(nav_path)-1:
                    finish = True
                    finish_result = 'success'
                    continue
                need_path_plan = False

            '''(4) Step and get new observations'''
            robot_position, robot_rotation = the_isaac_robot.get_world_pose()
            if finish_one_action:
                action_start_position = robot_position
                action_start_rotation = robot_rotation
                for i in range(action_list_index,len(action_list)):
                    if action_list[i] == action_list[action_list_index]:
                        action_aggregation.append(action_list[i])
                    else:
                        break
                env_actions = get_env_actions_aggregation(
                    action_aggregation = action_aggregation,
                    robot_position = robot_position,
                    robot_rotation = robot_rotation,
                )
                # print(f"=======action_list:{action_list},action_aggregation: {action_aggregation}")
                finish_one_action=False
            halfway_sample = is_halfway_sample_needed(action_aggregation,action_start_position,action_start_rotation,robot_position,robot_rotation,halfway_sample_count)
            obs = env.step(actions=env_actions, add_rgb_subframes=halfway_sample, render=halfway_sample)
            finish_one_action = is_one_action_finished(action_list[action_list_index], obs)
            
            halfway_collect = halfway_sample and halfway_sample_count < len(action_aggregation) - 1
            
            if halfway_collect or finish_one_action:
                if finish_one_action:
                    obs = env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=True, render=True)
                    step = step + 1
                
                halfway_sample_count = halfway_sample_count + 1
                process = current_point_index / len(nav_path)
                camera_pose = self.task.get_camera_poses_without_offset('pano_camera_0')
                robot_pose = self.task.get_robot_poses_without_offset()
                data_cache.collect_action(action_list[action_list_index])
                data_cache.collect_observation(
                    obs = obs,
                    step = step,
                    process = process,
                    camera_pose = camera_pose,
                    robot_pose = robot_pose,
                )
                log.info(f"[scan:{the_scan}][path:{the_path_id}] finish one action[halfway_sample_count:{halfway_sample_count}][step:{step}][ {action_list_index + halfway_sample_count} / {len(action_list)} ] {describe_action(action_list[action_list_index])}")

            if finish_one_action:
                action_list_index = action_list_index + len(action_aggregation)
                action_aggregation = []
                action_start_position = None
                action_start_rotation = None
                halfway_sample_count = 0
                
                if action_list[action_list_index - 1] == 1:
                    x1,x2 = real_points[action_list_index - 1][0],robot_position[0]
                    y1,y2 = real_points[action_list_index - 1][1],robot_position[1]
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if distance > 0.5:
                        log.info(f"[distance:{distance} > 0.5 ] replanning")
                        need_path_plan = True
                        continue
                else:
                    from omni.isaac.core.utils.rotations import quat_to_euler_angles,euler_angles_to_quat
                    _, _, real_yaw = quat_to_euler_angles(robot_rotation)
                    yaw_diff = abs(real_yaw - real_points[action_list_index - 1])
                    if yaw_diff > math.pi / 6:
                        log.info(f"[yaw_diff: {round(yaw_diff * (180 / math.pi))} 度 > 30 度] replanning")
                        need_path_plan = True
                        continue

                if action_list_index == len(action_list):
                    need_path_plan = True
                    current_point_index = current_point_index + 1


    def sample_one_path_continuous(self, path_index):
        step = 0
        env = self.env
        the_scan = self.scan
        the_data = self.path_list[path_index]
        the_path_id = the_data['trajectory_id']
        the_isaac_robot = self.isaac_robot
        the_robot = self.robot
        nav_path = the_data['reference_path']
        max_step = self.vln_config.settings.max_step
        
        finish = False
        finish_result = None
        current_point_index = 0
        need_path_plan = True
        exe_path=[]

        lmdb_path = os.path.join(self.vln_config.sample_episode_dir, self.vln_config.name)
        data_cache = SampleDataCache(lmdb_path = lmdb_path, path_id=the_path_id, instruction=the_data['instruction']['instruction_text'])
        self.reset_single_robot(the_data['start_position'], the_data['start_rotation'])

        # while(env.simulation_app.is_running()):
        #     env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
        while(env.simulation_app.is_running()):
            step = step + 1
            
            if finish:
                distance_str = "-"
                if finish_result == "success":
                    robot_position, _ = the_isaac_robot.get_world_pose()
                    x1,x2 = nav_path[-1][0],robot_position[0]
                    y1,y2 = nav_path[-1][1],robot_position[1]
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    distance_str = f"{round(distance, 2)}"

                log.info(f"[scan:{the_scan}][path:{the_path_id}] finish[step:{step}] result:{finish_result}, distance:{distance_str} m")
                data_cache.save_data(finish_result)
                progress_log_util.trace_end(
                    trajectory_id = the_path_id,
                    step_count=step,
                    result = finish_result,
                )
                break

            if step == 1:
                if fall_by_problematic_data(the_path_id):
                    for _ in range(100):
                        env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
                obs = env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=True, render=True)
                process = current_point_index / len(nav_path)
                camera_pose = self.task.get_camera_poses_without_offset('pano_camera_0')
                robot_pose = self.task.get_robot_poses_without_offset()
                data_cache.collect_observation(
                    obs = obs,
                    step = step,
                    process = process,
                    camera_pose = camera_pose,
                    robot_pose = robot_pose,
                )
                progress_log_util.trace_start(
                    trajectory_id = the_path_id,
                    step_count=step,
                )
                continue

            ''' (0) check the maximum step'''
            if step > max_step:
                log.error(f"[scan:{the_scan}][path:{the_path_id}]. Exceed the maximum steps: {max_step}")
                finish = True
                finish_result = 'max step'
                continue

            ''' (2) Check for the robot weather falls or stucks'''
            if step % 20 == 0:
                robot_position, robot_rotation = the_isaac_robot.get_world_pose()
                robot_bottom_z = the_robot.get_ankle_height() - self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']
                is_fall = check_robot_fall(robot_position, robot_rotation, robot_bottom_z)
                if is_fall:
                    finish = True
                    finish_result = 'fall'
                    continue
                is_stuck = self.stuck_checker.check_robot_stuck(robot_position, robot_rotation, cur_iter=step, max_iter=2500, threshold=0.2)
                if is_stuck:
                    finish = True
                    finish_result = 'stuck'
                    continue
                
            '''(3) check for action finish status and update navigation'''
            if need_path_plan:
                if current_point_index == 0:
                    log.info(f"[scan:{the_scan}][path:{the_path_id}] robot starts navigating")
                if current_point_index < len(nav_path)-1:
                    
                    robot_position, robot_rotation = the_isaac_robot.get_world_pose()
                    exe_path = plan_and_get_actions_continuous(
                        goal = nav_path[current_point_index + 1],
                        occupancy_map = self.occupancy_map,
                        topdown_map = self.topdown_map,
                        robot_position=robot_position, 
                        robot_rotation = robot_rotation,
                        offset=self.task._offset,
                        aperture = self.vln_config.maps.global_topdown_config.aperture,
                        width = self.vln_config.maps.global_topdown_config.width,
                        height= self.vln_config.maps.global_topdown_config.height,
                        path_planner = self.path_planner,
                    )
                    if exe_path is None or len(exe_path) == 0:
                        if step <= 20:
                            continue
                        else:
                            finish = True
                            finish_result = 'path planning'
                        continue
                    log.info(f"[scan:{the_scan}][path:{the_path_id}] robot is navigating to the {current_point_index + 1}-th target place.")
                    exe_path_new = deepcopy(exe_path)
                    for idx in range(len(exe_path)):
                        exe_path_new[idx] += np.array(self.task._offset)
                    exe_path = exe_path_new

                elif current_point_index == len(nav_path)-1:
                    finish = True
                    finish_result = 'success'
                    continue
                need_path_plan = False

            '''(4) Step and get new observations'''
            actions=[{'h1':{'move_along_path': [exe_path]}}]
            need_sample = step % (self.vln_config.sample_episodes.step_interval - 1) == 0
            obs = env.step(actions=actions, add_rgb_subframes=need_sample, render=need_sample)
            
            if need_sample:
                process = current_point_index / len(nav_path)
                camera_pose = self.task.get_camera_poses_without_offset('pano_camera_0')
                robot_pose = self.task.get_robot_poses_without_offset()
                data_cache.collect_observation(
                    obs = obs,
                    step = step,
                    process = process,
                    camera_pose = camera_pose,
                    robot_pose = robot_pose,
                )
            action_finished = is_action_finished('move_along_path', obs)
            if action_finished:
                need_path_plan = True
                current_point_index = current_point_index + 1