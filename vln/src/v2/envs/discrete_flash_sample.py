from .base import BaseSingleScanEnv
from grutopia.core.config import SimulatorConfig
from vln.src.v2.dataloader.sample import SamplePathKeyDataloader
from vln.src.v2.util.common_log_util import common_logger as log
from vln.src.v2.util import progress_log_util
from vln.src.v2.util.discrete_planner import AStarDiscretePlanner
from vln.src.v2.util.path_plan import plan_and_get_actions_discrete
from vln.src.v2.util.common import (
    check_robot_fall, 
    describe_action, 
    reset_topdown_camera,
    get_new_position_and_rotation
)
from vln.src.v2.util.data_collector import DataCollector
import numpy as np
import math

class DiscreteFlashSampleSingleScanEnv(BaseSingleScanEnv):
    
    def __init__(
            self,
            sim_config:SimulatorConfig,
            scene_asset_path,
            start_position,
            start_rotation,
            headless,
            dataloader:SamplePathKeyDataloader,
            aperture=200,
            max_step=25000,
        ):
        super().__init__(
            sim_config=sim_config,
            scene_asset_path=scene_asset_path,
            start_position=start_position,
            start_rotation=start_rotation,
            headless=headless,
        )
        self.dataloader = dataloader
        self.aperture = aperture
        self.max_step = max_step
        self.robot_ankle_height = self.sim_config.config_dict['tasks'][0]['robots'][0]['ankle_height']

    def execute_one_action(
        self,
        action,
    ):
        robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
        new_robot_position,new_robot_rotation = get_new_position_and_rotation(robot_position,robot_rotation,action)
        self.reset_robot(new_robot_position,new_robot_rotation)
        reset_topdown_camera(self.robot)
        self.env.step(actions=[{'h1':{'stand_still': []}}], render=True)
        return True, 'success'

    def sample(self):
        self.load_scan_and_robot()
        height, width = self.topdown_global_map_camera._camera._resolution
        self.path_planner = AStarDiscretePlanner(
            map_width = width,
            map_height= height,
            aperture = self.aperture,
            step_unit_meter = 0.25,
            angle_unit = 15,
            max_step = 50000,
        )
        sample_path_key_list = self.dataloader.sample_path_key_list
        path_key_data = self.dataloader.path_key_data
        path_key_split = self.dataloader.path_key_split
        scan = self.dataloader.target_scan

        progress_log_util.init(scan, len(sample_path_key_list), rank=self.dataloader.rank)
        progress_log_util.progress_logger.info(f"start sample scan: {scan}, total_path:{len(sample_path_key_list)}")

        for path_key in sample_path_key_list:
            split = path_key_split[path_key]
            data = path_key_data[path_key]
            nav_path = data['reference_path']
            trajectory_id = path_key.split('_')[0]
            log.info(f"split: {split}")
            log.info(f"scan: {scan}")
            log.info(f"trajectory_id_episode_id: {path_key}")
            log.info(f"data: {data}")
            progress_log_util.trace_start(
                trajectory_id = path_key,
                step_count=0,
            )
            data_collector = DataCollector(
                lmdb_path=self.dataloader.lmdb_path,
                key=str(trajectory_id),
                instruction=data['instruction']['instruction_text'],
            )
            start_position = data['start_position']
            start_rotation = data['start_rotation']
            self.reset_robot(start_position, start_rotation)
            self.warm_up(240)
            robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
            robot_bottom_z = self.robot.get_ankle_height() - self.robot_ankle_height
            is_fall = check_robot_fall(robot_position, robot_rotation, robot_bottom_z)
            if is_fall:
                progress_log_util.trace_end(
                    trajectory_id = path_key,
                    step_count=0,
                    result = 'fast_fall',
                )
                data_collector.save_data('fast_fall')
                log.info(f"[scan:{scan}][path:{trajectory_id}] finish[step:0] result: fast_fall")
                continue

            finish = False
            result = None
            current_point_index = 0
            self.step = 0

            while True:
                if finish:
                    distance_str = "-"
                    if result == "success":
                        data_collector.collect_observation_by_env(
                            env=self.env,
                            step=self.step,
                            process=current_point_index / len(nav_path),
                            camera_pose=self.task.get_camera_poses_without_offset('pano_camera_0'),
                            robot_pose=self.task.get_robot_poses_without_offset(),
                        )
                        data_collector.collect_action([0])
                        robot_position, _ = self.isaac_robot.get_world_pose()
                        distance = np.linalg.norm(robot_position[:2] - nav_path[-1][:2])
                        distance_str = f"{round(distance, 2)}"
                    log.info(f"[scan:{scan}][path:{trajectory_id}] finish[step:{self.step}] result:{result}, distance:{distance_str} m")
                    data_collector.save_data(result)
                    progress_log_util.trace_end(
                        trajectory_id = path_key,
                        step_count=self.step,
                        result = result,
                    )
                    break
                map_info = self.get_global_map(robot_height=1.55, dilation_iterations=2)
                camera_pose = self.topdown_global_map_camera.get_world_pose()[0] - self.task._offset
                
                robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
                # path_plan
                action_list, _, find_flag, reason = plan_and_get_actions_discrete(
                    map_info=map_info,
                    robot_position=robot_position,
                    robot_rotation=robot_rotation,
                    goal = nav_path[current_point_index + 1],
                    camera_pose = camera_pose,
                    aperture=self.aperture,
                    width=width,
                    height=height,
                    path_planner=self.path_planner,
                )
                if not find_flag or action_list is None or len(action_list) == 0:
                    finish = True
                    result = 'path planning'
                    if reason is not None:
                        result = reason
                    continue
                
                action_index = 0
                for action in action_list:
                    data_collector.collect_observation_by_env(
                        env=self.env,
                        step=self.step,
                        process=current_point_index / len(nav_path),
                        camera_pose=self.task.get_camera_poses_without_offset('pano_camera_0'),
                        robot_pose=self.task.get_robot_poses_without_offset(),
                    )
                    data_collector.collect_action(action)
                    action_success, fail_reason = self.execute_one_action(action)
                    log.info(f"[scan:{scan}][path:{path_key}] finish one action[step:{self.step}][ {action_index + 1} / {len(action_list)} ][result:{fail_reason}] {describe_action(action)}")
                    if not action_success:
                        finish = True
                        result = fail_reason
                        break
                    action_index +=1
                if action_index == len(action_list):
                    current_point_index += 1
                    if current_point_index == len(nav_path) - 1:
                        finish = True
                        result = 'success'
        
        progress_log_util.report()
