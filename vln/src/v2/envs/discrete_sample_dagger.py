from vln.src.v2.envs.discrete_sample import DiscreteSampleSingleScanEnv
from grutopia.core.config import SimulatorConfig
from vln.src.v2.dataloader.sample import SamplePathKeyDataloader
from grutopia.core.util.log import log
from vln.src.v2.util import progress_log_util
from vln.src.v2.util.discrete_planner import AStarDiscretePlanner
from vln.src.v2.util.common import check_robot_fall, describe_action
from vln.src.v2.util.stuck_checker import StuckChecker
from vln.src.v2.util.data_collector import DataCollector
from vln.src.models.init_policy import initialize_policy
from vln.src.v2.util.discrete_dagger_controller import DiscreteDaggerController
import numpy as np
import torch


class DiscreteSampleDaggerSingleScanEnv(DiscreteSampleSingleScanEnv):

    def __init__(
        self,
        sim_config:SimulatorConfig,
        scene_asset_path,
        start_position,
        start_rotation,
        headless,
        dataloader:SamplePathKeyDataloader,
        eval_config,
        aperture=200,
        max_step=25000,
        policy_probability=0.2,
    ):
        super().__init__(
            sim_config=sim_config,
            scene_asset_path=scene_asset_path,
            start_position=start_position,
            start_rotation=start_rotation,
            headless=headless,
            dataloader=dataloader,
            aperture=aperture,
            max_step=max_step,
        )
        self.policy_probability = policy_probability
        self.eval_config = eval_config
        #TODO:
        self.device = torch.device("cuda", 0)
        policy, _, _, _ = initialize_policy(
            self.eval_config,
            log,
            load_from_ckpt=True,
            device=self.device,
            load_from_pretrain=False,
            action_stats=None,
        )
        self.policy = policy
        self.rnn_states = torch.zeros(
            1,
            self.policy.num_recurrent_layers,
            self.eval_config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        
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
            episode_id = path_key.split('_')[1]
            log.info(f"split: {split}")
            log.info(f"scan: {scan}")
            log.info(f"trajectory_id: {trajectory_id}")
            log.info(f"episode_id: {episode_id}")
            log.info(f"data: {data}")
            progress_log_util.trace_start(
                trajectory_id = path_key,
                step_count=0,
            )
            dagger_controller = DiscreteDaggerController(data, self)
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
            stuck_checker = StuckChecker(self.task._offset,self.isaac_robot)

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
                    dagger_controller.report()
                    break

                action, action_type, rgb, depth, reason = dagger_controller.get_next_action(current_point_index)
                if action is None:
                    finish = True
                    result = 'path planning'
                    if reason is not None:
                        result = reason
                    continue
                
                env_action = [{'h1': {'move_by_descrete': [action]}}]
                data_collector.collect_observation(
                    rgb=rgb,
                    depth=depth,
                    step=self.step,
                    process=current_point_index / len(nav_path),
                    camera_pose=self.task.get_camera_poses_without_offset('pano_camera_0'),
                    robot_pose=self.task.get_robot_poses_without_offset(),
                )
                data_collector.collect_action(action)
                action_success, fail_reason = self.execute_one_action(env_action,stuck_checker)
                self.prev_action = action
                log.info(f"[scan:{scan}][path:{path_key}] finish one action[step:{self.step}][action_type:{action_type}][ {current_point_index + 1} / {len(nav_path)} ][{fail_reason}] {describe_action(action)}")
                if not action_success:
                    finish = True
                    result = fail_reason
                    continue
                robot_position, robot_rotation = self.task.get_robot_poses_without_offset()
                distance = np.linalg.norm(robot_position[:2] - nav_path[current_point_index + 1][:2])
                if distance < 0.25:
                    current_point_index += 1
                    if current_point_index == len(nav_path) - 1:
                        finish = True
                        result = 'success'

        progress_log_util.report()
