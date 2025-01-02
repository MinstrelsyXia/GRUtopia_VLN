from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
import numpy as np
from vln.src.v2.util.common import(
    create_robot_mask,
    freemap_to_accupancy_map,
)

class BaseSingleScanEnv:
    def __init__(
            self,
            sim_config:SimulatorConfig,
            scene_asset_path,
            start_position,
            start_rotation,
            headless,
        ):
        self.sim_config = sim_config
        self.scene_asset_path = scene_asset_path
        self.start_position = start_position
        self.start_rotation = start_rotation
        self.headless = headless
        self.env = None
        self.task = None
        self.robot = None
        self.isaac_robot = None

    def load_scan_and_robot(self):
        self.sim_config.config.tasks[0].scene_asset_path = self.scene_asset_path
        self.sim_config.config.tasks[0].robots[0].position = self.start_position
        self.sim_config.config.tasks[0].robots[0].orientation = self.start_rotation
        self.env = BaseEnv(self.sim_config, headless=self.headless, webrtc=False)
        self.task = self.env._runner.current_tasks[list(self.env._runner.current_tasks.keys())[0]]
        self.robot = self.task.robots[list(self.task.robots.keys())[0]]
        self.isaac_robot = self.robot.isaac_robot
        self.topdown_global_map_camera = self.robot.sensors['topdown_camera_500']
    
    def reset_robot(
        self,
        position,
        rotation,
    ):
        self.task.set_single_robot_poses_without_offset(position, rotation)
        self.isaac_robot.set_world_velocity(np.zeros(6))
        self.isaac_robot.set_joint_velocities(np.zeros(len(self.isaac_robot.dof_names)))
        self.isaac_robot.set_joint_positions(np.zeros(len(self.isaac_robot.dof_names)))
        self.isaac_robot.set_joint_efforts(np.zeros(len(self.isaac_robot.dof_names)))
    
    def get_global_map(
        self,
        robot_height,
        dilation_iterations=0,
        voxel_size=0.1,
        agent_radius=0.25,
    ):
        # 获取 free_map
        min_height = robot_height
        max_height = robot_height + 0.8
        data_info = self.topdown_global_map_camera.get_data()
        depth = np.array(data_info["depth"])
        flat_surface_mask = np.ones_like(depth, dtype=bool)
        depth_mask = ((depth >= min_height) & (depth < max_height)) | ((depth <= 0.5) & (depth > 0.02))
        robot_mask = create_robot_mask(self.topdown_global_map_camera)
        free_map = np.zeros_like(depth, dtype=int)
        free_map[flat_surface_mask & depth_mask] = 1
        free_map[robot_mask == 1] = 1
        accupancy_map = freemap_to_accupancy_map(
            topdown_global_map_camera=self.topdown_global_map_camera,
            freemap=free_map, 
            dilation_iterations=dilation_iterations,
            voxel_size=voxel_size,
            agent_radius=agent_radius,
        )
        return accupancy_map
    
    def warm_up(self, step_count):
        for _ in range(step_count - 1):
            self.env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=False, render=False)
        self.env.step(actions=[{'h1':{'stand_still': []}}], add_rgb_subframes=True, render=True)