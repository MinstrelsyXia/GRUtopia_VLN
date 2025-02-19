from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
import numpy as np
from vln.src.v2.util.common import(
    create_robot_mask,
    freemap_to_accupancy_map,
    set_seed,
    visualize_freemap,
)
import time
import sys

class BaseSingleScanEnv:
    def __init__(
            self,
            robot_name,
            sim_config:SimulatorConfig,
            scene_asset_path,
            start_position,
            start_rotation,
            headless,
        ):
        self.robot_name = robot_name
        self.sim_config = sim_config
        self.scene_asset_path = scene_asset_path
        self.start_position = start_position
        self.start_rotation = start_rotation
        self.headless = headless
        self.env = None
        self.task = None
        self.robot = None
        self.isaac_robot = None
        self.timestamp = time.time()
        self.fall_height_threshold = self.sim_config.config_dict['tasks'][0]['robots'][0]['fall_height_threshold']
        self.robot_height = self.sim_config.config_dict['tasks'][0]['robots'][0]['robot_height']

        # fall check
        self.fall_height_threshold = self.sim_config.config_dict['tasks'][0]['robots'][0]['fall_height_threshold']
        self.robot_height = self.sim_config.config_dict['tasks'][0]['robots'][0]['robot_height']

    def update_timestamp(self):
        self.timestamp = time.time()
        sys.stdout.flush()

    def create_light(self):
        from pxr import Gf, UsdLux, UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        distant_light = UsdLux.DistantLight.Define(stage, "/World/distant_light")
        distant_light.CreateIntensityAttr(1000)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        up_disk_light = UsdLux.DiskLight.Define(stage, "/World/up_disk_light")
        up_disk_light.CreateIntensityAttr(self.sim_config.config_dict['tasks'][0]['disk_light_intensity']) 
        up_disk_light.CreateRadiusAttr(50.0)
        up_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(up_disk_light).AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
        self.up_disk_light = up_disk_light
        self.up_disk_light_position = UsdGeom.Xformable(self.up_disk_light).AddTranslateOp()
        
        down_disk_light = UsdLux.DiskLight.Define(stage, "/World/down_disk_light")
        down_disk_light.CreateIntensityAttr(self.sim_config.config_dict['tasks'][0]['disk_light_intensity'])
        down_disk_light.CreateRadiusAttr(50.0)
        down_disk_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        self.down_disk_light = down_disk_light
        self.down_disk_light_position = UsdGeom.Xformable(self.down_disk_light).AddTranslateOp()

    def reset_light_position(self, position):
        from pxr import Gf
        raise_light = 1
        if self.robot_name == 'aliengo':
           raise_light+= 0.55 
        self.up_disk_light_position.Set(Gf.Vec3f(position[0],  position[1],   -position[2] - raise_light))
        self.down_disk_light_position.Set(Gf.Vec3f(position[0],  position[1],   position[2] + raise_light))
    

    def load_scan_and_robot(self):
        self.sim_config.config.tasks[0].scene_asset_path = self.scene_asset_path
        self.sim_config.config.tasks[0].robots[0].position = self.start_position
        self.sim_config.config.tasks[0].robots[0].orientation = self.start_rotation
        self.env = BaseEnv(self.sim_config, headless=self.headless, webrtc=False)
        set_seed(0)
        self.create_light()
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
        self.reset_light_position(position)
    
    def get_global_map(
        self,
        robot_height,
        dilation_iterations=0,
        voxel_size=0.1,
        agent_radius=0.25,
        robot_name='h1'
    ):
        # 获取 free_map
        min_height = robot_height
        max_height = robot_height + 0.8
        data_info = self.topdown_global_map_camera.get_data()
        depth = np.array(data_info["depth"])
        flat_surface_mask = np.ones_like(depth, dtype=bool)
        if self.robot_name == 'h1':
            depth_mask = ((depth >= min_height) & (depth < max_height)) | ((depth <= 0.5) & (depth > 0.02))
        elif self.robot_name == 'aliengo':
            base_height = self.robot.get_robot_base().get_world_pose()[0][2]
            foot_height = self.robot.get_ankle_height()
            min_height = base_height - foot_height + 0.05
            depth_mask = ((depth >= min_height) & (depth < max_height))
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
        visualize_freemap(free_map, accupancy_map, save_path='logs/map0.png') # 20250211: debug
        return accupancy_map
    
    def warm_up(self, step_count):
        for _ in range(step_count - 1):
            self.env.step(actions=[{self.robot_name:{'stand_still': []}}], add_rgb_subframes=False, render=False)
        obs = self.env.step(actions=[{self.robot_name:{'stand_still': []}}], add_rgb_subframes=True, render=True)
        return obs
    
    def stop(self):
        if(hasattr(self.env, 'simulation_app')):
            self.env.simulation_app.close()
    
    