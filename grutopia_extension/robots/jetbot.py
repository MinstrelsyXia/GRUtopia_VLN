import numpy as np
from omni.isaac.core.scenes import Scene
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.prims import RigidPrim

from grutopia.core.robot.robot import BaseRobot
from grutopia.core.util import log
# from grutopia_extension.configs.robots.jetbot import JetbotRobotCfg
from grutopia.core.robot.robot_model import RobotModel
import omni.isaac.core.utils.numpy.rotations as rot_utils


@BaseRobot.register('JetbotRobot')
class JetbotRobot(BaseRobot):
    def __init__(self, config, robot_model: RobotModel, scene: Scene):
        super().__init__(config, robot_model, scene)
        self._sensor_config = robot_model.sensors
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'jetbot {config.name} position    : ' + str(self._start_position))
        log.debug(f'jetbot {config.name} orientation : ' + str(self._start_orientation))

        usd_path = robot_model.usd_path

        log.debug(f'jetbot {config.name} usd_path         : ' + str(usd_path))
        log.debug(f'jetbot {config.name} config.prim_path : ' + str(config.prim_path))
        self.prim_path = str(config.prim_path)
        self.isaac_robot = WheeledRobot(
            prim_path=config.prim_path,
            name=config.name,
            wheel_dof_names=['left_wheel_joint', 'right_wheel_joint'],
            create_robot=True,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=usd_path,
        )

        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
            self.isaac_robot.set_local_scale(self._robot_scale)
        
        self._robot_base = RigidPrim(prim_path=config.prim_path + '/chassis', name=config.name + '_base')
        self._robot_left_wheel = RigidPrim(prim_path=config.prim_path + '/left_wheel', name=config.name + '_left_wheel')
        self._robot_right_wheel = RigidPrim(prim_path=config.prim_path + '/right_wheel', name=config.name + '_right_wheel')

    def get_robot_scale(self):
        return self._robot_scale

    def get_world_pose(self):
        return self.isaac_robot.get_world_pose()

    def get_robot_base(self) -> RigidPrim:
        return self._robot_base

    def get_ankle_height(self):
        return np.min([self._robot_left_wheel.get_world_pose()[0][2], self._robot_right_wheel.get_world_pose()[0][2]])

    def apply_action(self, action: dict):
        """
        Args:
            action (dict): inputs for controllers.
        """
        for controller_name, controller_action in action.items():
            # print(controller_name, "=============", controller_action)
            if controller_name not in self.controllers:
                log.warning(f'unknown controller {controller_name} in action')
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.isaac_robot.apply_action(control)
            
        # top-down camera reset
        if 'topdown_camera_500' in self.sensors:
            orientation_quat = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
            robot_pos = self.isaac_robot.get_world_pose()[0]
            self.sensors['topdown_camera_500']._camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2]+0.2],orientation_quat)
        
        if 'topdown_camera_50' in self.sensors:
            orientation_quat = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
            robot_pos = self.isaac_robot.get_world_pose()[0]
            self.sensors['topdown_camera_50']._camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2]+0.2],orientation_quat)

    def get_obs(self, add_rgb_subframes=False):
        position, orientation = self.isaac_robot.get_world_pose()

        # custom
        obs = {
            'position': position,
            'orientation': orientation,
            'joint_positions': self.isaac_robot.get_joint_positions(),
            'joint_velocities': self.isaac_robot.get_joint_velocities(),
            'controllers': {},
            'sensors': {},
        }

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs[c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs[sensor_name] = sensor_obs.get_data(add_rgb_subframes=add_rgb_subframes)
        return obs
