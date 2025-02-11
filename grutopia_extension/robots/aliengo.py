import os
from typing import Dict

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots.robot import Robot as IsaacRobot
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction, ArticulationActions
import omni.isaac.core.utils.numpy.rotations as rot_utils

import grutopia.core.util.string as string_utils
from grutopia.actuators import ActuatorBase, ActuatorBaseCfg, DCMotorCfg
from grutopia.core.config.robot import RobotUserConfig as Config
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.robot.robot_model import RobotModel
from grutopia.core.util import log


class Aliengo(IsaacRobot):

    actuators_cfg: dict = {
        'base_legs':
        DCMotorCfg(
            joint_names_expr=['.*_hip_joint', '.*_thigh_joint', '.*_calf_joint'],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
    }

    def __init__(self,
                 prim_path: str,
                 usd_path: str,
                 name: str,
                 position: np.ndarray = None,
                 orientation: np.ndarray = None,
                 scale: np.ndarray = None):
        add_reference_to_stage(prim_path=prim_path, usd_path=os.path.abspath(usd_path))
        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation, scale=scale)
        self.actuators: Dict[str, ActuatorBase]

    def set_gains(self, gains):
        """[summary]

        Args:
            kps (Optional[np.ndarray], optional): [description]. Defaults to None.
            kds (Optional[np.ndarray], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        num_leg_joints = 12
        kps = np.array([0.] * num_leg_joints)
        kds = np.array([0.] * num_leg_joints)

        if kps is not None:
            kps = self._articulation_view._backend_utils.expand_dims(kps, 0)
        if kds is not None:
            kds = self._articulation_view._backend_utils.expand_dims(kds, 0)
        self._articulation_view.set_gains(kps=kps, kds=kds, save_to_usd=False)

        # VERY IMPORTANT!!! additional physics parameter
        self._articulation_view.set_solver_position_iteration_counts(
            self._articulation_view._backend_utils.expand_dims(8, 0))
        self._articulation_view.set_solver_velocity_iteration_counts(
            self._articulation_view._backend_utils.expand_dims(0, 0))
        self._articulation_view.set_enabled_self_collisions(self._articulation_view._backend_utils.expand_dims(True, 0))

    def _process_actuators_cfg(self):
        self.actuators = dict.fromkeys(Aliengo.actuators_cfg.keys())
        for actuator_name, actuator_cfg in Aliengo.actuators_cfg.items():
            # type annotation for type checkersc
            actuator_cfg: ActuatorBaseCfg
            # create actuator group
            joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)

            stiffness, damping = self._articulation_view.get_gains()
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=1,
                device='cpu',
                stiffness=self._articulation_view.get_gains()[0][0],
                damping=self._articulation_view.get_gains()[1][0],
                armature=torch.tensor(self._articulation_view.get_armatures()),
                friction=torch.tensor(self._articulation_view.get_friction_coefficients()),
                effort_limit=torch.tensor(self._articulation_view._physics_view.get_dof_max_forces()),
                velocity_limit=torch.tensor(self._articulation_view._physics_view.get_dof_max_velocities()),
            )
            # log information on actuator groups
            self.actuators[actuator_name] = actuator

    def apply_actuator_model(self, control_action: ArticulationAction, controller_name: str,
                             joint_set: ArticulationSubset):
        actuator = self.actuators['base_legs']

        control_joint_pos = torch.tensor(control_action.joint_positions, dtype=torch.float32)
        control_actions = ArticulationActions(
            joint_positions=control_joint_pos,
            joint_velocities=torch.zeros_like(control_joint_pos),
            joint_efforts=torch.zeros_like(control_joint_pos),
            joint_indices=actuator.joint_indices,
        )

        joint_pos = torch.tensor(joint_set.get_joint_positions(), dtype=torch.float32)
        joint_vel = torch.tensor(joint_set.get_joint_velocities(), dtype=torch.float32)
        control_actions = actuator.compute(
            control_actions,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )
        if control_actions.joint_positions is not None:
            joint_set.set_joint_positions(control_actions.joint_positions)
        if control_actions.joint_velocities is not None:
            joint_set.set_joint_velocities(control_actions.joint_velocities)
        if control_actions.joint_efforts is not None:
            joint_set.set_joint_efforts(control_actions.joint_efforts)

    def find_joints(self, name_keys, joint_subset=None):
        """Find joints in the articulation based on the name keys.

        Please see the :func:`omni.isaac.orbit.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self._articulation_view.dof_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset)


@BaseRobot.register('AliengoRobot')
class AliengoRobot(BaseRobot):

    def __init__(self, config: Config, robot_model: RobotModel, scene: Scene):
        super().__init__(config, robot_model, scene)
        self._sensor_config = robot_model.sensors
        self._gains = robot_model.gains
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'aliengo {config.name}: position    : ' + str(self._start_position))
        log.debug(f'aliengo {config.name}: orientation : ' + str(self._start_orientation))

        usd_path = robot_model.usd_path
        if usd_path.startswith('/Isaac'):
            usd_path = get_assets_root_path() + usd_path

        log.debug(f'aliengo {config.name}: usd_path         : ' + str(usd_path))
        log.debug(f'aliengo {config.name}: config.prim_path : ' + str(config.prim_path))
        self.isaac_robot = Aliengo(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=usd_path,
        )

        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
            self.isaac_robot.set_local_scale(self._robot_scale)

        self._robot_ik_base = None

        self._robot_base = RigidPrim(prim_path=config.prim_path + '/base', name=config.name + '_base')
        
        self._robot_right_ankle = RigidPrim(prim_path=config.prim_path + '/RR_foot', name=config.name + 'rr_foot')
        self._robot_left_ankle = RigidPrim(prim_path=config.prim_path + '/FL_foot', name=config.name + 'fl_foot')

    def post_reset(self):
        super().post_reset()
        self.isaac_robot._process_actuators_cfg()
        if self._gains is not None:
            self.isaac_robot.set_gains(self._gains)

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_base(self) -> RigidPrim:
        return self._robot_base

    def get_robot_ik_base(self):
        return self._robot_ik_base

    def get_world_pose(self):
        return self._robot_base.get_world_pose()

    def get_ankle_height(self):
        return np.min([self._robot_right_ankle.get_world_pose()[0][2], self._robot_left_ankle.get_world_pose()[0][2]])

    def apply_action(self, action: dict):
        """
        Args:
            action (dict): inputs for controllers.
        """
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warn(f'unknown controller {controller_name} in action')
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.isaac_robot.apply_actuator_model(control, controller_name, controller.get_joint_subset())
        
        # top-down camera reset
        if 'topdown_camera_500' in self.sensors:
            orientation_quat = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
            robot_pos = self.isaac_robot.get_world_pose()[0]
            self.sensors['topdown_camera_500']._camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2]+0.15],orientation_quat)
        
        if 'topdown_camera_50' in self.sensors:
            orientation_quat = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
            robot_pos = self.isaac_robot.get_world_pose()[0]
            self.sensors['topdown_camera_50']._camera.set_world_pose([robot_pos[0], robot_pos[1], robot_pos[2]+0.15],orientation_quat)

    def get_obs(self, add_rgb_subframes=False):
        position, orientation = self._robot_base.get_world_pose()

        # custom
        obs = {
            'position': position,
            'orientation': orientation,
        }

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs[c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs[sensor_name] = sensor_obs.get_data(add_rgb_subframes=add_rgb_subframes)
        return obs
