from typing import List

import numpy as np
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.types import ArticulationAction

from grutopia.core.robot.controller import BaseController
from grutopia.core.robot.robot import BaseRobot
# from grutopia_extension.configs.controllers import DifferentialDriveControllerCfg


@BaseController.register('DifferentialDriveController')
class DifferentialDriveController(BaseController):
    def __init__(self, config, robot: BaseRobot, scene: Scene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)

    def forward(
        self,
        forward_speed: float = 0,
        rotation_speed: float = 0,
        lateral_speed: float = 0,
        scaler: float = 1,
    ) -> ArticulationAction:
        # TODO: this is not the real forward and rotate speed !!
        if forward_speed == 0 and rotation_speed == 0:
            return ArticulationAction(joint_velocities=np.array([0, 0]))

        # Compensate speed scaler when scaler is too small.
        speed_scaler = scaler
            
        if scaler < 0.05:
            speed_scaler = scaler * 15
        if scaler < 0.03:
            speed_scaler = scaler * 25

        # Basis (component) vectors which can be multiplied by speed
        # (and angle for speed) to move forward or spin
        forward_basis = np.array([1.0, 1.0])
        spin_basis = np.array([-1.0, 1.0])

        wheel_vel_for = forward_basis * forward_speed
        wheel_vel_rot = spin_basis * rotation_speed
        wheel_vel = (wheel_vel_for + wheel_vel_rot) * speed_scaler

        return ArticulationAction(joint_velocities=wheel_vel)

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): n-element 1d array containing:
              0. forward_speed (float)
              1. lateral_speed (float)
              1. rotation_speed (float)
        """
        assert len(action) == 3, 'action must contain 2 elements'
        return self.forward(
            forward_speed=action[0],
            rotation_speed=action[2],
            scaler=1 / self.robot.get_robot_scale()[0],
        )
        # w61 update: this is for the same inputs with other legged robots
