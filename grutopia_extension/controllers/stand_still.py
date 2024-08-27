from typing import List

import numpy as np
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.types import ArticulationAction

from grutopia.core.robot.controller import BaseController
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.robot.robot_model import ControllerModel
from grutopia.core.util.interaction import BaseInteraction


@BaseController.register('StandStillController')
class StandStillController(BaseController):
    """Stand Still Controller.
    """

    def __init__(self, config: ControllerModel, robot: BaseRobot, scene: Scene) -> None:
        self.config = config
        self.forward_speed_base = config.forward_speed
        self.rotation_speed_base = config.rotation_speed
        self.lateral_speed_base = config.lateral_speed

        super().__init__(config=config, robot=robot, scene=scene)

    def forward(self) -> ArticulationAction:
        forward_speed = 0
        lateral_speed = 0
        rotation_speed = 0

        return self.sub_controllers[0].forward(forward_speed=forward_speed,
                                               rotation_speed=rotation_speed,
                                               lateral_speed=lateral_speed)

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): 0-element 1d array.
        """
        assert len(action) == 0, 'action must be empty'
        return self.forward()
