from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.types import ArticulationAction

from grutopia.core.robot.controller import BaseController
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.robot.robot_model import ControllerModel
from grutopia.core.util import log

@BaseController.register('MoveAlongSpeedsController')
class MoveAlongSpeedsController(BaseController):
    """Controller for executing a sequence of speeds using a speed controller as sub-controller."""

    def __init__(self, config: ControllerModel, robot: BaseRobot, scene: Scene) -> None:
        self._user_config = None
        self.speeds: List[np.ndarray | List] = []  # 每个元素包含 [forward_speed, lateral_speed, rotation_speed]
        self.speed_idx = 0
        self.current_speed: np.ndarray | None = None
        
        # 每个速度执行的时间步数
        self.steps_per_speed = config.steps_per_speed if config.steps_per_speed is not None else 60
        self.current_steps = 0

        super().__init__(config=config, robot=robot, scene=scene)

    def forward(self, speeds: List[np.ndarray]) -> ArticulationAction:
        if self.speeds is not speeds:
            self.speeds = speeds
            self.speed_idx = 0
            self.current_steps = 0
            log.info('reset speeds')
            self.current_speed = np.array(deepcopy(self.speeds[self.speed_idx]))

        # 检查是否需要切换到下一个速度
        if self.current_steps >= self.steps_per_speed:
            if self.speed_idx < len(self.speeds) - 1:
                self.speed_idx += 1
                self.current_speed = np.array(deepcopy(self.speeds[self.speed_idx]))
                self.current_steps = 0
                log.info(f'switch to next speed: {self.current_speed}')

        self.current_steps += 1
        
        return self.sub_controllers[0].forward(
            forward_speed=float(self.current_speed[0]),
            lateral_speed=float(self.current_speed[1]), 
            rotation_speed=float(self.current_speed[2])
        )

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """Convert input action (in 1d array format) to joint signals to apply.

        Args:
            action (List | np.ndarray): 1-element 1d array containing
              0. speeds list (List[np.ndarray]), 每个元素是 [forward_speed, lateral_speed, rotation_speed]

        Returns:
            ArticulationAction: joint signals to apply.
        """
        assert len(action) == 1, 'action must contain 1 element'
        assert len(action[0]) > 0, 'speeds cannot be empty'
        return self.forward(speeds=action[0])

    def get_obs(self) -> Dict[str, Any]:
        finished = False
        total_speeds = len(self.speeds)
        if total_speeds > 0 and self.speed_idx == total_speeds - 1:
            if self.current_steps >= self.steps_per_speed:
                finished = True

        return {
            'current_index': self.speed_idx,
            'current_speed': self.current_speed,
            'total_speeds': total_speeds,
            'current_steps': self.current_steps,
            'finished': finished,
        }