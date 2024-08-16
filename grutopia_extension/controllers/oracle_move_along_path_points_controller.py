from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.types import ArticulationAction

from grutopia.core.robot.controller import BaseController
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.robot.robot_model import ControllerModel
from grutopia.core.util import log


@BaseController.register('OracleMoveAlongPathPointsController')
class OracleMoveAlongPathPointsController(BaseController):
    """Controller for moving alone a set of path points by utilizing a oracle move-to-point controller as sub-controller."""

    def __init__(self, config: ControllerModel, robot: BaseRobot, scene: Scene) -> None:
        self._user_config = None
        self.path_points: List[np.ndarray | List] = []
        self.path_point_idx = 0
        self.current_path_point: np.ndarray | None = None

        self.forward_speed = config.forward_speed if config.forward_speed is not None else 1.0
        self.rotation_speed = config.rotation_speed if config.rotation_speed is not None else 8.0
        self.threshold = config.threshold if config.threshold is not None else 0.02

        super().__init__(config=config, robot=robot, scene=scene)

    def forward(self,
                start_position: np.ndarray,
                start_orientation: np.ndarray,
                path_points: np.ndarray,
                threshold: float = 0.02,
                current_step: int = 10,
                step_interval: int = 30) -> ArticulationAction:

        if self.path_points is not path_points:
            self.path_points = path_points
            self.path_point_idx = 0
            log.info('reset path points')
            self.current_path_point = np.array(deepcopy(self.path_points[self.path_point_idx]))
            # self.current_path_point[-1] = 0

        if current_step % step_interval == 0:
            # Just make sure we ignore z components
            # start_position[-1] = 0 # The robot's actual current position

            dist_from_goal = np.linalg.norm(start_position[:2] - self.current_path_point[:2])
            if dist_from_goal < threshold:
                if self.path_point_idx < len(self.path_points) - 1:
                    self.path_point_idx += 1
                    self.current_path_point = np.array(deepcopy(self.path_points[self.path_point_idx]))
                    # self.current_path_point[-1] = 0
                    log.info(f'switch to next path point: {self.current_path_point}')

            return self.sub_controllers[0].forward(
                start_position=start_position,
                start_orientation=start_orientation,
                goal_position=self.current_path_point,
                threshold=threshold
            )
        else:
            self.sub_controllers[0].forward(
                start_position=start_position,
                start_orientation=start_orientation,
                goal_position=start_position,
                threshold=threshold
            )
            self.sub_controllers[0].finished = False
            return ArticulationAction()

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """Convert input action (in 1d array format) to joint signals to apply.

        Args:
            action (List | np.ndarray): 1-element 1d array containing
              0. path points list (List[np.ndarray])

        Returns:
            ArticulationAction: joint signals to apply.
        """
        # assert len(action) == 1, 'action must contain 1 elements'
        assert len(action[0]) > 0, 'path points cannot be empty'
        start_position, start_orientation = self.robot.get_world_pose()

        if len(action)>1 and 'current_step' in action[1]:
            current_step = action[1]['current_step']
        else:
            current_step = 10

        return self.forward(start_position=start_position,
                            start_orientation=start_orientation,
                            path_points=action[0],
                            threshold=self.threshold*self.robot.get_robot_scale()[0],
                            current_step=current_step,
                            )

    def get_obs(self) -> Dict[str, Any]:
        finished = False
        total_points = len(self.path_points)
        if total_points > 0 and self.path_point_idx == total_points - 1:
            finished = self.sub_controllers[0].get_obs().get('finished', False)

        return {
            'current_index': self.path_point_idx,
            'current_point': self.current_path_point,
            'total_points': total_points,
            'finished': finished,
        }
