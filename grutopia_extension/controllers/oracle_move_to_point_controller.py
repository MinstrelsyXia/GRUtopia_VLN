from typing import Any, Dict, List
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction

from grutopia.core.robot.controller import BaseController
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.robot.robot_model import ControllerModel


@BaseController.register('OracleMoveToPointController')
class OracleMoveToPoint(BaseController):
    """Controller for moving to a target point (Oracle)."""

    def __init__(self, config: ControllerModel, robot: BaseRobot, scene: Scene) -> None:
        self.last_threshold = None
        self._user_config = None
        self.goal_position: np.ndarray | None = None
        self.threshold: float | None = None

        self.forward_speed = config.forward_speed if config.forward_speed is not None else 1.0
        self.rotation_speed = config.rotation_speed if config.rotation_speed is not None else 8.0
        self.threshold = config.threshold if config.threshold is not None else 0.02

        self.finished = False
        self.point = None

        super().__init__(config=config, robot=robot, scene=scene)

    @staticmethod
    def get_angle_and_orientation(start_position, start_orientation, goal_position):
        normal_vec = np.array([0, 0, 1])
        robot_z_rot = quat_to_euler_angles(start_orientation)[-1]

        robot_vec = np.array([np.cos(robot_z_rot), np.sin(robot_z_rot), 0])
        robot_vec /= np.linalg.norm(robot_vec)

        target_vec = (goal_position - start_position)
        if np.linalg.norm(target_vec) == 0:
            return 0, start_orientation  
        
        target_vec /= np.linalg.norm(target_vec)

        dot_prod = np.dot(robot_vec, target_vec)
        cross_prod = np.cross(robot_vec, target_vec)

        if dot_prod > 1.0:
            dot_prod = 1.0
        angle = np.arccos(dot_prod)
        angle_sign = np.sign(np.dot(normal_vec, cross_prod))

        signed_angle = angle * angle_sign

        goal_orientation = OracleMoveToPoint.compute_goal_orientation(start_position, goal_position)

        return signed_angle, goal_orientation, robot_z_rot

    @staticmethod
    def get_angle(
        start_position,
        start_orientation,
        goal_position,
    ):
        normal_vec = np.array([0, 0, 1])
        robot_z_rot = quat_to_euler_angles(start_orientation)[-1]

        robot_vec = np.array([np.cos(robot_z_rot), np.sin(robot_z_rot), 0])
        robot_vec /= np.linalg.norm(robot_vec)

        target_vec = (goal_position - start_position)
        if np.linalg.norm(target_vec) == 0:
            return 0
        target_vec /= np.linalg.norm(target_vec)

        dot_prod = np.dot(robot_vec, target_vec)
        cross_prod = np.cross(robot_vec, target_vec)

        # Handle errors in floating-point arithmetic.
        if dot_prod > 1.0:
            dot_prod = 1.0
        angle = np.arccos(dot_prod)
        angle_sign = np.sign(np.dot(normal_vec, cross_prod))

        signed_angle = angle * angle_sign
        return signed_angle

    @staticmethod
    def compute_goal_orientation(start_position, goal_position):
        """
        Compute the goal orientation quaternion given the start orientation and goal position.
        
        Args:
        goal_position (tuple): Goal position as (x, y, z).
        start_position (tuple): Start position as (x, y, z).
        
        Returns:
        tuple: Goal orientation as a quaternion (w, x, y, z).
        """
        goal_orientation = euler_angles_to_quat(np.array([0, 0, np.arctan2(goal_position[1]-start_position[1], goal_position[0]-start_position[0])]))
        return goal_orientation

    def forward(self,
                start_position: np.ndarray,
                start_orientation: np.ndarray,
                goal_position: np.ndarray,
                threshold: float,
                topdown_camera_global=None,
                topdown_camera_local=None,
                is_hold=False) -> ArticulationAction:
        # Just make sure we ignore z components
        start_z = deepcopy(start_position[-1])
        goal_z = deepcopy(goal_position[-1])
        start_position[-1] = 0
        goal_position[-1] = 0

        self.goal_position = goal_position
        self.last_threshold = threshold
        angle_threshold = np.pi/4

        dist_from_goal = np.linalg.norm(start_position[:2] - goal_position[:2])
        if dist_from_goal < threshold:
            # the same point
            start_position = [start_position[0], start_position[1], start_z]
            self.robot.oracle_set_world_pose(start_position, start_orientation)
            if topdown_camera_local is not None:
                topdown_camera_local.set_world_pose(start_position)
            if topdown_camera_global is not None:
                topdown_camera_global.set_world_pose(start_position)
            self.point = None
            
            if is_hold:
                self.finished = False # 只是为了oracle过度interval
            else:
                self.finished = True

            return ArticulationAction()
        
        angle_to_goal, goal_orientation, robot_z_rot = OracleMoveToPoint.get_angle_and_orientation(start_position, start_orientation, goal_position)

        # Limit the robot to only rotate a maximum of pi/4
        if abs(angle_to_goal) > angle_threshold:
            # Calculate the limited angle
            limited_angle = angle_threshold * np.sign(angle_to_goal)
            next_angle = (robot_z_rot + limited_angle)%(2*np.pi)
            goal_orientation = euler_angles_to_quat(np.array([0, 0, next_angle]))
            goal_position = [start_position[0], start_position[1], start_z]
            self.finished = False
        else:
            goal_position = [goal_position[0], goal_position[1], goal_z]
        
        self.point = [goal_position, goal_orientation]
        self.robot.oracle_set_world_pose(goal_position, goal_orientation)

        if topdown_camera_local is not None:
            topdown_camera_local.set_world_pose(goal_position)
        if topdown_camera_global is not None:
            topdown_camera_global.set_world_pose(goal_position)

        return ArticulationAction()

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """Convert input action (in 1d array format) to joint signals to apply.

        Args:
            action (List | np.ndarray): n-element 1d array containing
              0. goal_position (np.ndarray)

        Returns:
            ArticulationAction: joint signals to apply.
        """
        assert len(action) == 1, 'action must contain 1 elements'
        start_position, start_orientation = self.robot.get_world_pose()
        return self.forward(start_position=start_position,
                            start_orientation=start_orientation,
                            goal_position=np.array(action[0]),
                            threshold=np.pi/4)

    def get_obs(self) -> Dict[str, Any]:
        if self.goal_position is None or self.last_threshold is None:
            return {}
        start_position = self.robot.isaac_robot.get_world_pose()[0]
        dist_from_goal = np.linalg.norm(start_position[:2] - self.goal_position[:2])
        finished = True if dist_from_goal < self.last_threshold else False
    
        res = {
            'point': self.point,
            'finished': finished,
        }
        return res
