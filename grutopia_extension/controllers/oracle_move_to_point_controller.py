from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.rotations import quat_to_euler_angles
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

        super().__init__(config=config, robot=robot, scene=scene)

    @staticmethod
    def vector_to_quaternion(vec):
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, vec)
        angle = np.arccos(np.dot(z_axis, vec))
        
        if np.linalg.norm(axis) < 1e-6:  
            return np.array([0, 0, 0, 1])  

        axis /= np.linalg.norm(axis)  
        s = np.sin(angle / 2)
        q = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2)])
        return q

    @staticmethod
    def quaternion_from_axis_angle(axis, angle):
        """Create a quaternion from an axis-angle representation."""
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        s = np.sin(angle / 2)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2)])

    @staticmethod
    def quaternion_multiply(q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

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

        goal_orientation = OracleMoveToPoint.vector_to_quaternion(target_vec)

        return signed_angle, goal_orientation

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

    def forward(self,
                start_position: np.ndarray,
                start_orientation: np.ndarray,
                goal_position: np.ndarray,
                threshold: float) -> ArticulationAction:
        # Just make sure we ignore z components
        start_z = start_position[-1]
        goal_z = goal_position[-1]
        start_position[-1] = 0
        goal_position[-1] = 0

        self.goal_position = goal_position
        self.last_threshold = threshold
        angle_threshold = np.pi/4

        dist_from_goal = np.linalg.norm(start_position - goal_position)
        if dist_from_goal < threshold:
            # the same point
            self.robot.isaac_robot.set_world_pose(start_position, start_orientation)
            return ArticulationAction()
        
        angle_to_goal, goal_orientation = OracleMoveToPoint.get_angle_and_orientation(start_position, start_orientation, goal_position)

        # Limit the robot to only rotate a maximum of pi/4
        # if abs(angle_to_goal) > angle_threshold:
        #     # Calculate the limited angle
        #     limited_angle = angle_threshold * np.sign(angle_to_goal)
            
        #     # Create a rotation quaternion for the limited angle
        #     rotation_axis = np.array([0, 0, 1])  # Assuming rotation around the Z-axis
        #     rotation_quaternion = OracleMoveToPoint.quaternion_from_axis_angle(rotation_axis, limited_angle)  
            
        #     # Combine the current orientation with the limited rotation
        #     new_orientation = OracleMoveToPoint.quaternion_multiply(start_orientation, rotation_quaternion) 
            
        #     # Set the robot's pose with the current position and new limited orientation
        #     self.robot.isaac_robot.set_world_pose(start_position, new_orientation)
        # else:
        #     # If the angle is within pi/4, directly set the robot to goal position and orientation
        #     self.robot.isaac_robot.set_world_pose(goal_position, goal_orientation)

        goal_position = [goal_position[0], goal_position[1], goal_z]
        self.robot.isaac_robot.set_world_pose(goal_position, goal_orientation) # TODO

        # We have reached the goal position.
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
        return {
            'finished': True,
        }
