from omni.isaac.core.scenes import Scene

from grutopia.core.config import TaskUserConfig
from grutopia.core.task import BaseTask


@BaseTask.register('VLNTask')
class VLNTask(BaseTask):

    def __init__(self, config: TaskUserConfig, scene: Scene):
        super().__init__(config, scene)

    def calculate_metrics(self) -> dict:
        pass

    def is_done(self) -> bool:
        return False

    def individual_reset(self):
        for name, metric in self.metrics.items():
            metric.reset()
    
    def set_robot_poses_without_offset(self, position, orientation):
        for idx, (robot_name, robot) in enumerate(self.robots.items()):
            position = position + self._offset 
            robot.isaac_robot.set_world_pose(position, orientation)
    
    def get_robot_poses_without_offset(self, robot_idx=0):
        for robot_name, robot in self.robots.items():
           positions = robot.isaac_robot.get_world_pose()[0]-self._offset 
           orientations = robot.isaac_robot.get_world_pose()[1]
        return positions, orientations
    
    def set_single_robot_poses_without_offset(self, position, orientation, robot_idx=0):
        position = position + self._offset 
        for robot_name, robot in self.robots.items():
            robot.isaac_robot.set_world_pose(position, orientation)
    
    def get_camera_poses_without_offset(self, camera, robot_idx=0):
        for robot_name, robot in self.robots.items():
            camera_positions = robot.sensors[camera].get_world_pose()[0]-self._offset 
            camera_orientation = robot.sensors[camera].get_world_pose()[1]
        return camera_positions, camera_orientation
