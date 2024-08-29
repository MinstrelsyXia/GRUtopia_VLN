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
    
    def set_robot_poses_without_offset(self, position_list, orientation_list):
        for idx, robot in enumerate(self.robots):
            position = position_list[idx] + self.config.offset
            robot.isaac_robot.set_world_pose(position, orientation_list[idx])
    
    def get_robot_poses_without_offset(self):
        positions = [robot.isaac_robot.get_world_pose()[0]-self.config.offset for robot in self.robots]
        orientations = [robot.isaac_robot.get_world_pose()[1] for robot in self.robots]
        return positions, orientations
    
    def set_single_robot_poses_without_offset(self, position, orientation, robot_idx=0):
        position = position + self.config.offset
        self.robots[robot_idx].isaac_robot.set_world_pose(position, orientation)
