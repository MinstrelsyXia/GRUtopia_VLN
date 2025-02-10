from typing import Dict

import numpy as np
import omni.replicator.core as rep
from omni.isaac.sensor import Camera as i_Camera

from grutopia.core.robot.robot import BaseRobot, Scene
from grutopia.core.robot.robot_model import SensorModel
from grutopia.core.robot.sensor import BaseSensor
from grutopia.core.util import log

import carb.settings

@BaseSensor.register('Camera')
class Camera(BaseSensor):
    """
    wrap of isaac sim's Camera class
    """

    def __init__(self, config: SensorModel, robot: BaseRobot, name: str = None, scene: Scene = None):
        super().__init__(config, robot, scene)
        self.pointcloud_camera = None
        self.pointcloud_anno = None
        self.name = name
        self.size = (320, 240)
        self._camera = self.create_camera()

    def create_camera(self) -> i_Camera:
        """Create an isaac-sim camera object.

        Initializes the camera's resolution and prim path based on configuration.

        Returns:
            i_Camera: The initialized camera object.
        """
        # Initialize the default resolution for the camera
        # Use the configured camera size if provided.
        if self.config.size is not None:
            self.size = self.config.size

        prim_path = self._robot.user_config.prim_path + '/' + self.config.prim_path
        log.debug('camera_prim_path: ' + prim_path)
        log.debug('name            : ' + self.config.name)
        log.debug(f'size            : {self.size}')

        camera = i_Camera(prim_path=prim_path, resolution=self.size)
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False) # !!!

        return camera

    def sensor_init(self) -> None:
        """
        Initialize the camera sensor.
        """
        if self.config.enable:
            self._camera.initialize()
            self._camera.add_distance_to_image_plane_to_frame()

    def get_data(self, add_rgb_subframes=False) -> Dict:
        if self.config.enable:
            rgba = {}
            depth = {}
            pointcloud = {}
            if self.config.camera_config is None or 'no_rgb' not in self.config.camera_config:
                if add_rgb_subframes:
                    rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
                rgba = self._camera.get_rgba()
            if add_rgb_subframes:
                rep.orchestrator.step(rt_subframes=0, delta_time=0.0, pause_timeline=False)
            depth = self._camera.get_depth()
            if self.config.camera_config and 'point_cloud' in self.config.camera_config:
                pointcloud = self._camera.get_pointcloud()
            return {'rgba': rgba, 'pointcloud': pointcloud, 'depth': depth}
        return {}
    
    def get_camera_data(self, data_type: list) -> Dict:
        output_data = {}
        # if "bbox" in data_type:
        #     output_data["bbox"] = self._camera.get_bbox()
        if "rgba" in data_type:
            rep.orchestrator.step(rt_subframes=10, delta_time=0.0, pause_timeline=False) # !!!
            output_data["rgba"] = self._camera.get_rgba()
            rep.orchestrator.step(rt_subframes=0, delta_time=0.0, pause_timeline=False)
        if "depth" in data_type:
            output_data["depth"] = self._camera.get_depth()
        if "pointcloud" in data_type: 
            output_data["pointcloud"] = self._camera.get_pointcloud()
        # if "normals" in data_type: 
        #     output_data["normals"] = self._camera.get_normals()
        # if "camera_params" in data_type:
        #     output_data["camera_params"] = self._camera.get_camera_params()
        return output_data

    def reset(self):
        del self._camera
        self._camera = self.create_camera()
        self.sensor_init()
    
    def get_world_pose(self):
        return self._camera.get_world_pose()