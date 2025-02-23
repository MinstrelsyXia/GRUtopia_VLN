from typing import Dict

import numpy as np
import omni.replicator.core as rep
from omni.isaac.sensor import Camera as i_Camera
from pxr import Usd, UsdGeom

from grutopia.core.robot.robot import BaseRobot, Scene
from grutopia.core.robot.robot_model import SensorModel
from grutopia.core.robot.sensor import BaseSensor
from grutopia.core.util import log

import carb.settings

class FineCamera(i_Camera):

    def get_render_product(self):
        return self._render_product

    def get_view_matrix_ros(self):
        """3D points in World Frame -> 3D points in Camera Ros Frame

        Returns:
            np.ndarray: the view matrix that transforms 3d points in the world frame to 3d points in the camera axes
                        with ros camera convention.
        """
        R_U_TRANSFORM = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        width, height = self.get_resolution()
        rp = rep.create.render_product(self.prim_path, resolution=(width, height))
        _camera_params = rep.annotators.get('CameraParams')
        _camera_params.attach(rp)
        camera_params = _camera_params.get_data()
        try:
            world_w_cam_u_T = self._backend_utils.transpose_2d(
                self._backend_utils.convert(
                    np.linalg.inv(camera_params['cameraViewTransform'].reshape(4, 4)),
                    dtype='float32',
                    device=self._device,
                    indexed=True,
                ))
        except np.linalg.LinAlgError:
            world_w_cam_u_T = self._backend_utils.transpose_2d(
                self._backend_utils.convert(
                    UsdGeom.Imageable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                    dtype='float32',
                    device=self._device,
                    indexed=True,
                ))
        r_u_transform_converted = self._backend_utils.convert(R_U_TRANSFORM,
                                                              dtype='float32',
                                                              device=self._device,
                                                              indexed=True)
        return self._backend_utils.matmul(r_u_transform_converted, self._backend_utils.inverse(world_w_cam_u_T))


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

        camera = FineCamera(prim_path=prim_path, resolution=self.size)
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