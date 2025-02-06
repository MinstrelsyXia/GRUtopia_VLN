from typing import Dict

import numpy as np
import omni.replicator.core as rep
from omni.isaac.sensor import Camera as i_Camera
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import XFormPrim
from pxr import Usd, UsdGeom

from grutopia.core.robot.robot import BaseRobot, Scene
from grutopia.core.robot.robot_model import SensorModel
from grutopia.core.robot.sensor import BaseSensor
from grutopia.core.util import log

import torch
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
    
    

from thirdparty.landmark_isaacsim_interaction.scgs_renderer import MultiModelSCGSRenderer
def get_xform_list_pose(model_xform_list):
    translations = []
    rotations = []
    scales = []
    for model_xform in model_xform_list:
        model_pose = model_xform.get_world_pose()
        model_scale = model_xform.get_world_scale()
        translations.append(model_pose[0])
        rotations.append(model_pose[1])
        scales.append(model_scale)
    return translations, rotations, scales

@BaseSensor.register('Camera_3dgs')
class Camera_3dgs(BaseSensor):
    """Camera controller for 3DGS rendering."""

    def __init__(self, config: SensorModel, robot: BaseRobot,name:str = None, scene: Scene = None) -> None:
        """Initialize Camera_3dgs controller.
        
        Args:
            config: Camera configuration
            robot: Robot instance
            scene: Scene instance
        """
        super().__init__(config=config, robot=robot, scene=scene)
        self.name = name
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


    def set_renderer(self, lego_xform_list ,lego_gs_root,lego_name_list,lego_device_number,lego_editable):
        self.lego_xform_list = lego_xform_list
                # Initialize 3DGS renderer
        self.scgs_renderer = MultiModelSCGSRenderer(
            model_root=lego_gs_root,
            model_name_list=lego_name_list,
            device_number=lego_device_number,
            editable=lego_editable
        )
        focal_length = self._camera.get_focal_length()*1000
        fovx = self._camera.get_horizontal_fov()
        fovy = self._camera.get_vertical_fov()  
        self.scgs_renderer.set_camera_params(self.size[0], self.size[1], focal_length, focal_length, 0.01, 100.0,fovx,fovy)
        

    def get_pc(self,depth,cam_transform_matrix):
        from vlmaps.application_my.utils import get_dummy_2d_grid, downsample_pc, visualize_pc
        grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
        pc = self._camera.get_world_points_from_image_coords(grid_2d, depth.flatten())
        pc = downsample_pc(pc, 150)
        return pc


    
    def get_data(self, add_rgb_subframes=False, render=False, gs3d=False):
        """获取相机数据，包括RGB、深度图和点云"""
        # if self has no attribute lego_xform_list, then return empty data
        if not hasattr(self, 'lego_xform_list'):
            return {}   
        if self.config.enable:
            rgb = {}
            depth = {}
            pc= {}
            if gs3d == True:
                # torch.cuda.empty_cache()
                cam_transform_matrix = get_relative_transform(get_prim_at_path(self._camera.prim_path), get_prim_at_path("/World"))
                # with torch.no_grad():   
                self.scgs_renderer.update_editing_package(*get_xform_list_pose(self.lego_xform_list))
                scgs_rendering = self.scgs_renderer.minicam_render(cam_transform_matrix)
                rgb = scgs_rendering['render'].detach().cpu().numpy()
                depth = scgs_rendering['depth'].detach().cpu().numpy()[0]    # shape：（H，W）
                # pc = self.get_pc(depth,cam_transform_matrix)
                pc = self._camera.get_pointcloud()
                #! or: pc = self._camera.get_pointcloud()
            
            return {
                'rgb': rgb,
                'depth': depth,
                'pointcloud': pc,
                # 'pointcloud': self.depth_to_pointcloud(render_results['depth'])  # 如果需要点云
            }
        return {}
    
    def sensor_init(self) -> None:
        """
        Initialize the camera sensor.
        """
        if self.config.enable:
            self._camera.initialize()
            self._camera.add_distance_to_image_plane_to_frame()

    def reset(self):
        del self._camera
        self._camera = self.create_camera()
        self.sensor_init()
######### setup sim ###########



