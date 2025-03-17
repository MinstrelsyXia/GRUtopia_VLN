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


    def set_renderer(self, lego_xform_list ,lego_gs_root,lego_name_list,lego_device_number,lego_editable):
        self.lego_xform_list = lego_xform_list
                # Initialize 3DGS renderer
        if lego_device_number <0:
            self.device_number = torch.cuda.device_count() - 1
        else:
            self.device_number = lego_device_number
        self.scgs_renderer = MultiModelSCGSRenderer(
            model_root=lego_gs_root,
            model_name_list=lego_name_list,
            device_number=self.device_number,
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

    def get_camera_params(self):
        # width, height = self._camera.get_resolution()
        # fx = self._camera.get_focal_length()
        # fy = self._camera.get_focal_length()
        # cx, cy = self._camera.get_horizontal_aperture(), self._camera.get_vertical_aperture()
        # camera_intrinsic = np.array([
        #     [554.25616,   0.     , 320.     ],
        #     [  0.     , 554.25616, 240.     ],
        #     [  0.     ,   0.     ,   1.     ]
        # ])
        camera_intrinsic = self._camera.get_camera_intrinsics()
        #! self._camera.get_camera_intrinsics()
        # 获取相机外参（位姿）
        cam_transform_matrix = get_relative_transform(
            get_prim_at_path(self._camera.prim_path), 
            get_prim_at_path("/World")
        )
        camera_pos = cam_transform_matrix[:3, 3]
        camera_rot = cam_transform_matrix[:3, :3]
        
        return camera_intrinsic, camera_pos, camera_rot

    def create_pointcloud_from_rgbd(self, depth, rgb=None):
        """从RGBD图像创建点云"""
        # 获取相机参数
        camera_intrinsic, camera_pos, camera_rot = self.get_camera_params()
        
        # 生成像素网格
        height, width = depth.shape
        v, u = np.meshgrid(range(height), range(width), indexing='ij')
        v = v.reshape(-1)
        u = u.reshape(-1)
        z = depth.reshape(-1)
        
        # 过滤无效深度值
        # valid_mask = z > 0
        # u = u[valid_mask]
        # v = v[valid_mask]
        # z = z[valid_mask]
        
        # 反投影到相机坐标系
        x = (u - camera_intrinsic[0,2]) * z / camera_intrinsic[0,0]
        y = (v - camera_intrinsic[1,2]) * z / camera_intrinsic[1,1]
        
        # 组织相机坐标系下的点云
        points_cam = np.stack([x, y, z], axis=1)
        
        # 转换到世界坐标系
        points_world = (camera_rot @ points_cam.T).T + camera_pos
        
        # # 如果有RGB信息，添加颜色
        # if rgb is not None:
        #     colors = rgb.reshape(-1, 3)[valid_mask]
        #     return points_world, colors
        
        return points_world
    
    def get_data(self, add_rgb_subframes=False, render=False):
        """获取相机数据，包括RGB、深度图和点云"""
        # if self has no attribute lego_xform_list, then return empty data
        if not hasattr(self, 'lego_xform_list'):
            return {}   
        if self.config.enable:
            rgb = {}
            depth = {}
            pc= {}
            torch.cuda.empty_cache()
            if add_rgb_subframes == True:
                cam_transform_matrix = get_relative_transform(get_prim_at_path(self._camera.prim_path), get_prim_at_path("/World"))
                with torch.no_grad():   
                    self.scgs_renderer.update_editing_package(*get_xform_list_pose(self.lego_xform_list))
                    scgs_rendering = self.scgs_renderer.minicam_render(cam_transform_matrix)
                torch.cuda.empty_cache()
                rgb = scgs_rendering['render'].detach().cpu().numpy()
                rgb = np.transpose(rgb, (1, 2, 0))  # shape [H,W,3]
                rgb = np.clip(rgb, 0, 1)
                rgb = (rgb * 255).astype(np.uint8)
                # rgb = rgb[:, :, ::-1]  # BGR to RGB
                depth = scgs_rendering['depth'].detach().cpu().numpy()[0]    # shape：（H，W）
                depth = depth * 100
                # pc = self.get_pc(depth,cam_transform_matrix)
            pc = self._camera.get_pointcloud()
                # pc[:,2] = -pc[:,2]
                # 利用cam_transform_matrix，将pc转换到世界坐标系
                # pc = self.create_pointcloud_from_rgbd(depth, cam_transform_matrix)
            
            return {
                'rgba': rgb,
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
