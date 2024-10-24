from typing import Dict
from PIL import Image
import numpy as np

from omni.isaac.sensor import Camera as i_Camera
import omni.replicator.core as rep

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
        super().__init__(config, robot, name)
        self._camera = self.create_camera()

    def create_camera(self) -> i_Camera:
        """Create an isaac-sim camera object.

        Initializes the camera's resolution and prim path based on configuration.

        Returns:
            i_Camera: The initialized camera object.
        """
        # Initialize the default resolution for the camera
        size = (320, 240)
        # Use the configured camera size if provided.
        if self.config.size is not None:
            size = self.config.size

        prim_path = self._robot.user_config.prim_path + '/' + self.config.prim_path
        self.prim_path = prim_path
        
        camera = i_Camera(prim_path=prim_path, resolution=size)
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        
        # obtain data from rep
        self._render_product_paths: list[str] = list()
        self.rp = rep.create.render_product(self.prim_path, size)
        if not isinstance(self.rp, str):
            render_prod_path = self.rp.path
        self._render_product_paths.append(render_prod_path)
        
        data_type = [
            "bbox",
            "rgba",
            "depth",
            "pointcloud",
            "normals",
            "camera_params"
        ]
        self._rep_registry: dict[str, list[rep.annotators.Annotator]] = {name: list() for name in data_type}
        
        # bbox
        self.bbox_receiver = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        self.bbox_receiver.attach(self.rp)
        self._rep_registry["bbox"].append(self.bbox_receiver)
        
        # rgba
        self.rgba_receiver = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self.rgba_receiver.attach(self.rp)
        self._rep_registry["rgba"].append(self.rgba_receiver)
        
        # depth
        self.depth_reveiver = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth_reveiver.attach(self.rp)
        self._rep_registry["depth"].append(self.depth_reveiver)
        
        # pointcloud
        self.pointcloud_receiver = rep.AnnotatorRegistry.get_annotator("pointcloud", init_params={"includeUnlabelled": True})
        self.pointcloud_receiver.attach(self.rp)
        self._rep_registry["pointcloud"].append(self.pointcloud_receiver)
        
        # camera_params
        self.camera_params_receiver = rep.AnnotatorRegistry.get_annotator("CameraParams")
        self.camera_params_receiver.attach(self.rp)
        self._rep_registry["camera_params"].append(self.camera_params_receiver)
        
        # normals
        self.normals_receiver = rep.AnnotatorRegistry.get_annotator("normals")
        self.normals_receiver.attach(self.rp)
        self._rep_registry["normals"].append(self.normals_receiver)

        # Setting capture on play to False will prevent the replicator from capturing data each frame
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False) # !!!

        # writer
        # if init_writer in self.config and self.config.init_writer:
        #     self.writer = rep.WriterRegistry.get("BasicWriter")
        #     self.writer.initialize(
        #         output_dir=f"{self.config.}/writer", rgb=True, depth=True
        #     )
        #     self.writer.attach(self.rp)
        
        log.debug('camera_prim_path: ' + prim_path)
        log.debug('name            : ' + self.config.name)
        log.debug(f'size            : {size}')
        return camera

    def sensor_init(self) -> None:
        if self.config.enable:
            self._camera.initialize()
            self._camera.add_distance_to_image_plane_to_frame()
            self._camera.add_semantic_segmentation_to_frame()
            self._camera.add_instance_segmentation_to_frame()
            self._camera.add_instance_id_segmentation_to_frame()
            self._camera.add_bounding_box_2d_tight_to_frame()
    
    def camera_get_rgba(self, add_subframes=True):
        if add_subframes:
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        return self.rgba_receiver.get_data()

    def get_data(self, data_type:list=None) -> Dict:
        if data_type is not None:
            return self.get_camera_data(data_type)
        else:
            if self.config.enable:
                rgba = self._camera.get_rgba()
                depth = self._camera.get_depth()
                # frame = self._camera.get_current_frame()
                camera_params = self._camera.get_camera_params()
                return {'rgba': rgba, 'depth': depth, 'camera_params': camera_params}
            return {}

    
    def get_camera_data(self, data_type: list) -> Dict:
        output_data = {}
        if "bbox" in data_type:
            output_data["bbox"] = self.bbox_receiver.get_data()
        if "rgba" in data_type:
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False) # !!!
            output_data["rgba"] = self.rgba_receiver.get_data()
        if "depth" in data_type:
            output_data["depth"] = self.depth_reveiver.get_data()
        if "pointcloud" in data_type: 
            output_data["pointcloud"] = self.pointcloud_receiver.get_data()
        if "normals" in data_type:
            output_data["normals"] = self.normals_receiver.get_data()
        if "camera_params" in data_type:
            output_data["camera_params"] = self.camera_params_receiver.get_data()
        return output_data
    
    def get_world_pose(self):
        return self._camera.get_world_pose()
    
    def write_rgb_data(self, rgb_data, file_path):
        rgb_img = Image.fromarray(rgb_data[:,:,:3], "RGB")
        rgb_img.save(file_path + ".png")
        
    def write_depth_data(self, depth_data, file_path, as_npy=False):
        depth_img = Image.fromarray(depth_data, "L")
        if as_npy:
            np.save(file_path + ".npy", depth_data)
        else:
            depth_img.save(file_path + ".png")