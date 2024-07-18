from typing import Dict

from omni.isaac.sensor import Camera as i_Camera
import omni.replicator.core as rep

from grutopia.core.robot.robot import BaseRobot, Scene
from grutopia.core.robot.robot_model import SensorModel
from grutopia.core.robot.sensor import BaseSensor
from grutopia.core.util import log


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
        self.rp = rep.create.render_product(self.prim_path, size)
        
        # bbox
        self.bbox_receiver = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        self.bbox_receiver.attach(self.rp)
        
        # rgba
        self.rgba_receiver = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self.rgba_receiver.attach(self.rp)
        
        # depth
        self.depth_reveiver = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth_reveiver.attach(self.rp)
        
        # pointcloud
        self.pointcloud_receiver = rep.AnnotatorRegistry.get_annotator("pointcloud", init_params={"includeUnlabelled": True})
        self.pointcloud_receiver.attach(self.rp)
        
        # camera_params
        self.camera_params_receiver = rep.AnnotatorRegistry.get_annotator("CameraParams")
        self.camera_params_receiver.attach(self.rp)
        
        # normals
        self.normals_receiver = rep.AnnotatorRegistry.get_annotator("normals")
        self.normals_receiver.attach(self.rp)
        
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

    def get_data(self, data_type:list=None) -> Dict:
        if data_type is not None:
            return self.get_camera_data(data_type)
        else:
            if self.config.enable:
                rgba = self._camera.get_rgba()
                depth = self._camera.get_depth()
                frame = self._camera.get_current_frame()
                return {'rgba': rgba, 'depth': depth, 'frame': frame}
            return {}
    
    def get_camera_data(self, data_type: list) -> Dict:
        output_data = {}
        if "bbox" in data_type:
            output_data["bbox"] = self.bbox_receiver.get_data()
        if "rgba" in data_type:
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
        
