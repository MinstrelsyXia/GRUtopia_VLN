import numpy as np
import open3d as o3d
from pxr import UsdGeom
import omni.replicator.core as rep
from collections import defaultdict

######################## compute bbox for every objects in current view ########################
# could use camera = rep.create.camera(position, rotation, ...) to create camera yourself
def get_camera_data(camera, resolution, data_names):
    """
    Get specified data from a camera.

    Parameters:
        camera: str or rep.Camera, the prim_path of the camera or a camera object created by rep.create.camera
        resolution: tuple, the resolution of the camera, e.g., (1920, 1080)
        data_names: list, a list of desired data names, can be any combination of "bbox", "rgba", "depth", "pointcloud", "camera_params"

    Returns:
        output_data: dict, a dict of data corresponding to the requested data names
    """
    
    output_data = {}

    # Create a render product for the specified camera and resolution
    rp = rep.create.render_product(camera, resolution)

    if "bbox" in data_names:
        bbox_2d_tight = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        bbox_2d_tight.attach(rp)
        output_data['bbox'] = bbox_2d_tight.get_data()

    if "rgba" in data_names:
        ldr_color = rep.AnnotatorRegistry.get_annotator("LdrColor")
        ldr_color.attach(rp)
        output_data['rgba'] = ldr_color.get_data()

    if "depth" in data_names:
        distance_to_image_plane = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        distance_to_image_plane.attach(rp)
        output_data['depth'] = distance_to_image_plane.get_data()
    
    if "pointcloud" in data_names:
        pointcloud = rep.AnnotatorRegistry.get_annotator("pointcloud")
        pointcloud.attach(rp)
        output_data['pointcloud'] = pointcloud.get_data()

    if "camera_params" in data_names:
        camera_params = rep.annotators.get("CameraParams").attach(rp)
        output_data['camera_params'] = camera_params.get_data()

    return output_data
