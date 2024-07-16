import numpy as np
import open3d as o3d
from pxr import UsdGeom
import omni.replicator.core as rep
from collections import defaultdict

def find_prims_with_suffix(stage, suffix):
    found_prims = []
    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        if prim_path.endswith(suffix):
            found_prims.append(prim)
    return found_prims

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

######################## compute bbox given the prim of an object ########################
def get_face_to_instance_by_2d_bbox(bbox: np.array, idToLabels, resolution):
    bbox = merge_tuples(bbox)
    label_to_bbox_area = []
    for row_idx in range(len(bbox)):
        id = str(bbox[row_idx][0])
        semantic_label = idToLabels[id]['class']
        bbox_area = bbox[row_idx][1]
        occlusion = bbox[row_idx][2]
        # if bbox_area >= 0.02 * resolution[0] * resolution[1] or occlusion > 0.7:
        label_to_bbox_area.append((semantic_label, bbox_area))
    if not label_to_bbox_area:
        return []
    # face_to_instance_id = sorted(
    #     label_to_bbox_area,
    #     key=lambda x: x[1],
    #     reverse=True
    # )[0][0]
    return [object_in_view[0] for object_in_view in label_to_bbox_area]

def to_list(data):
    res = []
    if data is not None:
        res = [_ for _ in data]
    return res

def merge_tuples(data):
    """
    Merge tuples with the same semanticId and compute weighted average for occlusionRatio
    based on the area of the bounding boxes.

    Parameters:
    data (list of tuples): Each tuple contains (semanticId, x_min, y_min, x_max, y_max, occlusionRatio)

    Returns:
    list of tuples: Merged tuples with (semanticId, total_area, weighted_average_occlusion_ratio)
    """
    # Dictionary to store the merged data
    merged_data = defaultdict(lambda: [0.0, 0.0])  # Initialize with total area and weighted occlusion sum

    # Traverse the original data and merge
    for entry in data:
        semantic_id, x_min, y_min, x_max, y_max, occlusion_ratio = entry
        area = (x_max - x_min) * (y_max - y_min)
        merged_data[semantic_id][0] += area                                    # Accumulate area
        merged_data[semantic_id][1] += occlusion_ratio * area                  # Accumulate weighted occlusion_ratio

    # Construct the merged list
    result = []
    for semantic_id, values in merged_data.items():
        total_area, weighted_occlusion_sum = values
        weighted_average_occlusion_ratio = weighted_occlusion_sum / total_area if total_area != 0 else 0
        result.append((semantic_id, total_area, weighted_average_occlusion_ratio))

    return result

def recursive_parse(prim):
    # print(prim.GetPath())
    translation = prim.GetAttribute("xformOp:translate").Get()
    if translation is None:
        translation = np.zeros(3)
    else:
        translation = np.array(translation)
    
    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is None:
        scale = np.ones(3)
    else:
        scale = np.array(scale)

    orient = prim.GetAttribute("xformOp:orient").Get()
    if orient is None:
        orient = np.zeros([4,1])
        orient[0] = 1.0
    else:
        # print(orient)
        r = orient.GetReal()
        i,j,k = orient.GetImaginary()

        orient = np.array([r,i,j,k]).reshape(4,1)

    transform = prim.GetAttribute("xformOp:transform").Get()
    if transform is None:
        transform = np.eye(4)
    else:
        transform = np.array(transform)

    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(orient)

    points_total = []
    faceuv_total = []
    normals_total = [] 
    faceVertexCounts_total = []
    faceVertexIndices_total = []
    mesh_total = []
    if prim.IsA(UsdGeom.Mesh):
        mesh_path = str(prim.GetPath()).split("/")[-1]
        if not mesh_path == 'SM_Dummy':
            mesh_total.append(mesh_path)
            points = prim.GetAttribute("points").Get()
            normals = prim.GetAttribute("normals").Get()
            faceVertexCounts = prim.GetAttribute("faceVertexCounts").Get()
            faceVertexIndices = prim.GetAttribute("faceVertexIndices").Get()
            faceuv = prim.GetAttribute("primvars:st").Get()
            normals = to_list(normals)
            faceVertexCounts = to_list(faceVertexCounts)
            faceVertexIndices = to_list(faceVertexIndices)
            faceuv = to_list(faceuv)
            points = to_list(points)
            ps = []
            for p in points:
                x,y,z = p
                p = np.array((x,y,z))
                ps.append(p)

            points = ps

            base_num = len(points_total)
            for idx in faceVertexIndices:
                faceVertexIndices_total.append(base_num + idx)
            
            faceVertexCounts_total += faceVertexCounts
            faceuv_total += faceuv
            normals_total += normals
            points_total += points

    # else:
    
    children = prim.GetChildren()

    for child in children:
        points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_list = recursive_parse(child)
        # child_path = child.GetPath()
        # if len(normals) > len(points):
        #     print(f"points is less than their normals, the prim is {child_path}")
        #     print(len(points), len(normals), len(points_total), len(normals_total), len(faceVertexCounts))

        base_num = len(points_total)
        for idx in faceVertexIndices:
            faceVertexIndices_total.append(base_num + idx)
        
        faceVertexCounts_total += faceVertexCounts
        faceuv_total += faceuv
        normals_total += normals[:len(points)]
        points_total += points
        mesh_total += mesh_list
    
    new_points = []
    for i, p in enumerate(points_total):
        pn = np.array(p)
        pn *= scale
        pn = np.matmul(rotation_matrix, pn)
        pn += translation
        new_points.append(pn)

    if len(new_points) > 0:
        points_mat = np.ones((len(new_points),4)).astype(np.float32)
        points_mat[:,:3] = np.array(new_points)
        # print(points_mat.shape, transform.shape)
        points_mat = np.matmul(points_mat, transform)
        new_points = [_ for _ in points_mat[:,:3]]

    return new_points, faceuv_total, normals_total, faceVertexCounts_total, faceVertexIndices_total, mesh_total

def compute_aabb(prim):
    points_total, faceuv_total, normals_total, faceVertexCounts_total, faceVertexIndices_total, mesh_total = recursive_parse(prim)
    points_total = np.array(points_total)
    if len(points_total) == 0:
        return None, None
    min_p = points_total.min(0)
    max_p = points_total.max(0)
    return min_p, max_p