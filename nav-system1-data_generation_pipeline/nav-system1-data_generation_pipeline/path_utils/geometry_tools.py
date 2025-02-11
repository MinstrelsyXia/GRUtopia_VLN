import numpy as np
import torch
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
# Function that calculate pairwise pointcloud distance
def pointcloud_distance(pcdA,pcdB,device='cpu'):
    pointsA = torch.tensor(np.array(pcdA.points),device=device)
    pointsB = torch.tensor(np.array(pcdB.points),device=device)
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

# Function that calculate pairwise pointcloud distance, but ignore the z-dimension
def numpy_2d_distance(AA,BB,device='cpu'):
    pointsA = torch.tensor(np.array(AA),device=device)[:,0:2]
    pointsB = torch.tensor(np.array(BB),device=device)[:,0:2]    
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

# Function that calculate pairwise pointcloud distance, but ignore the z-dimension
def pointcloud_2d_distance(pcdA,pcdB,device='cpu'):
    pointsA = torch.tensor(np.array(pcdA.points),device=device)[:,0:2]
    pointsB = torch.tensor(np.array(pcdB.points),device=device)[:,0:2]
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

# Function that calculate pairwise pointcloud distance, but ignore the z-dimension
def points_2d_distance(pcdA,pcdB,device='cpu'):
    pointsA = torch.tensor(np.array(pcdA),device=device)[:,0:2]
    pointsB = torch.tensor(np.array(pcdB),device=device)[:,0:2]
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

def get_pointcloud_from_depth(rgb:np.ndarray,depth:np.ndarray,intrinsic:np.ndarray,extrinsic:np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>0)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_x,pixel_z,-pixel_y],axis=-1)
    point_values = np.matmul(extrinsic,np.concatenate((point_values,np.ones((point_values.shape[0],1))),axis=-1).T).T[:,0:3]
    return filter_z,filter_x,depth_values,point_values,color_values

def generate_intrinsic(width,height,hfov,vfov):
    intrinsic = np.eye(3)
    intrinsic[0][0] = width / (2 * (np.tan(np.deg2rad(hfov)/2)))
    intrinsic[1][1] = height / (2 * (np.tan(np.deg2rad(vfov)/2)))
    intrinsic[0][2] = width / 2
    intrinsic[1][2] = height / 2
    return intrinsic

def cpu_pointcloud_from_array(points,colors):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return pointcloud

def project_to_camera(points,intrinsic,extrinsic):
    inv_extrinsic = np.linalg.inv(extrinsic)
    camera_points = np.concatenate((points,np.ones((points.shape[0],1))),axis=-1)
    camera_points = np.matmul(inv_extrinsic,camera_points.T).T[:,0:3]
    depth_values = -camera_points[:,2]
    filter_x = (camera_points[:,0] * intrinsic[0][0] / depth_values + intrinsic[0][2]).astype(np.int32)
    filter_z = (-camera_points[:,1] * intrinsic[1][1] / depth_values - intrinsic[1][2] + intrinsic[1][2]*2 - 1).astype(np.int32)
    return filter_x,filter_z,depth_values

def project_inview_area(image,points,intrinsic,extrinsic):
    filter_x,filter_z,depth_values = project_to_camera(points,intrinsic,extrinsic)
    area = np.where((filter_x >= 0) & (filter_x < image.shape[1]) & (filter_z >= 0) & (filter_z < image.shape[0]) & (depth_values > 0))[0].shape[0]
    return area

def project_inview_trajectory(image,points,intrinsic,extrinsic):
    filter_x,filter_z,_ = project_to_camera(points * np.array([1,1,0]),intrinsic,extrinsic)
    traj_mask = np.zeros_like(image)
    for i in range(filter_x.shape[0]-1):
        x1,y1,x2,y2 = filter_x[i],filter_z[i],filter_x[i+1],filter_z[i+1]
        if x1 > 0 and x1 < image.shape[1] and y1 > 0 and y1 < image.shape[0] and x2 > 0 and x2 < image.shape[1] and y2 > 0 and y2 < image.shape[0]:
            cv2.line(traj_mask,(filter_x[i],filter_z[i]),(filter_x[i+1],filter_z[i+1]),color=(255,255,255),thickness=10)
    return traj_mask.mean(axis=-1)

def select_view(pcd,intrinsic,extrinsics,width=640,height=480):
    sum_list = []
    for extrinsic in extrinsics:
        xs,zs,ds = project_to_camera(pcd,intrinsic,extrinsic)
        condition = (xs > 0) & (xs < width) & (zs > 0) & (zs < height) & (ds > 0)
        inview_amount = condition.sum()
        sum_list.append(inview_amount)
    return extrinsics[np.argmax(sum_list)]

def clockwise_angle(v1,v2):
    dot_product = np.dot(v1,v2)
    determinat = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(determinat,dot_product)
    if angle < 0:
        angle += 2*np.pi
    return angle

def world2frame(world_R1,world_T1,world_R2,world_T2):
    homo_RT = np.eye(4)
    homo_RT[0:3,0:3] = world_R1
    homo_RT[0:3,3] = world_T1
    R_rel = np.dot(world_R2,world_R1.T)
    T_rel = np.dot(np.linalg.inv(homo_RT),np.array([*world_T2,1]).T)[0:3]
    T_rel[-1] = -T_rel[-1]
    return R_rel,T_rel

def frame2world(base_R1,base_T1,frame_R1,frame_T1):
    homo_RT = np.eye(4)
    homo_RT[0:3,0:3] = base_R1
    homo_RT[0:3,3] = base_T1
    frame_T1[-1] = -frame_T1[-1]
    world_R = np.dot(frame_R1,base_R1)
    world_T = np.dot(homo_RT,np.array([*frame_T1,1])).T[0:3]
    return world_R,world_T 

def clockwise_angle(v1,v2):
    dot_product = np.dot(v1,v2)
    determinat = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(determinat,dot_product)
    if angle < 0:
        angle += 2*np.pi
    return angle

def build_transformation_mat(translation,rotation):
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector.")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(R.from_euler('xyz',rotation).as_matrix())
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
                           f"(3,3) or Euler angles of shape (3,) or (3,1).")

    return mat

def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff

def angle_add(angle1, angle2):
    sum = angle1 + angle2
    while sum > np.pi:
        sum -= 2 * np.pi
    while sum < -np.pi:
        sum += 2 * np.pi
    return sum

def normalized_angle(angle):
    while angle > 2 * np.pi:
        angle -= 2 * np.pi
    while angle < 0:
        angle += 2 * np.pi
    return angle