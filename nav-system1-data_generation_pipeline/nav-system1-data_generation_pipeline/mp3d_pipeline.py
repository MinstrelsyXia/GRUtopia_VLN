import blenderproc as bproc
import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import random
import trimesh
import open3d as o3d
from tqdm import tqdm
import debugpy
import torch
import json
import bpy
import matplotlib.pyplot as plt
import imageio
import shutil
from path_utils.path_planner import PathPlanner
from path_utils.geometry_tools import *
from tqdm import tqdm

# debugpy.listen(5678)
# debugpy.wait_for_client()
# define input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scene_index",type=int,default=0)
parser.add_argument("--gpu_id",type=int,default=2)
parser.add_argument("--matterport3d", help="Path to the 3D front file", default='/g0433_data/yangyuqiang/Matterport3D/data/')
parser.add_argument("--output_dir", help="Path to where the data should be saved",default="/ssd/yangyuqiang/agile_navigation_2d/matterport3d_trajectory/")
parser.add_argument("--image_height",type=int,default=180)
parser.add_argument("--image_width",type=int,default=320)
parser.add_argument("--camera_hfov",type=float,default=68)
parser.add_argument("--camera_vfov",type=float,default=42)
parser.add_argument("--ceiling_height",type=float,default=1.8)
parser.add_argument("--safe_distance",type=float,default=0.1)
args = parser.parse_known_args()[0]
if not os.path.exists(args.matterport3d):
    raise Exception("One of the two folders does not exist!")
# blenderproc initialization
bproc.init()
bproc.renderer.set_render_devices(False,"CUDA",[args.gpu_id])
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
house_id = os.listdir(os.path.join(args.matterport3d,'v1/scans'))[args.scene_index]

# only for debug
try:
    shutil.rmtree("%s/%s"%(args.output_dir,house_id))
except:
    pass
if True:
    bproc.renderer.set_max_amount_of_samples(1028)
    loaded_objects,loaded_floors = bproc.loader.load_matterport3d(
        args.matterport3d,house_id
    )
    scene_pcd = o3d.geometry.PointCloud()
    floor_heights = []
    flag = False
    for index,obj in enumerate([loaded_objects,loaded_floors]):
        if not isinstance(obj,bproc.types.MeshObject):
            continue
        obj_mesh = obj.mesh_as_trimesh().apply_transform(obj.get_local2world_mat())
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
        new_mesh.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(obj_mesh.vertex_normals)
        new_pcd = new_mesh.sample_points_uniformly(200000)
        # new_pcd = new_pcd.voxel_down_sample(0.05)
        new_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(np.array(new_pcd.points)) * 0.4)
        scene_pcd = scene_pcd + new_pcd
        # scene_pcd = scene_pcd.voxel_down_sample(0.05)
        
        if index == 1:
            counts, bin_edges = np.histogram(np.array(new_pcd.points)[:,2], bins=5)
            max_counts = np.max(counts)
            peak_bins = np.where(counts == max_counts)[0]
            floor_heights = [np.mean(bin_edges[peak_bins])]

     # initialize the path_planner
    path_planner = PathPlanner(ceiling_offset=1.5,safe_distance=0.25)
    scene_points = np.array(scene_pcd.points)
    
    for floor_index,current_floor in enumerate(tqdm(floor_heights)):
        path_planner.reset(current_floor,0.5,scene_pcd)
        for trajectory_num in range(100):
            trajectory_index = floor_index * 100 + trajectory_num
            
            # prepare writer
            os.makedirs("%s/%s/trajectory_%d/"%(args.output_dir,house_id,trajectory_index),exist_ok=True)
            os.makedirs("%s/%s/trajectory_%d/rgb/"%(args.output_dir,house_id,trajectory_index),exist_ok=True)
            os.makedirs("%s/%s/trajectory_%d/depth/"%(args.output_dir,house_id,trajectory_index),exist_ok=True)

            bproc.renderer.set_light_bounces(diffuse_bounces=np.random.randint(50,300), glossy_bounces=np.random.randint(50,300), max_bounces=np.random.randint(50,300),transmission_bounces=np.random.randint(50,300),transparent_max_bounces=np.random.randint(50,300))
            camera_height = np.random.uniform(0.25,1.25)
            camera_intrinsic = generate_intrinsic(args.image_width,args.image_height,args.camera_hfov,args.camera_vfov)
            bpy.context.scene.frame_end = 0
            random_index = np.random.choice(np.nonzero(path_planner.safe_value > 0.1)[0])
            camera_translation = np.array(path_planner.navigable_pcd.points)[random_index]
            camera_translation[2] = camera_height + current_floor
            pitch_rad = np.deg2rad((210 - 180*camera_height))
            camera_rotation = [np.clip(pitch_rad,np.pi/3,np.pi/2),0,0]
            
            tobase_extrinsic = bproc.math.build_transformation_mat(camera_translation * np.array([0,0,1]),camera_rotation)
            distance = np.sum(np.abs(np.array(path_planner.navigable_pcd.points) - camera_translation),axis=-1)
            condition = np.where((path_planner.safe_value>0.2) & (distance > 5.0))[0]
            if condition.shape[0] == 0:
                continue
            target_point = np.array(path_planner.navigable_pcd.points)[np.random.choice(condition)]
            # 随机采集一个start angle
            start_angle = random.uniform(-np.pi, np.pi)
            status,waypoints,wayrotations = path_planner.generate_trajectory(camera_translation,target_point, start_angle)
            if status == False:
                continue

            result_path_map = path_planner.visualize_trajectory_on_map(waypoints)

            waypoints = np.concatenate((waypoints,np.ones((waypoints.shape[0],1)) * camera_translation[2]),axis=-1)
            wayrotations = np.stack((np.array([camera_rotation[0]]*waypoints.shape[0]),np.zeros((waypoints.shape[0],)),wayrotations),axis=-1)
            path_pcd = cpu_pointcloud_from_array(waypoints,np.ones_like(waypoints) * np.array([0,0,0]))
            if waypoints.shape[0] > 500:
                continue
            
            o3d.io.write_point_cloud("%s/%s/trajectory_%d/path.ply"%(args.output_dir,house_id,trajectory_index),path_pcd+path_planner.navigable_pcd)
            cv2.imwrite("%s/%s/trajectory_%d/decision_map_path.jpg"%(args.output_dir,house_id,trajectory_index),result_path_map)
            cv2.imwrite("%s/%s/trajectory_%d/esdf.jpg"%(args.output_dir,house_id,trajectory_index), path_planner.esdf_color)

            fps_writer = imageio.get_writer("%s/%s/trajectory_%d/fps.mp4"%(args.output_dir,house_id,trajectory_index), fps=10)
            depth_writer = imageio.get_writer("%s/%s/trajectory_%d/depth.mp4"%(args.output_dir,house_id,trajectory_index), fps=10)
            
            camera_trajectory = []
            count = 0
            for pt,rt in zip(waypoints,wayrotations):
                count += 1
                # if count % 20 != 0:
                #     continue
                bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(pt, rt))
                camera_trajectory.append(bproc.math.build_transformation_mat(pt, rt).tolist())

            bproc.camera.set_intrinsics_from_K_matrix(camera_intrinsic,args.image_width,args.image_height)
            data = bproc.renderer.render()

            save_flag = True
            for ci,color,depth in zip(np.arange(len(data['colors'])),data['colors'],data['depth']):
                if np.where(color.sum(axis=-1)==0)[0].shape[0] > 5000:
                    save_flag = False
                    break
                fps_writer.append_data(color)
                depth_writer.append_data((np.clip(depth/5.0,0,1)*255.0).astype(np.uint8))
                cv2.imwrite("%s/%s/trajectory_%d/rgb/%d.jpg"%(args.output_dir,house_id,trajectory_index,ci),cv2.cvtColor(color,cv2.COLOR_BGR2RGB))
                cv2.imwrite("%s/%s/trajectory_%d/depth/%d.png"%(args.output_dir,house_id,trajectory_index,ci),(depth * 10000.0).astype(np.uint16))
            fps_writer.close()
            depth_writer.close()
            save_dict = {'camera_intrinsic':camera_intrinsic.tolist(),
                        'camera_extrinsic':tobase_extrinsic.tolist(),
                        'camera_trajectory':camera_trajectory}
            json_object = json.dumps(save_dict, indent=4)
            with open("%s/%s/trajectory_%d/data.json"%(args.output_dir,house_id,trajectory_index), "w") as outfile:
                outfile.write(json_object)
            
            if not save_flag:
                shutil.rmtree("%s/%s/"%(args.output_dir,house_id))
                break
            
            ########################Render a bev map for debug ########################
            bev_cam_pos = (camera_translation + target_point) / 2
            bev_cam_pos[2] = 2.2
            bev_camera_intrinsic = generate_intrinsic(800,800,100,100)
            bpy.context.scene.frame_end = 0
            bproc.camera.set_intrinsics_from_K_matrix(bev_camera_intrinsic,800,800)
            frame = bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(bev_cam_pos, [0, 0, 0]))
            data = bproc.renderer.render()
            # cv2.imwrite("%s/%s/trajectory_%d/bev.jpg"%(args.output_dir,house_id,trajectory_index), cv2.cvtColor(data['colors'][-1],cv2.COLOR_BGR2RGB))
            
            # plot trajectory
            # 获取相机到世界坐标系的变换矩阵以及相机内参矩阵            
            cam2world_matrix = bproc.camera.get_camera_pose(frame)
            # 遍历每个waypoint进行投影并记录投影后的2D坐标
            waypoints[:, 2] = 0.0
            projected_points_2d = bproc.camera.project_points(waypoints, frame).astype(int)
            image = cv2.cvtColor(data['colors'][-1], cv2.COLOR_BGR2RGB)
            # 在渲染图像上绘制红色的点来表示投影后的waypoints
            for point_2d in projected_points_2d:
                u = point_2d[0]
                v = point_2d[1]
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    cv2.circle(image, (u, v), 3, (0, 0, 255), -1)

            # 保存绘制好点的图像
            cv2.imwrite("%s/%s/trajectory_%d/bev_traj.jpg" % (args.output_dir, house_id, trajectory_index), image)
            ######################## Render bev finished        ########################

        
        

        