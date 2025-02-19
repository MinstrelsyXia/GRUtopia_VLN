import blenderproc as bproc
import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
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
#debugpy.listen(5678)
#debugpy.wait_for_client()
# define input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scene_index",type=int,default=0)
parser.add_argument("--gpu_id",type=int,default=2)
parser.add_argument("--front", help="Path to the 3D front file", default='/ssd/caiwenzhe/scene_datasets/3D-Front/3D-FRONT/')
parser.add_argument("--future_folder", help="Path to the 3D Future Model folder.",default='/ssd/caiwenzhe/scene_datasets/3D-Front/3D-FUTURE-model')
parser.add_argument("--front_3D_texture_path", help="Path to the 3D FRONT texture folder.",default='/ssd/caiwenzhe/scene_datasets/3D-Front/3D-FRONT-texture')
parser.add_argument("--cc_material_path", help="Path to the 3D FRONT texture folder.",default='/ssd/caiwenzhe/resources/cctextures')
parser.add_argument("--output_dir", help="Path to where the data should be saved",default="./demo_trajectory/3dfront/")
parser.add_argument("--image_height",type=int,default=180)
parser.add_argument("--image_width",type=int,default=320)
parser.add_argument("--camera_hfov",type=float,default=68)
parser.add_argument("--camera_vfov",type=float,default=42)
parser.add_argument("--ceiling_height",type=float,default=1.8)
parser.add_argument("--safe_distance",type=float,default=0.25)
args = parser.parse_known_args()[0]
if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")
# blenderproc initialization
bproc.init()
bproc.renderer.set_render_devices(False,"CUDA",[args.gpu_id])
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
cc_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble", "Rock", "Metal", "Concrete"])
floor_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Wood","WoodFloor","Tiles","Marble","Rock","Grass","Ground","PavingStones",'Carpet'])
wood_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Wood","Tiles","Bricks","Rock",'Metal','Fabric'])
marble_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Bricks",'Concrete','PaintedWood','Marble','Rock'])
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
scene_json_path = os.listdir(args.front)[args.scene_index]

if not os.path.exists("%s/%s"%(args.output_dir,scene_json_path[:-5])):
    bproc.renderer.set_max_amount_of_samples(1024)
    front_path = os.path.join(args.front,scene_json_path)
    loaded_objects = bproc.loader.load_front3d(
        json_path=front_path,
        future_model_path=args.future_folder,
        front_3D_texture_path=args.front_3D_texture_path,
        label_mapping=mapping
    )
    # blenderproc randomize textures
    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
    walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
    # convert blenderproc meshes into pointcloud
    scene_pcd = o3d.geometry.PointCloud()
    floor_heights = []
    flag = False
    for obj in loaded_objects:
        if not isinstance(obj,bproc.types.MeshObject):
            continue
        category = mapping.label_from_id(obj.get_cp('category_id'))
        obj_mesh = obj.mesh_as_trimesh().apply_transform(obj.get_local2world_mat())
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
        new_mesh.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(obj_mesh.vertex_normals)
        new_pcd = new_mesh.sample_points_uniformly(50000)
        new_pcd = new_pcd.voxel_down_sample(0.05)
        new_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(np.array(new_pcd.points)) * 0.4)
        scene_pcd = scene_pcd + new_pcd
        scene_pcd = scene_pcd.voxel_down_sample(0.05)
        if 'floor' in category:
            if len(floor_heights) == 0:
                floor_heights.append(np.array(new_pcd.points)[:,2].mean())
            else:
                distance = np.array(floor_heights) - np.array(new_pcd.points)[:,2].mean()
                if np.min(distance) > 1.0:
                    floor_heights.append(np.array(new_pcd.points)[:,2].mean())
                    
     # initialize the path_planner
    path_planner = PathPlanner(ceiling_offset=args.ceiling_height,safe_distance=args.safe_distance)
    scene_points = np.array(scene_pcd.points)
    current_floor = np.random.choice(floor_heights)
    path_planner.reset(current_floor,0.5,scene_pcd)
    
    for trajectory_num in range(100):
        for floor in floors:
            material = random.choice(floor_materials)
            for i in range(len(floor.get_materials())):
                floor.set_material(i, material)
        for door in baseboards_and_doors:
            material = random.choice(wood_materials)
            for i in range(len(door.get_materials())):
                door.set_material(i, material)
        for wall in walls:
            material = random.choice(marble_materials)
            for i in range(len(wall.get_materials())):
                wall.set_material(i, material)
        
        bproc.renderer.set_light_bounces(diffuse_bounces=np.random.randint(100,600), glossy_bounces=np.random.randint(100,600), max_bounces=np.random.randint(100,600),transmission_bounces=np.random.randint(100,600),transparent_max_bounces=np.random.randint(100,600))
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
        status,waypoints,wayrotations = path_planner.generate_trajectory(camera_translation,target_point)
        if status == False:
            continue
        waypoints = np.concatenate((waypoints,np.ones((waypoints.shape[0],1)) * camera_translation[2]),axis=-1)
        wayrotations = np.stack((np.array([camera_rotation[0]]*waypoints.shape[0]),np.zeros((waypoints.shape[0],)),wayrotations),axis=-1)
        path_pcd = cpu_pointcloud_from_array(waypoints,np.ones_like(waypoints) * np.array([0,0,0]))
        if waypoints.shape[0] > 500:
            continue
    
        os.makedirs("%s/%s/trajectory_%d/"%(args.output_dir,scene_json_path[:-5],trajectory_num),exist_ok=False)
        os.makedirs("%s/%s/trajectory_%d/rgb/"%(args.output_dir,scene_json_path[:-5],trajectory_num),exist_ok=False)
        os.makedirs("%s/%s/trajectory_%d/depth/"%(args.output_dir,scene_json_path[:-5],trajectory_num),exist_ok=False)
        o3d.io.write_point_cloud("%s/%s/trajectory_%d/path.ply"%(args.output_dir,scene_json_path[:-5],trajectory_num),path_pcd+path_planner.navigable_pcd)
        cv2.imwrite("%s/%s/trajectory_%d/decision_map.jpg"%(args.output_dir,scene_json_path[:-5],trajectory_num),path_planner.color_decision_map)
        fps_writer = imageio.get_writer("%s/%s/trajectory_%d/fps.mp4"%(args.output_dir,scene_json_path[:-5],trajectory_num), fps=10)
        depth_writer = imageio.get_writer("%s/%s/trajectory_%d/depth.mp4"%(args.output_dir,scene_json_path[:-5],trajectory_num), fps=10)
        
        camera_trajectory = []
        for pt,rt in zip(waypoints,wayrotations):
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
            cv2.imwrite("%s/%s/trajectory_%d/rgb/%d.jpg"%(args.output_dir,scene_json_path[:-5],trajectory_num,ci),cv2.cvtColor(color,cv2.COLOR_BGR2RGB))
            cv2.imwrite("%s/%s/trajectory_%d/depth/%d.png"%(args.output_dir,scene_json_path[:-5],trajectory_num,ci),(depth * 10000.0).astype(np.uint16))
        fps_writer.close()
        depth_writer.close()
        save_dict = {'camera_intrinsic':camera_intrinsic.tolist(),
                    'camera_extrinsic':tobase_extrinsic.tolist(),
                    'camera_trajectory':camera_trajectory}
        json_object = json.dumps(save_dict, indent=4)
        with open("%s/%s/trajectory_%d/data.json"%(args.output_dir,scene_json_path[:-5],trajectory_num), "w") as outfile:
            outfile.write(json_object)
        
        if not save_flag:
            shutil.rmtree("%s/%s/"%(args.output_dir,scene_json_path[:-5]))
            break

        
        

        