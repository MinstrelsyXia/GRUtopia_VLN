import os
import sys
import json
import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
from path_utils.path_planner import PathPlanner
from path_utils.geometry_tools import *

import json
import time
from Scaffold_GS.gaussian_renderer import render, prefilter_voxel
from tqdm import tqdm
from Scaffold_GS.utils.general_utils import safe_state
from argparse import ArgumentParser
from Scaffold_GS.arguments import ModelParams, PipelineParams, get_combined_args
from Scaffold_GS.gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
    name_list = []
    per_view_dict = {}
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        rendering = render_pkg["render"]

    
sys.path.append("/home/PJLAB/caiwenzhe/Desktop/agile2d_project/agile2d_generation_pipeline/")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str,default="/home/PJLAB/caiwenzhe/Desktop/agile2d_project/3dgs_assets/")
    parser.add_argument("--ceiling_offset",type=float,default=2.0)
    parser.add_argument("--robot_height",type=float,default=1.0)
    parser.add_argument("--safe_distance",type=str,default=0.25)
    parser.add_argument("--scale",type=float,default=0.2)
    parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--image_width",type=int,default=320)
    parser.add_argument("--image_height",type=int,default=180)
    parser.add_argument("--camera_hfov",type=float,default=69)
    parser.add_argument("--camera_vfov",type=float,default=42)
    parser.add_argument("--output_dir",type=str,default="/home/PJLAB/caiwenzhe/Desktop/agile2d_project/agile2d_generation_pipeline/demo_trajectory_3dgs/")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
path_planner = PathPlanner(ceiling_offset=args.ceiling_offset,safe_distance=args.safe_distance)
for scene_name in os.listdir(args.input_dir):
    house_id = scene_name
    mesh_path = os.path.join(args.input_dir,scene_name,'mesh.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    scene_pcd = mesh.sample_points_uniformly(number_of_points=1000000)
    scene_pcd_points = np.array(scene_pcd.points) * args.scale
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_points)
    scene_pcd = scene_pcd.voxel_down_sample(0.05)
    floor_height = np.quantile(np.array(scene_pcd.points)[:,2],0.1)
    path_planner.reset(floor_height,args.robot_height,scene_pcd)
    o3d.io.write_point_cloud("test.ply",path_planner.navigable_pcd)
    for trajectory_num in tqdm(range(args.episodes)):
        trajectory_index = trajectory_num
        
        camera_height = np.random.uniform(0.25,1.25)
        camera_intrinsic = generate_intrinsic(args.image_width,args.image_height,args.camera_hfov,args.camera_vfov)
        random_index = np.random.choice(np.nonzero(path_planner.safe_value > 0.1)[0])
        camera_translation = np.array(path_planner.navigable_pcd.points)[random_index]
        camera_translation[2] = camera_height + floor_height
        pitch_rad = np.deg2rad((210 - 180*camera_height))
        camera_rotation = [np.clip(pitch_rad,np.pi/3,np.pi/2),0,0]
        tobase_extrinsic = build_transformation_mat(camera_translation * np.array([0,0,1]),camera_rotation)
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
        camera_trajectory = []
        for pt,rt in zip(waypoints,wayrotations):
            camera_trajectory.append(build_transformation_mat(pt, rt).tolist())
        save_dict = {'camera_intrinsic':camera_intrinsic.tolist(),
                     'camera_extrinsic':tobase_extrinsic.tolist(),
                     'camera_trajectory':camera_trajectory}
        json_object = json.dumps(save_dict, indent=4)
        
        os.makedirs("%s/%s/trajectory_%d/"%(args.output_dir,house_id,trajectory_index),exist_ok=True)
        o3d.io.write_point_cloud("%s/%s/trajectory_%d/path.ply"%(args.output_dir,house_id,trajectory_index),path_pcd+path_planner.navigable_pcd)
        with open("%s/%s/trajectory_%d/data.json"%(args.output_dir,house_id,trajectory_index), "w") as outfile:
            outfile.write(json_object)