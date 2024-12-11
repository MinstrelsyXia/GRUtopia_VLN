import os
import sys
import json
import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
from path_utils.path_planner import PathPlanner
from path_utils.geometry_tools import *
sys.path.append("/home/PJLAB/caiwenzhe/Desktop/agile2d_project/agile2d_generation_pipeline/")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_obj_dir",type=str,default="/home/PJLAB/caiwenzhe/Desktop/agile2d_project/agile2d_generation_pipeline/demo_assets/")
    parser.add_argument("--output_type",type=str,default='obj')
    parser.add_argument("--ceiling_offset",type=float,default=1.5)
    parser.add_argument("--robot_height",type=float,default=1.0)
    parser.add_argument("--safe_distance",type=str,default=0.25) 
    parser.add_argument("--episodes",type=int,default=100) 
    parser.add_argument("--image_width",type=int,default=320)
    parser.add_argument("--image_height",type=int,default=180)
    parser.add_argument("--camera_hfov",type=float,default=69)
    parser.add_argument("--camera_vfov",type=float,default=42)
    parser.add_argument("--output_dir",type=str,default="/home/PJLAB/caiwenzhe/Desktop/agile2d_project/agile2d_generation_pipeline/demo_trajectory_grutopia/")
    parser.add_argument("--export_animations",action='store_true')
    parser.add_argument("--merge_meshes",action='store_true')
    parser.add_argument("--export_materials",action='store_true')
    args = parser.parse_known_args()[0]
    return args

args = get_args()
path_planner = PathPlanner(ceiling_offset=args.ceiling_offset,safe_distance=args.safe_distance)
for filename in os.listdir(args.input_obj_dir):
    house_id = filename
    print(os.path.join(args.input_obj_dir,filename,"start_result.obj"))
    mesh = o3d.io.read_triangle_mesh(os.path.join(args.input_obj_dir,filename,"start_result.obj"))
    scene_pcd = mesh.sample_points_uniformly(number_of_points=1000000)
    scene_pcd = scene_pcd.voxel_down_sample(0.05)
    scene_pcd_points = np.array(scene_pcd.points) * 0.01
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_points)
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

# open isaacsim and load the stage
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core import World
import imageio

world = World(physics_dt=0.05, rendering_dt=0.01, stage_units_in_meters=1.0)
camera = Camera(prim_path="/World/camera",
                position=np.array([0,0,0]),
                dt=0.05,
                resolution=(1280, 720),
                orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 90.0]),degrees=True),
                )
camera.set_focal_length(1.4)
camera.set_focus_distance(0.205)
camera.set_clipping_range(0.01,10000)

add_reference_to_stage(os.path.join(args.input_obj_dir,house_id,"start_result.usd"),"/World")
world.reset()
camera.initialize()
camera.add_motion_vectors_to_frame()
camera.add_distance_to_image_plane_to_frame()
for i in range(20):
    world.step(render=True)
    
for filename in tqdm(os.listdir("%s/%s"%(args.output_dir,house_id))):
    trajectory_data_path = os.path.join(args.output_dir,house_id,filename,"data.json")
    trajectory_data = json.load(open(trajectory_data_path))
    camera_intrinsic = np.array(trajectory_data['camera_intrinsic'])
    camera_extrinsic = np.array(trajectory_data['camera_extrinsic'])
    camera_trajectory = np.array(trajectory_data['camera_trajectory'])

    fps_writer = imageio.get_writer(os.path.join(args.output_dir,house_id,filename,"fps.mp4"),fps=10)
    depth_writer = imageio.get_writer(os.path.join(args.output_dir,house_id,filename,"depth.mp4"),fps=10)
    os.makedirs(os.path.join(args.output_dir,house_id,filename,'rgb/'))
    os.makedirs(os.path.join(args.output_dir,house_id,filename,'depth/'))
    
    #initialize the camera pose
    for _ in range(10):
        camera_pos = camera_trajectory[0,0:3,3]*100.0
        camera_rot = camera_trajectory[0,0:3,0:3]
        world.step(render=True)
        camera_euler_angles = rot_utils.rot_matrices_to_quats(camera_rot)
        camera_euler_angles = rot_utils.quats_to_euler_angles(camera_euler_angles)
        camera_euler_angles[0],camera_euler_angles[1],camera_euler_angles[2] = camera_euler_angles[1],np.clip((np.pi/2 - camera_euler_angles[0])/2.0,0,np.pi/8),camera_euler_angles[2]+np.pi/2
        camera.set_world_pose(camera_pos,rot_utils.euler_angles_to_quats(camera_euler_angles))
        world.step(render=True)
    
    for frame_index,camera_ext in enumerate(camera_trajectory):
        camera_pos = camera_ext[0:3,3]*100.0
        camera_rot = camera_ext[0:3,0:3]
        world.step(render=True)
        camera_euler_angles = rot_utils.rot_matrices_to_quats(camera_rot)
        camera_euler_angles = rot_utils.quats_to_euler_angles(camera_euler_angles)
        camera_euler_angles[0],camera_euler_angles[1],camera_euler_angles[2] = camera_euler_angles[1],np.clip((np.pi/2 - camera_euler_angles[0])/2.0,0,np.pi/8),camera_euler_angles[2]+np.pi/2
        camera.set_world_pose(camera_pos,rot_utils.euler_angles_to_quats(camera_euler_angles))
        
        rgb = cv2.cvtColor(camera.get_rgba()[:,:,:3],cv2.COLOR_BGR2RGB)
        depth = camera.get_depth()
        depth[np.isinf(depth)] = 0.0
        depth = depth / 100.0
        
        fps_writer.append_data(camera.get_rgba()[:,:,:3])
        depth_writer.append_data((depth*10000.0).astype(np.uint16))
        cv2.imwrite(os.path.join(args.output_dir,house_id,filename,'rgb/',"%d.jpg"%frame_index),rgb)
        cv2.imwrite(os.path.join(args.output_dir,house_id,filename,'depth/',"%d.png"%frame_index),(depth*10000.0).astype(np.uint16))

    