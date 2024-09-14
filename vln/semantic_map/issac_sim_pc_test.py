from omni.isaac.lab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# from omni.isaac.kit import SimulationApp
# sim_app = SimulationApp({"headless": True})
import matplotlib.pyplot as plt
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera

import open3d as o3d

# example: see the position of two cubes, 3d->2d->3d
# my_world = World(stage_units_in_meters=1.0)

# cube_2 = my_world.scene.add(
#     DynamicCuboid(
#         prim_path="/new_cube_2",
#         name="cube_1",
#         position=np.array([5.0, 3, 1.0]),
#         scale=np.array([0.6, 0.5, 0.2]),
#         size=1.0,
#         color=np.array([255, 0, 0]),
#     )
# )

# cube_3 = my_world.scene.add(
#     DynamicCuboid(
#         prim_path="/new_cube_3",
#         name="cube_2",
#         position=np.array([-5, 1, 3.0]),
#         scale=np.array([0.1, 0.1, 0.1]),
#         size=1.0,
#         color=np.array([0, 0, 255]),
#         linear_velocity=np.array([0, 0, 0.4]),
#     )
# )

# camera = Camera(
#     prim_path="/World/camera",
#     position=np.array([0.0, 0.0, 25.0]),
#     frequency=20,
#     resolution=(256, 256),
#     orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
# )

# my_world.scene.add_default_ground_plane()
# my_world.reset()
# camera.initialize()

# i = 0
# camera.add_motion_vectors_to_frame()
# reset_needed = False
# while simulation_app.is_running():
#     my_world.step(render=True)
#     print(camera.get_current_frame())
#     if i == 100:

#         points_2d = camera.get_image_coords_from_world_points(
#             np.array([cube_3.get_world_pose()[0], cube_2.get_world_pose()[0]])
#         )
#         points_3d = camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
#         print(points_2d)
#         print(points_3d)
#         imgplot = plt.imshow(camera.get_rgba()[:, :, :3])
#         plt.show()
#         print(camera.get_current_frame()["motion_vectors"])
#     if my_world.is_stopped() and not reset_needed:
#         reset_needed = True
#     if my_world.is_playing():
#         if reset_needed:
#             my_world.reset()
#             reset_needed = False
#     i += 1


# simulation_app.close()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
my_world.reset()
pose = np.loadtxt("/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606/poses.txt")

camera = Camera(
    prim_path="/World/h1/torso_link/h1_pano_camera_0",
    position=pose[0,:3],
    frequency=20,
    resolution=(256, 256),
    orientation=pose[0,3:],
)
from omni.isaac.core.utils.prims import get_prim_at_path
focal_length_value = 18
horizontal_aperture_value = 20
vertical_aperture_value = 20    
camera.initialize()
camera.add_motion_vectors_to_frame()
camera_prim = get_prim_at_path("/World/camera")
camera_prim.GetAttribute("focalLength").Set(focal_length_value)
camera_prim.GetAttribute("horizontalAperture").Set(horizontal_aperture_value)
camera_prim.GetAttribute("verticalAperture").Set(vertical_aperture_value)
depth_map1 = np.load("/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606/depth/pano_camera_0_depth_step_8.npy")

depth_map2 = np.load("/ssd/xiaxinyuan/code/w61-grutopia/logs/sample_episodes/s8pcmisQ38h/id_2606/depth/pano_camera_0_depth_step_11.npy")

height, width = depth_map1.shape

# Generate a meshgrid of pixel coordinates
x = np.arange(width)
y = np.arange(height)
xx, yy = np.meshgrid(x, y)

# Flatten the meshgrid arrays to correspond to the flattened depth map
xx_flat = xx.flatten()
yy_flat = yy.flatten()

# Combine the flattened x and y coordinates into a 2D array of points
points_2d = np.vstack((xx_flat, yy_flat)).T  # Shape will be (N, 2), where N = height * width



depth_map1=depth_map1.flatten() 

while simulation_app.is_running():
    points_3d = camera.get_world_points_from_image_coords(points_2d, depth_map1)

    o3d.visualization.draw_geometries([points_3d])
    points_2d = camera.get_image_coords_from_world_points(
                points_3d)

    print(points_2d)
simulation_app.close()