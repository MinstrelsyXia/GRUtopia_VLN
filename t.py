import open3d as o3d
import numpy as np

loaded_pcd = o3d.io.read_point_cloud("point_cloud.pcd")
loaded_coordinate_frame = o3d.io.read_triangle_mesh("coordinate_frame.ply")

o3d.visualization.draw_geometries([loaded_pcd, loaded_coordinate_frame])
