import lmdb
import msgpack_numpy
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime
import os

def world_to_pixel(world_pose, camera_pose,aperture,width,height):
    cx, cy = camera_pose[0]*10/aperture*width, -camera_pose[1]*10/aperture*height
    X, Y = world_pose[0]*10/aperture*width, -world_pose[1]*10/aperture*height
    pixel_x = width - (X - cx + width/2)
    pixel_y = Y - cy + height/2
    return [pixel_x, pixel_y]

def vis_nav_path(start_pixel, goal_pixel, path, occupancy_map, img_save_path='path_planning.jpg'):
    cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])
    bounds = [0, 1, 3, 254, 256]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(10, 10))
    # plt.imshow(occupancy_map, cmap='binary', origin='lower')
    plt.imshow(occupancy_map, cmap=cmap, norm=norm, origin='upper')

    # Plot start and goal points
    plt.plot(start_pixel[1], start_pixel[0], 'ro', markersize=6, label='Start')
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo', markersize=6, label='Goal')

    # Plot the path
    if len(path) > 0:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'xb-', linewidth=1, markersize=5, label='Path')

    # Customize the plot
    plt.title('Path planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.colorbar(label='Occupancy (0: Free, 1: Occupied)')

    # Save the plot
    plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
    print(f"Saved path planning visualization to {img_save_path}")
    plt.close()
def generate_eval_key(ckpt_name, path_key):
    return f"eval_{ckpt_name}_{path_key}"
ckpt_name="ckpt.cma"
path_key="15_1"
project_path = '/ssd/zhaohui/workspace/w61_grutopia_0102'
name = '20250102_ckpt_cma'
lmdb_path = project_path + f'/data/sample_episodes/{name}'
database_read = lmdb.open(f"{lmdb_path}/sample_data.lmdb", readonly=True, lock=False)
# database_write = lmdb.open(f"{lmdb_path}/sample_data.lmdb", map_size=1 * 1024 * 1024 * 1024 * 1024, max_dbs=0)
# with database_write.begin(write=True) as txn:
#     key_write = generate_eval_key(ckpt_name=ckpt_name,path_key=path_key).encode()
#     txn.delete(key_write)
with database_read.begin() as txn:
    key = generate_eval_key(ckpt_name=ckpt_name,path_key=path_key).encode()
    value = txn.get(key)
    if value is None:
        print(f"value is None")
        sys.exit()
    else:
        value = msgpack_numpy.unpackb(value)
        reference_path=value['reference_path']
        pred_traj_list=value['pred_traj_list']
        # exe_path=value['ext_info']['exe_path']
        camera_pose=value['ext_info']['camera_pose']
        aperture=value['ext_info']['aperture']
        width=value['ext_info']['width']
        height=value['ext_info']['height']
        map_info=value['ext_info']['map_info']
        reference_path_pixel = []
        for point in reference_path:
            pixel = world_to_pixel(point,camera_pose,aperture,width,height)
            reference_path_pixel.append(pixel)
        pred_traj_pixel = []
        for point in pred_traj_list:
            pixel = world_to_pixel(point,camera_pose,aperture,width,height)
            pred_traj_pixel.append(pixel)
        # exe_path_pixel = []
        # for point in exe_path:
        #     pixel = world_to_pixel(point,camera_pose,aperture,width,height)
        #     exe_path_pixel.append(pixel)
        date_str = path_key #f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
        file_name = f"{date_str}_reference.jpg" 
        vis_nav_path(
            start_pixel=reference_path_pixel[0], 
            goal_pixel=reference_path_pixel[-1], 
            path=reference_path_pixel,
            occupancy_map=map_info,
            img_save_path=os.path.join(f'{project_path}/test/zhaohui/', file_name)
        )
        # file_name = f"{date_str}_expect.jpg" 
        # vis_nav_path(
        #     start_pixel=reference_path_pixel[0], 
        #     goal_pixel=reference_path_pixel[-1], 
        #     path=exe_path_pixel,
        #     occupancy_map=map_info,
        #     img_save_path=os.path.join(f'{project_path}/test/zhaohui/', file_name)
        # )
        file_name = f"{date_str}_real.jpg" 
        vis_nav_path(
            start_pixel=reference_path_pixel[0], 
            goal_pixel=reference_path_pixel[-1], 
            path=pred_traj_pixel,
            occupancy_map=map_info,
            img_save_path=os.path.join(f'{project_path}/test/zhaohui/', file_name)
        )