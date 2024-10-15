
class NotFound(Exception):
    """Exception raised when a specific object is not found on the map."""
    def __init__(self, message="Object not found or insufficient data in pc_mask."):
        self.message = message
        super().__init__(self.message)

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

################### angle utils ########################

#! identical to vln.utils



################### visualize utils ########################
def visualize_naive_occupancy_map(occupied_ids,save_path):
    # visualize semantic map:
    occupancy_map = np.zeros(occupied_ids.shape[:2])
    for i in range(occupied_ids.shape[0]):
        for j in range(occupied_ids.shape[1]):
            for k in range(occupied_ids.shape[2]):
                if occupied_ids[i,j,k] > 0:
                    occupancy_map[i,j] = 1

    # visualize occupancy_map:
    
    plt.imsave(save_path,occupancy_map,cmap='gray')

import cv2
def visualize_matrix(matrix, save_path):
    """
    使用 OpenCV 可视化一个 0/1 的 2D 矩阵
    :param matrix: 要可视化的矩阵: bool shape=(n,n)
    :param window_name: 窗口名称
    """
    # 将矩阵转换为图像格式
    img = (matrix.astype(np.float32) * 255).astype(np.uint8)
    
    # 显示图像
    cv2.imwrite(save_path,img)

def visualize_visgraph_and_path(G, path_vg, save_path='visgraph_with_path.png'):
    """
    可视化已构建好的可见图及路径。
    
    参数:
    - G: 已经构建好的可见图对象 (VisGraph)
    - path_vg: 由 shortest_path 生成的路径
    - save_path: 保存路径的文件名
    """
    # 创建一个图形
    fig, ax = plt.subplots()

    # 绘制可见图的边
    for edge in G.visgraph.edges:
        start = edge.p1
        end = edge.p2
        ax.plot([start.x, end.x], [start.y, end.y], 'g--')  # 绿色虚线代表可见边

    # 绘制路径
    if path_vg:
        path_xs = [p.x for p in path_vg]
        path_ys = [p.y for p in path_vg]
        ax.plot(path_xs, path_ys, 'r-', linewidth=2, marker='o')  # 红色线条，带圆点
        ax.plot(path_xs[0], path_ys[0], 'bo', label='Start')      # 蓝色起点标记
        ax.plot(path_xs[-1], path_ys[-1], 'ro', label='Goal')     # 红色终点标记

    # 设置坐标系比例，保持x和y尺度相同
    ax.set_aspect('equal')

    # 添加图例
    ax.legend()

    # 保存图像到文件
    plt.savefig(save_path)
    print(f"Graph with path saved as {save_path}")
    plt.show()
    # visualize_visgraph_and_path(G, path_vg, save_path='tmp/visgraph_with_path_output.png')

def save_point_cloud_image(pcd, save_path="point_cloud.jpg"):
    # 设置无头渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window()  # 创建一个不可见的窗口
    ctr = vis.get_view_control()

    # 设定特定的视角
    ctr.set_front([0, 0, -1])  # 设置相机朝向正面
    ctr.set_lookat([0, 0, 0])  # 设置相机目标点为原点
    ctr.set_up([0, 0, 1])   
    # 创建点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # 捕获当前视图并保存为图像
    vis.capture_screen_image(save_path)
    vis.destroy_window()
              
def visualize_pc(pcd,headless,save_path = 'pc.jpg'):
    '''
    pcd:     after:    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(points_3d)
    '''
    if headless==True:
        save_point_cloud_image(pcd,save_path=save_path)
        return
    else:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]) 
        o3d.io.write_point_cloud("point_cloud.pcd", pcd)
        o3d.io.write_triangle_mesh("coordinate_frame.ply", coordinate_frame)
        return

def get_dummy_2d_grid(width,height):
    # Generate a meshgrid of pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # Flatten the meshgrid arrays to correspond to the flattened depth map
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Combine the flattened x and y coordinates into a 2D array of points
    points_2d = np.vstack((xx_flat, yy_flat)).T  # Shape will be (N, 2), where N = height * width
    return points_2d

def downsample_pc(pc, depth_sample_rate):
    '''
    INput: points:(N,3); rate:downsample rate:int
    Output: downsampled_points:(N/rate,3)
    '''
    # np.random.seed(42)
    shuffle_mask = np.arange(pc.shape[0])
    np.random.shuffle(shuffle_mask)
    shuffle_mask = shuffle_mask[::depth_sample_rate]
    pc = pc[shuffle_mask,:]
    return pc


def save_point_cloud_image(pcd, save_path="point_cloud.jpg"):
    # 设置无头渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window()  # 创建一个不可见的窗口
    ctr = vis.get_view_control()

    # 设定特定的视角
    ctr.set_front([0, 0, -1])  # 设置相机朝向正面
    ctr.set_lookat([0, 0, 0])  # 设置相机目标点为原点
    ctr.set_up([0, 0, 1])   
    # 创建点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # 捕获当前视图并保存为图像
    vis.capture_screen_image(save_path)
    vis.destroy_window()
              



