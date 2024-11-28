
class NotFound(Exception):
    """Exception raised when a specific object is not found on the map."""
    def __init__(self, message="Object not found or insufficient data in pc_mask."):
        self.message = message
        super().__init__(self.message)
class EarlyFound(Exception):
    '''
    Exception raised when a subgoal is not reached but te ultimate goal is found
    '''
    def __init__(self, message="Subgoal not reached but ultimate goal is found."):
        self.message = message
        super().__init__(self.message)


import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import re

def extract_parameters(input_string):
    """
    提取字符串中括号内的所有参数，并去除引号
    :param input_string: 输入字符串，如 "self.move_to_object('open doors')" 或 "self.move_in_between('desk','chair')"
    :return: 提取的参数列表，如 ["open doors"] 或 ["desk", "chair"]
    """
    # 使用正则表达式匹配所有引号内的内容
    matches = re.findall(r'[\'"](.*?)[\'"]', input_string)
    
    return matches if matches else None

def extract_self_methods(input_string):
    """
    从输入字符串中提取以 `self.` 开头的所有部分，并放到一个列表中
    :param input_string: 输入字符串
    :return: 提取的部分列表
    """
    # 使用正则表达式匹配以 `self.` 开头的部分
    matches = re.findall(r'self\.\w+\(.*?\)', input_string)
    return matches

# # 示例输入
# input_string = "python\nself.move_to_left('counter')\nself.face('counter')\nself.move_west('counter')\nself.with_object_on_right('counter')\nself.move_east('chair')\nself.move_to_object('sofa')\n"

# # 提取以 `self.` 开头的部分
# extracted_methods = extract_self_methods(input_string)

# # 打印结果
# print(extracted_methods)

################### angle utils ########################

#! identical to vln.utils



################### visualize utils ########################

import numpy as np
import matplotlib.pyplot as plt

def visualize_subgoal_images(rgbs, similarities, chosen_idx, subgoal, save_path):
    """
    将多张图像拼接并添加说明文字
    
    Args:
        rgbs: 图像列表
        similarities: 相似度列表 
        chosen_idx: 被选中的图像索引
        subgoal: 子目标描述
        save_path: 保存路径
    """
    rgbs = [np.asarray(img).astype(np.uint8) if not isinstance(img, np.ndarray) 
else img.astype(np.uint8) for img in rgbs]
    # 计算图像网格布局
    n_images = len(rgbs)
    n_cols = min(4, n_images)  # 每行最多4张图
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # 创建图像
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    # 添加总标题
    plt.suptitle(f"Subgoal is: {subgoal}, chosen image {chosen_idx}", fontsize=14)
    
    # 绘制每张图像
    for idx, (img, sim) in enumerate(zip(rgbs, similarities)):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.imshow(img)
        plt.axis('off')
        
        # 为选中的图像使用红色标题,其他使用黑色
        color = 'red' if idx == chosen_idx else 'black'
        plt.title(f"Image {idx}, similarity: {sim:.3f}", color=color, pad=10)
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为总标题留出空间
    plt.savefig(save_path)
    plt.savefig('/tmp/choice_of_frontier.png')
    plt.close()



def visualize_occupancy_map_with_contours(occupied_ids, contours, curr_pos, curr_angle_deg, save_path):
    # 创建占用地图
    occupancy_map = np.zeros(occupied_ids.shape[:2])
    for i in range(occupied_ids.shape[0]):
        for j in range(occupied_ids.shape[1]):
            for k in range(occupied_ids.shape[2]):
                if occupied_ids[i, j, k] > 0:
                    occupancy_map[i, j] = 1

    # 创建可视化图
    plt.imshow(occupancy_map, cmap='gray', alpha=0.5)

    # 绘制contours
    for contour in contours:
        plt.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2)

    # 计算朝向的结束点
    curr_x, curr_y = curr_pos
    angle_rad = np.deg2rad(curr_angle_deg)
    arrow_length = 10  # 可以根据需要调整箭头长度
    end_x = curr_x + arrow_length * np.cos(angle_rad)
    end_y = curr_y + arrow_length * np.sin(angle_rad)

    # 绘制朝向箭头
    plt.arrow(curr_x, curr_y, end_x - curr_x, end_y - curr_y, head_width=1, head_length=2, fc='red', ec='red')

    # 保存可视化图
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()



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
              
def visualize_pc(pcd,headless=False,save_path = 'pc.jpg'):
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




