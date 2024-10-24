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



PCD_GLOBAL = o3d.geometry.PointCloud()


def test_pc(camera,depth):
    global PCD_GLOBAL
    grid_2d =  get_dummy_2d_grid(depth.shape[1],depth.shape[0])
    pc = camera.get_world_points_from_image_coords(grid_2d, depth.flatten())
    pc_downsampled = downsample_pc(pc, 150)
    pcd_global = o3d.geometry.PointCloud()
    pcd_global.points = o3d.utility.Vector3dVector(pc_downsampled)
    PCD_GLOBAL+=pcd_global
    visualize_pc(PCD_GLOBAL,headless=False, save_path = "1.jpg")