import os
import cv2
from natsort import natsorted
from collections import defaultdict

def cameras_to_video(image_folder, output_video_name, camera_list=['pano_camera_0'], fps=30):
    img_dict = defaultdict(list)
    for file in os.listdir(image_folder):
        if file.endswith(".png"):
            for camera in camera_list:
                if camera in file:
                    img_dict[camera].append(os.path.join(image_folder, file))
    
    for camera_name, images in img_dict.items():
        images_to_video(image_folder, output_video_name, camera_name, images, fps)

def images_to_video(image_folder, output_video_name, camera_name=None, images=None, fps=30):
    if images is None:
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
    if camera_name is not None:
        output_video_name = output_video_name.replace('.mp4', f'_{camera_name}.mp4')
    
    # 自然排序图片
    images = natsorted(images)
    
    # 读取第一张图片来获取尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # cv2.destroyAllWindows()
    video.release()

    print(f"Video saved as {output_video_name}")

# 使用示例
image_folder = '/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes/train/7y3sRwLe3Va/id_991'
path_id = image_folder.split('/')[-1]
output_video = f'logs/images/output_video_{path_id}.mp4'
fps = 2  # 你可以根据需要调整帧率
camera_list = ['pano_camera_0', 'h1_pano_camera_debug']

cameras_to_video(image_folder, output_video, camera_list, fps)
# images_to_video(image_folder, output_video, fps)