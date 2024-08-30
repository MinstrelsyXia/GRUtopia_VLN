import os
import cv2
from natsort import natsorted

def images_to_video(image_folder, output_video_name, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
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
image_folder = '/ssd/wangliuyi/code/w61_grutopia_new/logs/sample_episodes/train/7y3sRwLe3Va/id_593'
path_id = image_folder.split('/')[-1]
output_video = f'logs/images/output_video_{path_id}.mp4'
fps = 2  # 你可以根据需要调整帧率

images_to_video(image_folder, output_video, fps)