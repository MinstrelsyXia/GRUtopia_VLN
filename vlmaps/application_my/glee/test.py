from vlmaps.application_my.glee.glee_detector import initialize_glee, glee_segmentation, visualize_segmentation, visualize_detection
import cv2

import sys
sys.path.append('/ssd/xiaxinyuan/code/w61-grutopia/vlmaps/vlmaps/GLEE')
print(sys.path)


img_file = "sample_episodes_safe/s8pcmisQ38h/id_37/rgb/pano_camera_0_image_step_269.png"
img = cv2.imread(img_file)
model = initialize_glee(device="cuda:1")
bbox_pred, mask_pred, classes, scores = glee_segmentation(img, model,device="cuda:1")

seg = visualize_segmentation(img, classes, mask_pred)
det = visualize_detection(img, classes, bbox_pred)

cv2.imwrite("seg.png", seg)
cv2.imwrite("det.png", det)