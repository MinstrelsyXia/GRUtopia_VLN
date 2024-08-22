import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 加载预训练模型和特征提取器
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# ADE20K
label_map = {
    0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "tree", 5: "ceiling", 6: "road", 7: "bed",
    8: "windowpane", 9: "grass", 10: "cabinet", 11: "sidewalk", 12: "person", 13: "earth",
    14: "door", 15: "table", 16: "mountain", 17: "plant", 18: "curtain", 19: "chair",
    20: "car", 21: "water", 22: "painting", 23: "sofa", 24: "shelf", 25: "house", 26: "sea",
    27: "mirror", 28: "rug", 29: "field", 30: "armchair", 31: "seat", 32: "fence", 33: "desk",
    34: "rock", 35: "wardrobe", 36: "lamp", 37: "bathtub", 38: "railing", 39: "cushion",
    40: "base", 41: "box", 42: "column", 43: "signboard", 44: "chest of drawers", 45: "counter",
    46: "sand", 47: "sink", 48: "skyscraper", 49: "fireplace", 50: "refrigerator", 51: "grandstand",
    52: "path", 53: "stairs", 54: "runway", 55: "case", 56: "pool table", 57: "pillow",
    58: "screen door", 59: "stairway", 60: "river", 61: "bridge", 62: "bookcase", 63: "blind",
    64: "coffee table", 65: "toilet", 66: "flower", 67: "book", 68: "hill", 69: "bench",
    70: "countertop", 71: "stove", 72: "palm", 73: "kitchen island", 74: "computer", 75: "swivel chair",
    76: "boat", 77: "bar", 78: "arcade machine", 79: "hovel", 80: "bus", 81: "towel",
    82: "light", 83: "truck", 84: "tower", 85: "chandelier", 86: "awning", 87: "streetlight",
    88: "booth", 89: "television receiver", 90: "airplane", 91: "dirt track", 92: "apparel",
    93: "pole", 94: "land", 95: "bannister", 96: "escalator", 97: "ottoman", 98: "bottle",
    99: "buffet", 100: "poster", 101: "stage", 102: "van", 103: "ship", 104: "fountain",
    105: "conveyer belt", 106: "canopy", 107: "washer", 108: "plaything", 109: "swimming pool",
    110: "stool", 111: "barrel", 112: "basket", 113: "waterfall", 114: "tent", 115: "bag",
    116: "minibike", 117: "cradle", 118: "oven", 119: "ball", 120: "food", 121: "step",
    122: "tank", 123: "trade name", 124: "microwave", 125: "pot", 126: "animal", 127: "bicycle",
    128: "lake", 129: "dishwasher", 130: "screen", 131: "blanket", 132: "sculpture", 133: "hood",
    134: "sconce", 135: "vase", 136: "traffic light", 137: "tray", 138: "ashcan", 139: "fan",
    140: "pier", 141: "crt screen", 142: "plate", 143: "monitor", 144: "bulletin board",
    145: "shower", 146: "radiator", 147: "glass", 148: "clock", 149: "flag"
}
# 加载图像
image = Image.open("vln/semantic_map/pano_camera_180_rgba_5000.jpg")

# 预处理图像
inputs = feature_extractor(images=image, return_tensors="pt")

# 模型推理
outputs = model(**inputs)
logits = outputs.logits

# 获取预测结果
predicted_mask = logits.argmax(dim=1)[0]

num_classes = len(label_map)
colors = plt.cm.get_cmap('tab20c')(np.linspace(0, 1, num_classes))
custom_cmap = ListedColormap(colors)

# 创建图像对比
plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
im = plt.imshow(predicted_mask, cmap=custom_cmap, interpolation='nearest')
plt.title("Segmentation Mask")
plt.axis('off')

# 添加颜色条
cbar = plt.colorbar(im, ticks=range(num_classes), orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_ticklabels([label_map[i] for i in range(num_classes)])
cbar.set_label('Classes', rotation=270, labelpad=25)

plt.tight_layout()

# 保存结果
plt.savefig('vln/semantic_map/result_with_labels.jpg', dpi=300, bbox_inches='tight')
print("Result with labels has been saved as 'result_with_labels.jpg'")


