# import torch

# model_path = 'data/checkpoints/cma/ckpt.10.pth'
# try:
#     # 加载模型权重，如果在 CPU 上加载 GPU 保存的模型，需要添加 map_location
#     ckpt_weight = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
#     print(f"模型成功加载自: {model_path}")
# except Exception as e:
#     print(f"加载模型时出错: {str(e)}")

import torch
from habitat.config.default import Config
from torch.serialization import add_safe_globals

model_path = 'data/checkpoints/cma/ckpt.10.pth'
try:
    # 添加 Config 到安全全局变量列表
    add_safe_globals([Config])
    ckpt_weight = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"模型成功加载自: {model_path}")
except Exception as e:
    print(f"加载模型时出错: {str(e)}")