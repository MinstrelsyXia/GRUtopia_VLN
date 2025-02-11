import os
import sys

def check_cuda_installation(paths):
    print("# CUDA 安装路径检查\n")
    
    # 定义标准CUDA目录结构中的关键文件夹
    key_directories = [
        'bin',
        'lib64',
        'include'
    ]
    
    results = []
    for path in paths:
        base_path = path  # 获取父目录
        score = 0
        found_dirs = []
        
        # 检查每个关键目录是否存在
        for dir_name in key_directories:
            full_path = os.path.join(base_path, dir_name)
            if os.path.exists(full_path):
                score += 1
                found_dirs.append(dir_name)
        
        if score > 0:
            results.append({
                'path': base_path,
                'score': score,
                'found': found_dirs
            })
    
    # 按分数排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("检查结果：\n")
    if results:
        for result in results:
            print(f"路径: {result['path']}")
            print(f"匹配度: {result['score']}/{len(key_directories)}")
            print(f"发现的关键目录: {', '.join(result['found'])}\n")
    else:
        print("未找到符合CUDA安装结构的路径")

    return results

if __name__ == "__main__":
    # 你提供的路径列表
    paths = [
        "/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/numba/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_inductor/codegen/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/include/ATen/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/include/ATen/native/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/include/torch/csrc/jit/codegen/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/include/torch/csrc/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/include/c10/cuda",
        "/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/backends/cuda",
        "/isaac-sim/exts/omni.pip.compute/pip_prebundle/cv2/cuda",
        "/isaac-sim/kit/dev/include/omni/graph/core/cuda",
        "/isaac-sim/.venv/lib/python3.10/site-packages/transformers/models/deformable_detr/custom_kernel/cuda",
        "/isaac-sim/.venv/lib/python3.10/site-packages/cv2/cuda",
        "/isaac-sim/.venv/lib/python3.10/site-packages/triton/third_party/cuda",
        "/isaac-sim/.venv/lib/python3.10/site-packages/open3d/cuda",
        "/isaac-sim/extscache/omni.sensors.nv.common-1.0.1+lx64.r.cp310/include/omni/sensors/cuda",
        "/isaac-sim/GRUtopia/vlmaps/vlmaps/GLEE/glee/models/pixel_decoder/ops/src/cuda"
    ]
    check_cuda_installation(paths)