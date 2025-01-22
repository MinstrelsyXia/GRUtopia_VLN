import os
import shutil
import isaacsim
from omni.isaac.kit import SimulationApp
_simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0})
from pxr import Usd, UsdShade, Sdf

def copy_fixed_usd_files(src_dir, dest_dir):
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == "fixed.usd":
                # Construct the full path of the file
                src_file_path = os.path.join(root, file)

                # Determine the relative path for the destination
                rel_path = os.path.relpath(root, src_dir)

                # Create the corresponding directory in the destination
                dest_sub_dir = os.path.join(dest_dir, rel_path)
                os.makedirs(dest_sub_dir, exist_ok=True)

                # Construct the destination file path
                dest_file_path = os.path.join(dest_sub_dir, file)

                # Copy the file to the destination
                shutil.copy2(src_file_path, dest_file_path)
                print(f'Copied: {src_file_path} to {dest_file_path}')

import os
import shutil

def bak_and_update_configs_for_fixed(src_dir):
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == "config.yaml":
                # Construct the full path of the file
                src_file_path = os.path.join(root, file)

                # Determine the relative path for the destination
                rel_path = os.path.relpath(root, src_dir)

                # Create the corresponding directory in the destination
                dest_file_path = src_file_path.replace('config.yaml', 'config_bak.yaml')

                # Copy the file to the destination
                shutil.copy2(src_file_path, dest_file_path)
                print(f'Copied: {src_file_path} to {dest_file_path}')

                # Open config.yaml, modify the third line, and save
                with open(src_file_path, 'r') as f:
                    lines = f.readlines()

                # Modify the third line (index 2) to change usd_file_name to 'fixed.usd'
                if len(lines) >= 3:
                    lines[2] = 'usd_file_name: fixed.usd\n'

                # Write the modified lines back to config.yaml
                with open(src_file_path, 'w') as f:
                    f.writelines(lines)
                print(f'Modified: {src_file_path} to set usd_file_name to fixed.usd')

def update_usd_aldebo_map_per_path(usd_path, docker_path=None):
    # 打开 USD 文件
    # stage = Usd.Stage.Open('/ssd/wangliuyi/code/Matterport3D/data/v1/scans/1LXtFkjw3qL/matterport_mesh/b94039b4eb8947bdb9ff5719d9173eae/fixed_copy.usd')
    # stage = Usd.Stage.Open('/ssd/wangliuyi/code/Matterport3D/data/v1/scans/V2XKFyX4ASd/matterport_mesh/04d3f2105168491db767ad1fe7bc39df/fixed.usd')
    stage = Usd.Stage.Open(usd_path)

    # 获取 stage 下的所有 prim
    all_prims = [prim for prim in stage.Traverse()]
    all_prim_paths = [prim.GetPath() for prim in all_prims]


    # 遍历 Looks 文件夹下的所有材质 Prim
    for prim_idx, prim in enumerate(all_prims):
        prim_path = all_prim_paths[prim_idx]
        if 'jpg' not in prim_path.pathString:
            continue
        
        # 检查是否是 UsdShade.Shader 类型的 Prim
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            
            # 获取 Shader 的所有输入
            shader_inputs = shader.GetInputs()

            # 遍历所有输入，查找 Albedo 纹理（通常是 diffuseTexture 或类似名称）
            for shader_input in shader_inputs:
                if shader_input.GetBaseName() == "diffuse_texture":  # 根据实际命名调整
                    # 获取当前 Albedo 纹理的路径
                    current_value = shader_input.Get()
                    print(f"Shader {prim.GetPath()} 当前 Albedo 纹理路径: {current_value}")

                    current_path = current_value.path
                    text_file = current_path.split('/')[-1]
                    current_path_split = current_path.split('/')
                    for split_idx, split in enumerate(current_path_split):
                        if split == 'scans':
                            scan_name = current_path_split[split_idx+1]
                        if split == 'matterport_mesh':
                            mesh_name = current_path_split[split_idx+1]

                    # 相对路径，GUI直接打开usd可以索引，但是程序里索引不到
                    # new_path = '/'.join(current_path.split('/')[-2:])
                    # new_path = './' + new_path 

                    # 绝对路径
                    if docker_path is not None:
                        new_path = os.path.join(docker_path,'data/v1/scans',scan_name,'matterport_mesh',mesh_name,'textures',text_file)
                    else:
                        new_path = os.path.join('/'.join(usd_path.split('/')[:-1]), 'textures', text_file)

                    if docker_path is not None:
                        new_path_1 = new_path.split('/')[8]
                        new_path_2 = new_path.split('/')[-1].split('_')[0]
                    else:
                        new_path_1 = new_path.split('/')[10]
                        new_path_2 = new_path.split('/')[-1].split('_')[0]
                    if new_path_1 != new_path_2:
                        print(1)
                    # 设置新的 Albedo 纹理路径
                    new_texture_path = Sdf.AssetPath(new_path)
                    shader_input.Set(new_texture_path)

                    print(f"Shader {prim.GetPath()} 的 Albedo 纹理路径已修改为: {new_texture_path.path}")

    # 保存修改
    stage.GetRootLayer().Save()

    print("USD 文件中的 Albedo 纹理路径已更新")


def update_usd_aldebo_map_path(src_dir, copy_to_new=False, docker_path=None):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == "fixed.usd":
                # Construct the full path of the file
                src_file_path = os.path.join(root, file)
                if copy_to_new:
                    new_src_file_path = src_file_path.replace('fixed.usd', 'fixed_docker.usd') # for docker
                    shutil.copy2(src_file_path, new_src_file_path)
                    update_usd_aldebo_map_per_path(new_src_file_path, docker_path=docker_path)
                else:
                    update_usd_aldebo_map_per_path(src_file_path)


if __name__ == '__main__':
    src_file_path = '/ssd/share/Matterport3D/data/v1/scans/E9uDoFAP3SH/matterport_mesh/e996abcc45ad411fa7f406025fcf2a63/fixed_copy.usd'
    docker_path='/isaac-sim/Matterport3D'
    update_usd_aldebo_map_per_path(src_file_path , docker_path)