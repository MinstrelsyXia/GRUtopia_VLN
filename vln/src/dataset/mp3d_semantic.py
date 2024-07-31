import numpy as np
import torch
import trimesh
import pandas as pd

import isaacsim
from pxr import Usd, UsdGeom, Gf, Sdf

from grutopia_extension.utils import get_stage_prim_paths

def load_mp3d_cat40(category_mapping_file, mpcat40_file, device="cpu"):
    mapping = pd.read_csv(category_mapping_file, sep="\t")
    mapping_mpcat40 = torch.tensor(mapping["mpcat40index"].to_numpy(), device=device, dtype=torch.long)
    # load defined colors for mpcat40
    mapping_40 = pd.read_csv(mpcat40_file, sep="\t")
    color = mapping_40["hex"].to_numpy()
    color = torch.tensor(
        [(int(color[i][1:3], 16), int(color[i][3:5], 16), int(color[i][5:7], 16)) for i in range(len(color))],
        device=device,
        dtype=torch.uint8,
    )
    return mapping_mpcat40, color

def load_ply(ply_file):
    curr_trimesh = trimesh.load(ply_file)
    faces_raw = curr_trimesh.metadata["_ply_raw"]["face"]["data"]
    face_id_category_mapping = torch.tensor(
        [single_face[3] for single_face in faces_raw], device=device
    )
    
    vertices = curr_trimesh.vertices
    faces = curr_trimesh.faces
    # Assuming each vertex has a semantic label
    # semantic_labels = curr_trimesh.visual.vertex_colors[:, 3]  # Assuming the semantic label is stored in the alpha channel
    return vertices, faces, face_id_category_mapping

def map_semantics_to_usd(usd_file, vertices, faces, face_id_category_mapping, mapping_mpcat40, color):
    stage = Usd.Stage.Open(usd_file)
    root = stage.GetPrimAtPath("/World")
    # stage_prims = get_stage_prim_paths(print_prim=False)
    
    for i, face in enumerate(faces):
        face_vertices = vertices[face]
        face_semantic_label = face_id_category_mapping[face].flatten().type(torch.long)
        # Assuming that each face has a unique semantic label
        semantic_category = mapping_mpcat40[face_semantic_label]
        face_color = color[semantic_category].tolist()
        print(1)
        
        # Create a new Mesh in USD for each face (or modify existing mesh)
        # mesh_name = f"Face_{i}"
        # mesh_path = f"/World/{mesh_name}"
        # mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
        
        # # Set the vertices and faces
        # mesh_prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in face_vertices])
        # mesh_prim.GetFaceVertexIndicesAttr().Set(face.tolist())
        # mesh_prim.GetFaceVertexCountsAttr().Set([3])
        
        # # Set the color (semantic information)
        # color_attr = mesh_prim.CreateDisplayColorAttr()
        # color_attr.Set([Gf.Vec3f(*[c/255.0 for c in face_color])])
    
    # Save the modified USD
    # stage.GetRootLayer().Save()

def read_navmesh(navmesh_file):
    with open(navmesh_file, 'r', encoding="cp1252") as f:
        habitat_navmesh = f.read()
    return habitat_navmesh

if __name__ == "__main__":
    # category_mapping_file = "/home/pjlab/w61/GRUtopia/vln/data/mappings/category_mapping.tsv"
    # mpcat40_file = "/home/pjlab/w61/GRUtopia/vln/data/mappings/mpcat40.tsv"
    # usd_file = "/home/pjlab/Matterport3D/data/v1/scans/1LXtFkjw3qL/matterport_mesh/b94039b4eb8947bdb9ff5719d9173eae/isaacsim_b94039b4eb8947bdb9ff5719d9173eae.usd"
    # ply_file = "/home/pjlab/Matterport3D/data/v1/scans/1LXtFkjw3qL/house_segmentations/1LXtFkjw3qL.ply"
    # device = "cpu"
    # mapping_mpcat40, color = load_mp3d_cat40(category_mapping_file, mpcat40_file, device)
    # vertices, faces, face_id_category_mapping = load_ply(ply_file)
    # map_semantics_to_usd(usd_file, vertices, faces, face_id_category_mapping, mapping_mpcat40, color)
    
    habitat_navmesh_file = "/home/pjlab/Matterport3D/data/v1/tasks/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.navmesh"
    read_navmesh(habtiat)
    print(1)
