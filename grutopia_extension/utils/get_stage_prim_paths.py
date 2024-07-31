import omni

from pxr import Usd, UsdGeom

def get_stage_prim_paths(print_prim=False):
    # Get the current stage
    stage = omni.usd.get_context().get_stage()

    # Function to recursively get all prim paths
    def get_all_prim_paths(prim, prim_paths):
        prim_paths.append(prim.GetPath().pathString)
        for child in prim.GetChildren():
            get_all_prim_paths(child, prim_paths)

    # Get all prim paths in the stage
    root_prim = stage.GetPseudoRoot()
    all_prim_paths = []
    get_all_prim_paths(root_prim, all_prim_paths)

    # Print all prim paths
    if print_prim:
        for path in all_prim_paths:
            print(path)
    
    return all_prim_paths