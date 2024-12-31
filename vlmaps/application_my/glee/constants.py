from vlmaps.application_my.glee.object_list import categories
GLEE_CONFIG_PATH = "vlmaps/vlmaps/GLEE/configs/SwinL.yaml"
GLEE_CHECKPOINT_PATH = "vlmaps/vlmaps/GLEE/weights/GLEE_SwinL_Scaleup10m.pth"
DETECT_OBJECTS = [[cat['name'].lower()] for cat in categories]
INTEREST_OBJECTS = ['bed','chair','toilet','potted_plant','television_set','sofa']



