from __future__ import annotations  # This allows us to hint types that do not yet exist like omni.usd etc
import isaacsim
import carb
import omni.kit.app
from omni.isaac.kit import SimulationApp
import os
app = omni.kit.app.get_app()
    '/ssd/share/isaac-sim-4.0.0/exts/omni.isaac.kit/omni/isaac/kit/simulation_app.py', 
args = [
    '/ssd/share/isaac-sim-4.0.0/apps/omni.isaac.sim.python.kit', 
    '--/app/tokens/exe-path=/ssd/share/isaac-sim-4.0.0/kit', 
    '--/persistent/app/viewport/displayOptions=3094', 
    '--/rtx/materialDb/syncLoads=True', 
    '--/rtx/hydra/materialSyncLoads=True', 
    '--/omni.kit.plugin/syncUsdLoads=True', 
    '--/app/renderer/resolution/width=1280', 
    '--/app/renderer/resolution/height=720', 
    '--/app/window/width=1440', 
    '--/app/window/height=900', 
    '--/renderer/multiGpu/enabled=True', 
    '--/app/fastShutdown=True', 
    '--ext-folder', 
    '/ssd/share/isaac-sim-4.0.0/exts', 
    '--ext-folder', 
    '/ssd/share/isaac-sim-4.0.0/apps', 
    '--/physics/cudaDevice=0', 
    '--portable', 
    '--no-window', 
    '--/app/window/hideUi=1', 
    '--test_verbose', 
    '--path_id', 
    '1726', 
    '--split', 
    'val_seen', 
    '--vln_cfg_file', 
    'vln/configs/vln_cfg.yaml', 
    '--headless'
]
app.startup("kit", '/ssd/share/isaac-sim-4.0.0/kit', args)
import omni.usd
import time

start = time.time()
stage = omni.usd.get_context().new_stage()
end = time.time()
print(f"duration: {end - start} s")
