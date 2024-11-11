import isaacsim
from omni.isaac.kit import SimulationApp
import time

start = time.time()
_simulation_app = SimulationApp({'headless': True, 'anti_aliasing': 0})
end = time.time()
print(f"duration: {end - start} s")