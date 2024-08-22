import isaacsim

from omni.isaac.kit import SimulationApp

# Initialize the simulation app
CONFIG = {
    "headless": True,  # Set to True if running in headless mode (without GUI)
}
simulation_app = SimulationApp(CONFIG)