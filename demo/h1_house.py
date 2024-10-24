from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
import numpy as np
# file_path = './GRUtopia/demo/configs/h1_house.yaml'
file_path = 'demo/configs/h1_house.yaml'

sim_config = SimulatorConfig(file_path)

headless = False
webrtc = False

if is_in_container():
    headless = True
    webrtc = True

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name

i = 0

# import omni.kit.viewport

# # 获取当前的视口窗口
# viewport_window = omni.kit.viewport.window.get_viewport_window_instances()

# # 设置灯光模式为 Camera Light
# viewport_window.set_light_enabled(True)  # 打开灯光
# viewport_window.set_light_mode(omni.kit.viewport_legacy.LightMode.CAMERA)  # 设置为 Camera Light 模式
# actions = {'h1': {'move_with_keyboard': []}}
actions = {'h1': {'move_along_path':[np.array([[1,1,1.06],[2,2,1.06],[3,3,1.06],[10,10,1.06]])]}}

while env.simulation_app.is_running():
    i += 1
    # if i< 10000:
    #     env_actions = [{'h1': {'stand_still': []}}]
    #     env.step(actions=env_actions)
    #     continue
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)

    if i % 100 == 0:
        print(i)

env.simulation_app.close()
