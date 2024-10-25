# import json
from typing import Any, Dict, List

import numpy as np

from grutopia.core.config import SimulatorConfig
from grutopia.core.util import log


class BaseEnv:
    """
    Env base class. All tasks should inherit from this class(or subclass).
    ----------------------------------------------------------------------
    """

    def __init__(self, config: SimulatorConfig, headless: bool = True, webrtc: bool = False, native: bool = False) -> None:
        self._simulation_config = None
        self._render = None
        # Setup Multitask Env Parameters
        self.env_map = {}
        self.obs_map = {}

        self.config = config.config
        self.env_num = config.env_num
        self._column_length = int(np.sqrt(self.env_num))

        # Init Isaac Sim
        import isaacsim
        from omni.isaac.kit import SimulationApp
        self.headless = headless
        # self._simulation_app = SimulationApp({'headless': self.headless, 'anti_aliasing': 0, 'renderer': 'RayTracing'})
        self._simulation_app = SimulationApp({'headless': self.headless, 'anti_aliasing': 0,'multi_gpu': False}) # !!!

        if webrtc:
            from omni.isaac.core.utils.extensions import enable_extension  # noqa

            self._simulation_app.set_setting('/app/window/drawMouse', True)
            self._simulation_app.set_setting('/app/livestream/proto', 'ws')
            self._simulation_app.set_setting('/app/livestream/websocket/framerate_limit', 60)
            self._simulation_app.set_setting('/ngx/enabled', False)
            enable_extension('omni.services.streamclient.webrtc')

        elif native:
            from omni.isaac.core.utils.extensions import enable_extension  # noqa

            self._simulation_app.set_setting("/app/window/drawMouse", True)
            self._simulation_app.set_setting("/app/livestream/proto", "ws")
            self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
            self._simulation_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.livestream.native")

        from grutopia.core import datahub  # noqa E402.
        from grutopia.core.runner import SimulatorRunner  # noqa E402.

        self._runner = SimulatorRunner(config=config)
        # self._simulation_config = sim_config

        log.debug(self.config.tasks)
        # create tasks
        self._runner.add_tasks(self.config.tasks)
        return

    @property
    def runner(self):
        return self._runner

    @property
    def is_render(self):
        return self._render

    def get_dt(self):
        return self._runner.dt

    def step(self, actions: List[Dict[str, Any]], add_rgb_subframes=False, render=False) -> List[Dict[str, Any]]:
        """
        run step with given action(with isaac step)

        Args:
            actions (List[Dict[str, Any]]): action(with isaac step)

        Returns:
            List[Dict[str, Any]]: observations(with isaac step)
        """
        if len(actions) != len(self.config.tasks):
            raise AssertionError('len of action list is not equal to len of task list')
        _actions = []
        for action_idx, action in enumerate(actions):
            _action = {}
            for k, v in action.items():
                _action[f'{k}_{action_idx}'] = v
            _actions.append(_action)
        action_after_reshape = {
            self.config.tasks[action_idx].name: action
            for action_idx, action in enumerate(_actions)
        }

        # log.debug(action_after_reshape)
        self._runner.step(action_after_reshape, add_rgb_subframes=add_rgb_subframes, render=render)
        observations = self.get_observations()
        return observations

    def reset(self, envs: List[int] = None):
        """
        reset the environment(use isaac word reset)

        Args:
            envs (List[int]): env need to be reset(default for reset all envs)
        """
        if envs is not None:
            if len(envs) == 0:
                return
            log.debug(f'============= reset: {envs} ==============')
            # int -> name
            self._runner.reset([self.config.tasks[e].name for e in envs])
            return self.get_observations(), {}
        self._runner.reset()
        return self.get_observations(), {}

    def get_observations(self, add_rgb_subframes=False) -> List[Dict[str, Any]]:
        """
        Get observations from Isaac environment
        Returns:
            List[Dict[str, Any]]: observations
        """
        _obs = self._runner.get_obs(add_rgb_subframes=add_rgb_subframes)
        return _obs

    def render(self, mode='human'):
        return

    def close(self):
        """close the environment"""
        self._simulation_app.close()
        return

    @property
    def simulation_config(self):
        """config of simulation environment"""
        return self._simulation_config

    @property
    def simulation_app(self):
        """simulation app instance"""
        return self._simulation_app
    
    def reset_env(self):
        # tasks = self.runner._world._current_tasks
        # self.runner._world._current_tasks['h1_locomotion_0'].robots['h1_0'].sensors['camera'].__del__() # !!!!
        self.runner._world.clear()
