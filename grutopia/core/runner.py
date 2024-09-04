import time
from typing import List

# import numpy as np
from omni.isaac.core import World
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa F401
from omni.physx.scripts import utils
from pxr import Usd  # noqa
import omni.replicator.core as rep

# Init
from grutopia.core.config import SimulatorConfig, TaskUserConfig
from grutopia.core.register import import_all_modules_for_register
from grutopia.core.scene import delete_prim_in_stage  # noqa F401
from grutopia.core.scene import create_object, create_scene  # noqa F401
from grutopia.core.task.task import BaseTask, create_task
from grutopia.core.util import log
from grutopia.npc import NPC


class SimulatorRunner:

    def __init__(self, config: SimulatorConfig):
        import_all_modules_for_register()

        self._simulator_config = config.config
        physics_dt = self._simulator_config.simulator.physics_dt if self._simulator_config.simulator.physics_dt is not None else None
        rendering_dt = self._simulator_config.simulator.rendering_dt if self._simulator_config.simulator.rendering_dt is not None else None
        physics_dt = eval(physics_dt) if isinstance(physics_dt, str) else physics_dt
        rendering_dt = eval(rendering_dt) if isinstance(rendering_dt, str) else rendering_dt
        self.dt = physics_dt
        log.debug(f'Simulator physics dt: {self.dt}')
        self._world = World(physics_dt=self.dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._scene = self._world.scene
        self._stage = self._world.stage

        # setup scene
        prim_path = '/'
        if self._simulator_config.env_set.bg_type is None:
            self._scene.add_default_ground_plane()
        elif self._simulator_config.env_set.bg_type != 'default':
            source, prim_path = create_scene(self._simulator_config.env_set.bg_path, prim_path_root='background')
            add_reference_to_stage(source, prim_path)

        self.npc: List[NPC] = []
        for npc_config in config.config.npc:
            self.npc.append(NPC(npc_config))

        self.render_interval = self._simulator_config.simulator.rendering_interval if self._simulator_config.simulator.rendering_interval is not None else 5
        log.info(f'rendering interval: {self.render_interval}')
        self.render_trigger = 0

    @property
    def current_tasks(self) -> dict[str, BaseTask]:
        return self._world._current_tasks

    def _warm_up(self, steps=10, render=True):
        for _ in range(steps):
            self._world.step(render=render)

    def add_tasks(self, configs: List[TaskUserConfig]):
        for config in configs:
            task = create_task(config, self._scene)
            self._world.add_task(task)

        self._world.reset()

        # for task in self.current_tasks.values():
        #     for robot in task.robots.values():
        #         for sensor in robot.sensors.values():
        #             sensor.sensor_init()

        self._warm_up()

    def step(self, actions: dict, render: bool = True, add_rgb_subframes=False):
        # start_time = time.time() 
        for task_name, action_dict in actions.items():
            task = self.current_tasks.get(task_name)
            for name, action in action_dict.items():
                if name in task.robots:
                    task.robots[name].apply_action(action)
        
        # apply_action_time = time.time() - start_time

        self.render_trigger += 1
        # render = render and self.render_trigger > self.render_interval
        render = render or self.render_trigger > self.render_interval
        if self.render_trigger > self.render_interval:
            self.render_trigger = 0
        
        if add_rgb_subframes:
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False) # !!!

        # world_step_start_time = time.time()
        self._world.step(render=render)
        # world_step_time = time.time() - world_step_start_time 

        if add_rgb_subframes:
            rep.orchestrator.step(rt_subframes=0, delta_time=0.0, pause_timeline=False) # !!!

        # log.info(f"apply_action time: {apply_action_time:.4f}s, world_step time: {world_step_time:.4f}s")

        obs = self.get_obs(add_rgb_subframes=add_rgb_subframes)


        for npc in self.npc:
            try:
                npc.feed(obs)
            except Exception as e:
                log.error(f'fail to feed npc {npc.name} with obs: {e}')

        if render:
            return obs

    def get_obs(self, add_rgb_subframes=False):
        obs = {}
        for task_name, task in self.current_tasks.items():
            obs[task_name] = task.get_observations(add_rgb_subframes=add_rgb_subframes)
        return obs

    def get_current_time_step_index(self) -> int:
        return self._world.current_time_step_index

    def reset(self, tasks: List[str] = None):
        if tasks is None:
            self._world.reset()
            return
        for task in tasks:
            self.current_tasks[task].individual_reset()

    def get_obj(self, name: str) -> XFormPrim:
        return self._world.scene.get_object(name)

    def remove_collider(self, prim_path: str):
        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.removeCollider(build)

    def add_collider(self, prim_path: str):
        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.setCollider(build, approximationShape=None)
