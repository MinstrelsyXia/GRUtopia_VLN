# import random
import traceback
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict

from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask as OmniBaseTask
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg

from grutopia.core.config import TaskUserConfig
from grutopia.core.robot import init_robots
from grutopia.core.scene import create_object, create_scene
from grutopia.core.task.metric import BaseMetric, create_metric
from grutopia.core.util import log

import omni.usd
from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
import omni.kit.actions.core


class BaseTask(OmniBaseTask, ABC):
    """
    wrap of omniverse isaac sim's base task

    * enable register for auto register task
    * contains robots
    """
    tasks = {}

    def __init__(self, config: TaskUserConfig, scene: Scene):
        self.objects = None
        self.robots = None
        name = config.name
        offset = config.offset
        super().__init__(name=name, offset=offset)
        self._scene = scene
        self.config = config

        self.metrics: dict[str, BaseMetric] = {}
        self.steps = 0
        self.work = True

        for metric_config in config.metrics:
            self.metrics[metric_config.name] = create_metric(metric_config)

    def load(self):
        if self.config.scene_asset_path is not None:
            # source, prim_path = create_scene(self.config.scene_asset_path,
            #                                  prim_path_root=f'World/env_{self.config.env_id}/scene')
            # create_prim(prim_path,
            #             usd_path=source,
            #             scale=self.config.scene_scale,
            #             translation=[self.config.offset[idx] + i for idx, i in enumerate(self.config.scene_position)])
            prim_path = f"/World/env_{self.config.env_id}/scene"
            _xform_prim = prim_utils.create_prim(
                prim_path= f"/World/env_{self.config.env_id}/scene", 
                translation=[self.config.offset[idx] + i for idx, i in enumerate(self.config.scene_position)], 
                usd_path=self.config.scene_asset_path
            )

            # apply collider properties
            collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
            sim_utils.define_collision_properties(_xform_prim.GetPrimPath(), collider_cfg)

            # create physics material
            physics_material = RigidBodyMaterialCfg(
                static_friction=1, # 0.5
                dynamic_friction=1, # 0.5
                restitution=0.0,
                improve_patch_friction=True,
                friction_combine_mode='average',
                restitution_combine_mode='average',
                compliant_contact_stiffness=0.0,
                compliant_contact_damping=0.0
            )

            physics_material_cfg: sim_utils.RigidBodyMaterialCfg = physics_material
            # spawn the material
            physics_material_cfg.func(f"{prim_path}/physicsMaterial", physics_material)
            sim_utils.bind_physics_material(_xform_prim.GetPrimPath(), f"{prim_path}/physicsMaterial")

            # add colliders and physics material
            ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=physics_material)
            ground_plane = ground_plane_cfg.func(f"{prim_path}/GroundPlane", ground_plane_cfg)
            ground_plane.visible = False

            # lights
            action_registry = omni.kit.actions.core.get_action_registry()
            # switches to camera lighting
            action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
            action.execute()

        self.robots = init_robots(self.config, self._scene)
        self.objects = {}
        for obj in self.config.objects:
            _object = create_object(obj)
            _object.set_up_scene(self._scene)
            self.objects[obj.name] = _object
        log.info(self.robots)
        log.info(self.objects)

    def set_up_scene(self, scene: Scene) -> None:
        self._scene = scene
        self.load()

    def get_observations(self, data_type=None) -> Dict[str, Any]:
        """
        Returns current observations from the objects needed for the behavioral layer.

        Return:
            Dict[str, Any]: observation of robots in this task
        """
        if not self.work:
            return {}
        obs = {}
        for robot_name, robot in self.robots.items():
            try:
                obs[robot_name] = robot.get_obs(data_type=data_type)
            except Exception as e:
                log.error(self.name)
                log.error(e)
                traceback.print_exc()
                return {}
        return obs

    def update_metrics(self):
        for _, metric in self.metrics.items():
            metric.update()

    def calculate_metrics(self) -> dict:
        metrics_res = {}
        for name, metric in self.metrics.items():
            metrics_res[name] = metric.calc()

        return metrics_res

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns True of the task is done.

        Raises:
            NotImplementedError: this must be overridden.
        """
        raise NotImplementedError

    def individual_reset(self):
        """
        reload this task individually without reloading whole world.
        """
        raise NotImplementedError

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        self.steps += 1
        return

    def post_reset(self) -> None:
        """Calls while doing a .reset() on the world."""
        self.steps = 0
        for robot in self.robots.values():
            robot.post_reset()
        return

    @classmethod
    def register(cls, name: str):
        """
        Register a task with its name(decorator).
        Args:
            name(str): name of the task
        """

        def decorator(tasks_class):
            cls.tasks[name] = tasks_class

            @wraps(tasks_class)
            def wrapped_function(*args, **kwargs):
                return tasks_class(*args, **kwargs)

            return wrapped_function

        return decorator


def create_task(config: TaskUserConfig, scene: Scene):
    task_cls = BaseTask.tasks[config.type]
    return task_cls(config, scene)
