from __future__ import annotations
import functools
from typing import TYPE_CHECKING, Callable, Union
import numpy as np
from gymnasium import spaces
from highway_env.vehicle.controller import MDPVehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]

class ActionType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        raise NotImplementedError

    def act(self, action: Action) -> None:
        raise NotImplementedError

    def get_available_actions(self):
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle

class DiscreteMetaAction(ActionType):
    ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

    def __init__(self, env, target_speeds=None, **kwargs):
        super().__init__(env)
        # 22 m/s = 79.2 km/h
        # 34 m/s = 122.4 km/h
        # Chia thành 5 mức tốc độ để Action FASTER/SLOWER mượt mà
        default_speeds = np.linspace(22, 34, 5) 
        
        self.target_speeds = np.array(target_speeds) if target_speeds is not None else default_speeds
        self.actions = self.ACTIONS_ALL
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(MDPVehicle, target_speeds=self.target_speeds)

    def act(self, action: int) -> None:
        self.controlled_vehicle.act(self.actions[int(action)])

    def get_available_actions(self) -> list[int]:
        return list(self.actions.keys())

def action_factory(env: AbstractEnv, config: dict) -> ActionType:
    if config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    raise ValueError("Unknown action type")