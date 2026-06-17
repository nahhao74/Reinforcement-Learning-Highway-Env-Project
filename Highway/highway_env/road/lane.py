from __future__ import annotations
from abc import ABCMeta, abstractmethod
import numpy as np
from highway_env import utils

class AbstractLane:
    __metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    line_types: list[LineType]

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        raise NotImplementedError()

    def on_lane(self, position: np.ndarray, longitudinal: float = None, lateral: float = None, margin: float = 0) -> bool:
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = (
            np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin
            and -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_on

    def is_reachable_from(self, position: np.ndarray) -> bool:
        if self.forbidden: return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = (
            np.abs(lateral) <= 2 * self.width_at(longitudinal)
            and 0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        )
        return is_close

    def after_end(self, position: np.ndarray, longitudinal: float = None, lateral: float = None) -> bool:
        """
        Kiểm tra xem xe đã đi qua điểm cuối của làn đường chưa.
        """
        if longitudinal is None:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    def distance(self, position: np.ndarray):
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(self, position: np.ndarray, heading: float | None, heading_weight: float = 1.0):
        if heading is None: return self.distance(position)
        s, r = self.local_coordinates(position)
        angle = np.abs(utils.wrap_to_pi(heading - self.heading_at(s)))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight * angle

class LineType:
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3

class StraightLane(AbstractLane):
    """Làn đường thẳng - Loại duy nhất bạn cần cho Highway."""
    def __init__(self, start, end, width=AbstractLane.DEFAULT_WIDTH, line_types=None, forbidden=False, speed_limit=20, priority=0):
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        self.heading = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start + longitudinal * self.direction + lateral * self.direction_lateral

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width


    
    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        # Tối ưu hóa: Tránh tạo array mới và gọi np.dot nếu không cần thiết
        # Code cũ: delta = position - self.start
        # Code cũ: longitudinal = np.dot(delta, self.direction)
        
        dx = position[0] - self.start[0]
        dy = position[1] - self.start[1]
        
        # Tính tay tích vô hướng (Dot Product) nhanh hơn overhead của numpy cho array nhỏ
        longitudinal = dx * self.direction[0] + dy * self.direction[1]
        lateral = dx * self.direction_lateral[0] + dy * self.direction_lateral[1]
        
        return float(longitudinal), float(lateral)

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": "highway_env.road.lane.StraightLane",
            "config": {
                "start": self.start.tolist(),
                "end": self.end.tolist(),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }

def lane_from_config(cfg: dict) -> AbstractLane:
    return utils.class_from_path(cfg["class_path"])(**cfg["config"])