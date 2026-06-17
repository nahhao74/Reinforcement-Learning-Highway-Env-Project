from __future__ import annotations
from abc import ABC
import numpy as np
from highway_env import utils

class RoadObject(ABC):
    LENGTH: float = 5.0
    WIDTH: float = 2.0

    def __init__(self, road, position, heading=0, speed=0):
        self.road = road
        self.position = np.array(position, dtype=np.float64)
        self.heading = heading
        self.speed = speed
        self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading) if self.road else None
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.collidable = True
        self.solid = True
        self.crashed = False
        self.hit = False
        self.impact = np.zeros(self.position.shape)
        self.check_collisions = True
        self.diagonal = np.sqrt(self.LENGTH**2 + self.WIDTH**2)

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, speed=None):
        lane = road.network.get_lane(lane_index)
        speed = speed if speed is not None else lane.speed_limit
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed)

    def handle_collisions(self, other, dt=0):
        if other is self or not (self.collidable and other.collidable): return
        if not (self.check_collisions or other.check_collisions): return

        # Fast spherical check
        if np.linalg.norm(other.position - self.position) > (self.diagonal + other.diagonal) / 2 + self.speed * dt:
            return

        if utils.are_polygons_intersecting(self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt)[0]:
            if self.solid and other.solid:
                self.crashed = True
                other.crashed = True

    @property
    def direction(self): return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self): return self.speed * self.direction

    def polygon(self):
        points = np.array([
            [-self.LENGTH / 2, -self.WIDTH / 2], [-self.LENGTH / 2, +self.WIDTH / 2],
            [+self.LENGTH / 2, +self.WIDTH / 2], [+self.LENGTH / 2, -self.WIDTH / 2]
        ]).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[c, -s], [s, c]])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])

    def lane_distance_to(self, other):
        if not other: return np.nan
        return self.lane.local_coordinates(other.position)[0] - self.lane.local_coordinates(self.position)[0]

    @property
    def on_road(self): return self.lane.on_lane(self.position)
    
    def to_dict(self, origin_vehicle=None):
        d = {"x": self.position[0], "y": self.position[1], "vx": self.velocity[0], "vy": self.velocity[1]}
        if origin_vehicle:
            for key in d: d[key] -= origin_vehicle.to_dict()[key]
        return d

class Obstacle(RoadObject):
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.solid = True

class Landmark(RoadObject):
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.solid = False