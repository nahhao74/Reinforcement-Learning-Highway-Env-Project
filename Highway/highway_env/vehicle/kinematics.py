from __future__ import annotations
import copy
from collections import deque
import numpy as np
from highway_env import utils
from highway_env.vehicle.objects import RoadObject

# SỬA FILE kinematics.py

class Vehicle(RoadObject):
    LENGTH = 5.0
    WIDTH = 2.0
    MAX_SPEED = 60.0 
    MIN_SPEED = -40.0

    # ... (Giữ nguyên __init__ và các hàm khác)

    @classmethod
    def create_random(cls, road, speed=None, lane_id=None, spacing=1):
        """
        Tạo xe ngẫu nhiên.
        SỬA: Mặc định speed = 60km/h (16.67 m/s) cho vật cản.
        """
        # Chọn làn đường (Nếu lane_id không được chỉ định thì chọn random)
        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        
        # SỬA LOGIC TỐC ĐỘ
        if speed is None:
            speed = 16.67 # 60 km/h cố định cho vật cản
        
        # Logic rải xe: Tính toán vị trí x0 sao cho không đè lên xe khác
        # Lấy offset dựa trên mật độ xe hiện tại trên làn đó
        vehicles_on_lane = [v for v in road.vehicles if v.lane_index == (_from, _to, _id)]
        n_vehicles = len(vehicles_on_lane)
        
        # Khoảng cách an toàn cơ bản
        default_spacing = 20 + 1.0 * speed 
        
        # Tìm vị trí trống (đơn giản hóa để chạy nhanh)
        if n_vehicles == 0:
            x0 = road.np_random.uniform(0, 100) # Xe đầu tiên random trong 100m đầu
        else:
            # Tìm xe xa nhất trên làn này
            last_vehicle_pos = max([v.position[0] for v in vehicles_on_lane])
            # Đặt xe mới phía trước xe xa nhất một khoảng random
            x0 = last_vehicle_pos + default_spacing * road.np_random.uniform(0.9, 1.2) * spacing

        return cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)

    def act(self, action=None):
        if action:
            self.action = action

    def step(self, dt):
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = -1.0 * self.speed

        # Kinematic Bicycle Model (Simplified)
        beta = np.arctan(1 / 2 * np.tan(float(self.action["steering"])))
        v = self.speed * np.array([np.cos(self.heading + beta), np.sin(self.heading + beta)])
        self.position += v * dt
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += float(self.action["acceleration"]) * dt
        self.speed = np.clip(self.speed, self.MIN_SPEED, self.MAX_SPEED)

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            "presence": 1,
            "x": self.position[0], "y": self.position[1],
            "vx": self.velocity[0], "vy": self.velocity[1],
            "heading": self.heading,
            # QUAN TRỌNG: Thêm lane_id để Agent biết mình đang ở đâu
            "lane_id": self.lane_index[2] if self.lane_index else 0 
        }
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                if key in d:
                    d[key] -= origin_dict[key]
        return d