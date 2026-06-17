from __future__ import annotations
import numpy as np
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

class IDMVehicle(ControlledVehicle):
    """ Xe Traffic: Tự động chạy theo mô hình IDM (Gas) và MOBIL (Steering) """
    ACC_MAX = 6.0
    COMFORT_ACC_MAX = 3.0
    COMFORT_ACC_MIN = -5.0
    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH
    TIME_WANTED = 1.5
    DELTA = 4.0
    POLITENESS = 0.0 # 0: Ích kỷ (chèn ép), 1: Lịch sự
    LANE_CHANGE_MIN_ACC_GAIN = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0
    LANE_CHANGE_DELAY = 1.0

    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None, target_speed=None, route=None, enable_lane_change=True, timer=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY

    def act(self, action=None):
        if self.crashed: return
        action = {}
        self.follow_road()
        
        # Lane Change Logic (MOBIL)
        if self.enable_lane_change:
            self.change_lane_policy()

        action["steering"] = self.steering_control(self.target_lane_index)

        # Acceleration Logic (IDM)
        front_vehicle, _ = self.road.neighbour_vehicles(self, self.lane_index)
        action["acceleration"] = self.idm_acceleration(self, front_vehicle)
        
        # Safe merging
        if self.lane_index != self.target_lane_index:
            front_vehicle, _ = self.road.neighbour_vehicles(self, self.target_lane_index)
            action["acceleration"] = min(action["acceleration"], self.idm_acceleration(self, front_vehicle))

        action["acceleration"] = np.clip(action["acceleration"], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)

    def step(self, dt):
        self.timer += dt
        super().step(dt)

    def idm_acceleration(self, ego, front):
        if not ego: return 0
        target_speed = getattr(ego, "target_speed", 0)
        if ego.lane and ego.lane.speed_limit: target_speed = min(target_speed, ego.lane.speed_limit)
        
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(max(ego.speed, 0) / utils.not_zero(target_speed), self.DELTA))
        if front:
            d = ego.lane_distance_to(front)
            desired_gap = self.DISTANCE_WANTED + ego.speed * self.TIME_WANTED + ego.speed * (ego.speed - front.speed) / (2 * np.sqrt(-self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN))
            acceleration -= self.COMFORT_ACC_MAX * np.power(desired_gap / utils.not_zero(d), 2)
        return acceleration

    def change_lane_policy(self):
        if self.lane_index != self.target_lane_index: return
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer): return
        
        for lane_index in self.road.network.side_lanes(self.lane_index):
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position): continue
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index):
        # Kiểm tra an toàn
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        if new_following:
             acc_new_following = self.idm_acceleration(new_following, self)
             if acc_new_following < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED: return False # Gây nguy hiểm cho xe sau
        
        # Kiểm tra lợi ích (Acc Gain)
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_acc_current = self.idm_acceleration(self, old_preceding)
        self_acc_new = self.idm_acceleration(self, new_preceding)
        
        if self_acc_new - self_acc_current > self.LANE_CHANGE_MIN_ACC_GAIN:
            return True
        return False