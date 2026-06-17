from __future__ import annotations
from collections import OrderedDict
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from gymnasium import spaces
from highway_env import utils
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

class ObservationType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle

class KinematicObservation(ObservationType):
    """
    Quan sát dạng vector tọa độ (Nhanh nhất cho training).
    Trả về danh sách V xe gần nhất: [x, y, vx, vy, lane_id]
    """
    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy", "lane"] 

    # --- ĐÃ XÓA THAM SỐ see_behind TRONG __INIT__ ---
    def __init__(self, env, features=None, vehicles_count=5, absolute=False, normalize=True, **kwargs):
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        # self.see_behind = see_behind  <-- ĐÃ XÓA DÒNG NÀY
        self.absolute = absolute
        self.normalize_obs = normalize
        
        # Dải giá trị để normalize
        self.features_range = {
            "x": [-100, 100], 
            "y": [-100, 100], 
            "vx": [-60, 60], 
            "vy": [-60, 60],
            "lane": [0, 4]
        }

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-1, high=1, dtype=np.float32)

    def normalize(self, df):
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road: 
            return np.zeros(self.space().shape)
        
        ego = self.observer_vehicle
        
        
        close_vehicles = self.env.road.close_objects_to(
            ego, self.env.PERCEPTION_DISTANCE, count=self.vehicles_count - 1, see_behind=False
        )
        all_vehicles = [ego] + close_vehicles
        
        data = []
        for v in all_vehicles:
            if self.absolute:
                v_dict = v.to_dict(origin_vehicle=None)
            else:
                v_dict = v.to_dict(origin_vehicle=ego)
            
            # Thêm thông tin lane index (0, 1, 2...)
            try:
                v_dict["lane"] = v.lane_index[2] 
            except:
                v_dict["lane"] = 0 
            data.append(v_dict)

        df = pd.DataFrame.from_records(data)
        
        # Tính khoảng cách để sort (ưu tiên xe gần)
        if self.absolute:
            ego_dict = df.iloc[0] 
            dx = df["x"] - ego_dict["x"]
            dy = df["y"] - ego_dict["y"]
            df["distance"] = dx**2 + dy**2
        elif "x" in df.columns and "y" in df.columns:
            df["distance"] = df["x"]**2 + df["y"]**2
        
        if "distance" in df.columns:
            df = df.sort_values(by="distance")
            df = df.drop(columns=["distance"])
            
        df = df.head(self.vehicles_count)

        # Padding nếu không đủ xe
        df_final = pd.DataFrame(0, index=df.index, columns=self.features)
        for f in self.features:
            if f in df.columns:
                df_final[f] = df[f]
        
        if "presence" in self.features:
            df_final["presence"] = 1

        if self.normalize_obs:
            df_final = self.normalize(df_final)
        
        current_count = df_final.shape[0]
        if current_count < self.vehicles_count:
            padding = np.zeros((self.vehicles_count - current_count, len(self.features))) 
            df_padding = pd.DataFrame(padding, columns=self.features)
            df_final = pd.concat([df_final, df_padding], ignore_index=True)
            
        return df_final.values.astype(np.float32)

class SensorObservation(ObservationType):
    """
    Mô phỏng 5 Sensor + TTC Logic.
    Output vector size (7,):
    1. Front Dist (Normalized)      [0: Gần, 1: Xa]
    2. Front TTC (Normalized)       [0: Sắp đâm, 1: An toàn]
    3. Front-Left Dist              [0: Gần, 1: Xa]
    4. Front-Right Dist             [0: Gần, 1: Xa]
    5. Side-Left Presence           [1: Có xe, 0: Trống]
    6. Side-Right Presence          [1: Có xe, 0: Trống]
    7. Ego Speed (Normalized)       [Vận tốc bản thân]
    """
    def __init__(self, env, max_ttc=5.0, **kwargs):
        super().__init__(env)
        self.sensor_range = 100.0 # Tầm nhìn 100m
        self.ttc_horizon = max_ttc 

    def space(self):
        return spaces.Box(shape=(7,), low=-1.0, high=1.0, dtype=np.float32)

    def observe(self):
        ego = self.observer_vehicle
        if not ego or not self.env.road:
            return np.zeros(7, dtype=np.float32)

        # Mặc định an toàn (1.0) hoặc không có xe (0.0 cho presence)
        s_front_dist = 1.0
        s_front_ttc = 1.0   
        s_front_left = 1.0
        s_front_right = 1.0
        s_side_left = 0.0   
        s_side_right = 0.0  

        # Lấy các xe xung quanh
        vehicles = self.env.road.close_objects_to(ego, self.sensor_range)

        for v in vehicles:
            if v is ego: continue

            # Tính toạ độ tương đối
            long_dist = v.position[0] - ego.position[0] # Khoảng cách dọc
            lat_dist = v.position[1] - ego.position[1]  # Khoảng cách ngang
            
            # Check làn
            is_same_lane = abs(lat_dist) < 2.0
            is_left_lane = -6.0 < lat_dist < -2.0
            is_right_lane = 2.0 < lat_dist < 6.0

            # 1. SENSOR TRƯỚC (Front) & TTC
            if is_same_lane and long_dist > 0: 
                # Distance
                norm_dist = np.clip(long_dist / self.sensor_range, 0, 1)
                if norm_dist < s_front_dist:
                    s_front_dist = norm_dist
                    
                    # --- TÍNH TTC ---
                    rel_speed = ego.speed - v.speed # Dương = Đang lao tới gần
                    if rel_speed > 0.1: # Chỉ tính TTC nếu đang lao tới
                        ttc_val = max(0, long_dist - ego.LENGTH) / rel_speed
                        s_front_ttc = np.clip(ttc_val / self.ttc_horizon, 0.0, 1.0)
                    else:
                        s_front_ttc = 1.0

            # 2. SENSOR TRƯỚC-TRÁI
            if is_left_lane and long_dist > 0:
                norm_dist = np.clip(long_dist / self.sensor_range, 0, 1)
                if norm_dist < s_front_left:
                    s_front_left = norm_dist

            # 3. SENSOR TRƯỚC-PHẢI
            if is_right_lane and long_dist > 0:
                norm_dist = np.clip(long_dist / self.sensor_range, 0, 1)
                if norm_dist < s_front_right:
                    s_front_right = norm_dist

            # 4. SENSOR HÔNG (Check điểm mù)
            if is_left_lane and abs(long_dist) < ego.LENGTH * 1.5:
                s_side_left = 1.0
            if is_right_lane and abs(long_dist) < ego.LENGTH * 1.5:
                s_side_right = 1.0

        # Ego Speed normalized
        ego_speed_norm = ego.speed / 60.0

        return np.array([
            s_front_dist,       # Khoảng cách xe trước
            s_front_ttc,        # TTC xe trước (quan trọng nhất)
            s_front_left,       
            s_front_right,      
            s_side_left,        
            s_side_right,       
            ego_speed_norm      
        ], dtype=np.float32)

class TupleObservation(ObservationType):
    def __init__(self, env, observation_configs, **kwargs):
        super().__init__(env)
        self.observation_types = [observation_factory(env, config) for config in observation_configs]

    def space(self):
        return spaces.Tuple([obs.space() for obs in self.observation_types])

    def observe(self):
        return tuple(obs.observe() for obs in self.observation_types)

def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    t = config.get("type", "")
    if t == "Kinematics": return KinematicObservation(env, **config)
    
    if t == "Sensor": return SensorObservation(env, **config)
    if t == "Danger": return SensorObservation(env, **config) 
    
    if t == "Tuple": return TupleObservation(env, **config)
    raise ValueError(f"Unknown observation type: {t}")