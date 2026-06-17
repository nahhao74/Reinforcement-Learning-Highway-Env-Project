from __future__ import annotations
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle

class HighwayEnvCustom(AbstractEnv):
    """
    Môi trường Highway Custom:
    - Tích hợp Anti-Camping (Stuck Penalty).
    - Tích hợp Tactical Braking (Phanh chiến thuật).
    - Logic vượt xe thực dụng (Availability Masking).
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Tuple",
                "observation_configs": [
                    {
                        "type": "Kinematics", 
                        "vehicles_count": 10,
                        "features": ["presence", "x", "y", "vx", "vy", "lane"], 
                        "normalize": True, 
                        "absolute": False 
                    },
                    {"type": "Danger", "max_ttc": 2.5}
                ]
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 15, 25, 35, 45, 55] 
            },
            "lanes_count": 4,
            "duration": 60,
            "simulation_frequency": 30,
            "policy_frequency": 15,
            
            # --- CẤU HÌNH MẶC ĐỊNH 
            "collision_reward": -20.0,
            "high_speed_reward": 1.5,     
            "low_speed_penalty": -0.2,    
            "overspeed_penalty": -0.5,
            "invalid_action_penalty": -3.0,
            
            "overtake_reward": 5.0,       
            "better_lane_reward": 3.0,    
            "danger_penalty": -1.0,       
            "headway_penalty": -0.5,      
            "unsafe_lane_change_penalty": -10.0, 
            "lane_change_penalty": -0.5, 
            "stuck_penalty_coeff": 1.0,

            "vehicles_count": 15,
            "vehicles_density": 1.0,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        # Khởi tạo bộ đếm stuck
        self.stuck_steps = 0 

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=50), 
            np_random=self.np_random, 
            record_history=False 
        )

    def _create_vehicles(self) -> None:
        self.controlled_vehicles = []
        
        # Xe Agent
        lane = self.road.network.get_lane(("0", "1", 1)) 
        vehicle = self.action_type.vehicle_class(
            road=self.road,
            position=lane.position(0, 0),
            heading=lane.heading_at(0),
            speed=25 
        )
        vehicle.MAX_SPEED = 60 
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        # Xe Traffic
        total_vehicles = self.config["vehicles_count"]
        lanes_count = self.config["lanes_count"]
        vehicles_per_lane = total_vehicles // lanes_count 
        if vehicles_per_lane * lanes_count < total_vehicles: vehicles_per_lane += 1

        for lane_idx in range(lanes_count):
            for i in range(vehicles_per_lane):
                speed_variation = self.np_random.uniform(-6, 6)
                base_traffic_speed = 20
                random_speed = base_traffic_speed + speed_variation
                
                v = IDMVehicle.create_random(
                    self.road, 
                    speed=random_speed,
                    lane_id=lane_idx, 
                    spacing=1 / self.config["vehicles_density"]
                )
                
                stagger_offset = (lane_idx % 2) * 40 
                base_x = 60 + (i * 90) + stagger_offset
                noise_x = self.np_random.uniform(-30, 30)
                
                lane_obj = self.road.network.get_lane(("0", "1", lane_idx))
                v.position = lane_obj.position(base_x + noise_x, 0)
                v.MAX_SPEED = 50 
                
                safe_spawn = True
                for existing_v in self.road.vehicles:
                    if np.linalg.norm(existing_v.position - v.position) < 15:
                        safe_spawn = False
                        break
                if safe_spawn:
                    self.road.vehicles.append(v)

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * p for name, p in rewards.items())
        return reward

    def _get_lane_distance(self, lane_index):
        if lane_index < 0 or lane_index >= self.config["lanes_count"]: return -1.0 
        closest_dist = 200.0 
        for v in self.road.vehicles:
            if v is self.vehicle: continue
            try: v_lane = v.lane_index[2]
            except: continue
            if v_lane == lane_index:
                dist = v.position[0] - self.vehicle.position[0]
                if 0 < dist < closest_dist:
                    closest_dist = dist     
        return closest_dist
    
    def _is_lane_occupied(self, lane_index):
        if lane_index < 0 or lane_index >= self.config["lanes_count"]: return True
        
        # 1. Thông số cơ bản
        current_lane = self.vehicle.lane_index[2]
        ego_x = self.vehicle.position[0]
        current_speed = self.vehicle.speed
        
        # 2. Tính khoảng cách cần thiết để lách (d_req)
        d_req = 5.0 + (current_speed * 0.6) 
        
        # 3. Tìm khoảng cách xe gần nhất phía trước (d_front) ở làn HIỆN TẠI
        d_front = 999.0
        for v in self.road.vehicles:
            if v is self.vehicle: continue
            if v.lane_index[2] == current_lane:
                dist = v.position[0] - ego_x
                if 0 < dist < d_front:
                    d_front = dist
                    
        # 4. Quét làn MỤC TIÊU (Target Lane)
        for v in self.road.vehicles:
            if v is self.vehicle: continue
            if v.lane_index[2] == lane_index:
                d_side = v.position[0] - ego_x
                
                # --- TRƯỜNG HỢP A: Kẹt sườn (Song song) ---
                # Vùng cấm tuyệt đối (-5m đến +5m)
                if -5.0 < d_side < 5.0:
                    return True 
                
                # --- TRƯỜNG HỢP B: Xe bên cạnh ở phía trước ---
                if d_side > 5.0:
                    # Tính "Khoảng cách xanh lá" (Green Gap)
                    # Là khoảng trống giữa đuôi xe bên cạnh và đuôi xe trước mặt
                    # Lưu ý: Nếu không có xe trước mặt (d_front=999), green_gap là vô tận -> Thoáng
                    
                    if d_front != 999.0:
                        green_gap = d_side - d_front
                        
                        # LOGIC QUAN TRỌNG CỦA BẠN:
                        # Nếu Green Gap đủ lớn -> Coi như thoáng (Bỏ qua xe d_side)
                        if green_gap > d_req:
                            # Agent hiểu: "Tuy có xe bên cạnh, nhưng nó ở tít xa so với xe trước mặt mình.
                            # Khoảng trống giữa 2 xe đó đủ để mình lách lên và chui vào."
                            continue 
                    
                    # Nếu Green Gap nhỏ (2 xe đi gần ngang nhau) -> So sánh d_side với d_req như bình thường
                    # (Tức là nếu không chui lọt giữa 2 xe, thì phải đợi xe bên cạnh đi khuất hẳn)
                    if d_side < d_req + 10.0: # Cộng thêm buffer an toàn nếu không dùng chiến thuật chui giữa
                         return True

        return False

    def _rewards(self, action: int) -> dict[str, float]:
        speed = self.vehicle.speed
        
        # ... (Phần 1. TỐC ĐỘ giữ nguyên) ...
        # Copy lại logic Speed cũ của bạn vào đây
        high_speed = 0.0
        low_speed_pen = 0.0
        overspeed_pen = 0.0
        MIN_GOOD = 25.0 
        MAX_GOOD = 55.0 

        if speed < MIN_GOOD:
            diff = MIN_GOOD - speed
            low_speed_pen = 1.0 - np.exp(-(diff**2) / 50.0) 
            # Logic phạt đi chậm
            if speed < 15.0: low_speed_pen += (15.0 - speed) / 3.0 
        elif speed > MAX_GOOD:
            diff = speed - MAX_GOOD
            overspeed_pen = 1.0 - np.exp(-(diff**2) / 50.0)
        else:
            high_speed = utils.lmap(speed, [MIN_GOOD, MAX_GOOD], [0.5, 1.0])

        # ... (Phần 2. HÀNH ĐỘNG giữ nguyên) ...
        invalid_action_pen = 0.0
        unsafe_lane_change_pen = 0.0
        lane_change_pen = 0.0 
        stability_bonus = 0.0 
        
        try: current_lane = self.vehicle.lane_index[2]
        except: current_lane = 0
        
        if action == 0 and current_lane == 0: invalid_action_pen = 1.0 
        elif action == 2 and current_lane == self.config["lanes_count"] - 1: invalid_action_pen = 1.0 
        
        if action == 0: 
            if self._is_lane_occupied(current_lane - 1): unsafe_lane_change_pen = 1.0
        elif action == 2: 
            if self._is_lane_occupied(current_lane + 1): unsafe_lane_change_pen = 1.0

        if action == 0 or action == 2:
            lane_change_pen = 1.0
        elif action == 1 and speed > 30.0:
            stability_bonus = 0.2

        # --- 3. LOGIC STUCK (ANTI-CAMPING) ---
        gap_current = self._get_lane_distance(current_lane)
        is_impeded = (0 < gap_current < 60.0) and (speed < 35.0)
        
        # [QUAN TRỌNG] Logic "BIẾT ĐỦ" (Satisfaction)
        # Nếu đường trước mặt thoáng hơn 80m -> Agent đang ở trạng thái sung sướng.
        # Lúc này KHÔNG CẦN tìm làn tốt hơn nữa.
        is_satisfied = gap_current > 80.0

        if is_impeded: self.stuck_steps += 1
        else: self.stuck_steps = 0 
            
        stuck_pen = 0.0
        if self.stuck_steps > 0:
            stuck_pen = 0.05 * (np.exp(self.stuck_steps / 60.0) - 1.0)
            stuck_pen = min(stuck_pen, 5.0)

        # --- 4. VƯỢT XE: GREEN GAP ---
        overtake_bonus = 0.0
        better_lane_bonus = 0.0
        
        d_req = 5.0 + (speed * 0.6)
        
        gap_left_front = self._get_lane_distance(current_lane - 1)
        gap_right_front = self._get_lane_distance(current_lane + 1)
        
        can_turn_left = not self._is_lane_occupied(current_lane - 1)
        can_turn_right = not self._is_lane_occupied(current_lane + 1)

        def calculate_green_gap(side_gap, front_gap):
            if front_gap >= 60.0: return side_gap
            diff = side_gap - front_gap
            if diff < 0: return -999.0
            return diff

        green_gap_left = calculate_green_gap(gap_left_front, gap_current)
        green_gap_right = calculate_green_gap(gap_right_front, gap_current)

        val_left = green_gap_left if can_turn_left and current_lane > 0 else -999.0
        val_right = green_gap_right if can_turn_right and current_lane < self.config["lanes_count"] - 1 else -999.0
        
        best_gap = max(val_left, val_right)

        # Chỉ xét thưởng vượt xe nếu:
        # 1. Không rẽ ẩu
        # 2. VÀ CHƯA THỎA MÃN (is_satisfied = False) -> Nếu đường thoáng rồi thì đừng rẽ nữa!
        if unsafe_lane_change_pen == 0 and not is_satisfied:
            
            # --- RẼ TRÁI ---
            if action == 0 and can_turn_left:
                if green_gap_left > d_req:
                    if val_left >= best_gap:
                        overtake_bonus = 1.0
                        better_lane_bonus = 1.0
                    else:
                        overtake_bonus = 0.5 

            # --- RẼ PHẢI ---
            elif action == 2 and can_turn_right:
                if green_gap_right > d_req:
                    if val_right >= best_gap:
                        overtake_bonus = 1.0
                        better_lane_bonus = 1.0
                    else:
                        overtake_bonus = 0.5

        # 5. DYNAMIC BOOST
        if is_impeded:
            if overtake_bonus > 0:
                overtake_bonus *= 1.5 
                better_lane_bonus *= 1.5
                lane_change_pen *= 0.5

        # ... (Phần trả về return giữ nguyên) ...
        # (Copy phần return cũ vào đây)
        # --- 6. AN TOÀN ---
        danger_pen = 0.0
        headway_pen = 0.0
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        
        if front_vehicle:
            dist = self.vehicle.lane_distance_to(front_vehicle) - self.vehicle.LENGTH
            rel_speed = self.vehicle.speed - front_vehicle.speed 
            safe_distance = 5.0 + self.vehicle.speed * 0.1
            if dist < safe_distance:
                headway_pen = np.clip(1.0 - (dist / safe_distance), 0.0, 1.0)
            if rel_speed > 0 and dist < 100:
                ttc = dist / rel_speed
                if ttc < 1.0: danger_pen = np.clip(1.0 - (ttc / 1.0), 0.0, 1.0)

        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": float(high_speed) + float(stability_bonus), 
            "low_speed_penalty": float(low_speed_pen),
            "overspeed_penalty": float(overspeed_pen),
            "invalid_action_penalty": float(invalid_action_pen),
            "overtake_reward": float(overtake_bonus),
            "better_lane_reward": float(better_lane_bonus),
            "danger_penalty": float(danger_pen),
            "headway_penalty": float(headway_pen),
            "unsafe_lane_change_penalty": float(unsafe_lane_change_pen),
            "lane_change_penalty": float(lane_change_pen),
            "stuck_penalty": -float(stuck_pen) 
        }
    
    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]
    
    def _compute_danger_metrics(self) -> dict: return {}