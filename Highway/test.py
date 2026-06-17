import gymnasium as gym
import numpy as np
import time
import os
import sys
import pygame # [FIX] Import trực tiếp để xử lý sự kiện cửa sổ
from stable_baselines3 import PPO
from highway_env.wrappers.tuple_flatten import TupleFlattenWrapper
from highway_env.envs.highway_env_custom import HighwayEnvCustom

# Map tên hành động
ACTION_MAP = {
    0: "⬅️ L_CHG",
    1: "⏸️ IDLE",
    2: "➡️ R_CHG",
    3: "🚀 FAST",
    4: "🐢 SLOW"
}

def create_eval_env(render_mode="human"):
    """
    Tạo môi trường Test với Config tối ưu 'Ambulance Mode'
    """
    config = {
        "observation": {
            "type": "Tuple",
            "observation_configs": [
                { "type": "Kinematics", "vehicles_count": 10, "features": ["presence", "x", "y", "vx", "vy", "lane"], "normalize": True, "absolute": False, "see_behind": True },
                {"type": "Danger", "max_ttc": 2.5}
            ]
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 15, 25, 35, 45, 55] 
        },
        "lanes_count": 4,
        "duration": 60, 
        
        # --- CẤU HÌNH CHIẾN THUẬT ---
        "stuck_penalty_coeff": 1.0,   
        "overtake_reward": 5.0,       
        "better_lane_reward": 3.0,    
        "high_speed_reward": 1.5,     
        
        # KỶ LUẬT
        "lane_change_penalty": -2.0,  
        "headway_penalty": -15.0,     
        "unsafe_lane_change_penalty": -10.0, 
        "low_speed_penalty": -0.2,    
        
        "collision_reward": -30.0,    
        # [CẬP NHẬT] Đồng bộ với file train mới
        "vehicles_count": 50,         
        "vehicles_density": 2.5,      
        
        "overspeed_penalty": -1.0,
        "invalid_action_penalty": -3.0,
        "danger_penalty": -1.0,
        "normalize_reward": True, 
    }
    env = HighwayEnvCustom(config=config, render_mode=render_mode)
    env = TupleFlattenWrapper(env)
    return env

def get_front_gap(env, vehicle):
    """Tính khoảng cách xe gần nhất phía trước (cùng làn)"""
    dist = 999.0
    for v in env.road.vehicles:
        if v is vehicle: continue
        if v.lane_index[2] == vehicle.lane_index[2]:
            d = v.position[0] - vehicle.position[0]
            if 0 < d < dist:
                dist = d
    return dist

def check_lane_status(env, vehicle):
    """
    KIỂM TRA LÀN BẰNG LOGIC 'GREEN GAP' (SO LE)
    """
    current_lane = vehicle.lane_index[2]
    my_pos = vehicle.position[0]
    my_speed = vehicle.speed
    
    # 1. Tính d_req (Khoảng cách cần thiết để lách dựa trên tốc độ)
    # Hệ số 0.6s cho xe cứu thương (xe thường 2s)
    d_req = 5.0 + (my_speed * 0.6)
    
    # 2. Tìm d_front (Xe vật cản trước mặt) để làm mốc so sánh
    d_front = 999.0
    for v in env.road.vehicles:
        if v is vehicle: continue
        if v.lane_index[2] == current_lane:
            dist = v.position[0] - my_pos
            if 0 < dist < d_front: d_front = dist

    left_lane_idx = current_lane - 1
    right_lane_idx = current_lane + 1
    max_lanes = 4 

    def get_status(target_lane):
        if target_lane < 0 or target_lane >= max_lanes: 
            return "\033[90m❌ WALL\033[0m"
        
        # Tìm xe gần nhất ở làn mục tiêu (d_side)
        d_side = 999.0
        blocked_by_parallel = False
        
        for v in env.road.vehicles:
            if v is vehicle: continue
            if v.lane_index[2] == target_lane:
                dist = v.position[0] - my_pos
                # Check song song (-5m đến +5m)
                if -5.0 < dist < 5.0: blocked_by_parallel = True
                # Check xe phía trước
                if dist > 0 and dist < d_side: d_side = dist
        
        # --- LOGIC HIỂN THỊ ---
        if blocked_by_parallel:
            return "\033[91m⛔ PARALLEL\033[0m"
        
        # Nếu làn bên cạnh trống trơn
        if d_side == 999.0:
            return "\033[92m✅ EMPTY\033[0m"
            
        # LOGIC SO LE (GREEN GAP)
        if d_front != 999.0:
            green_gap = d_side - d_front
            
            # Logic xử lý số âm: Nếu Gap âm -> Xe bên cạnh gần hơn xe trước -> Không thể rẽ
            if green_gap < 0:
                return f"\033[91m⛔ BLOCKED\033[0m"

            # Nếu khoảng hở giữa 2 xe > khoảng cách yêu cầu -> THOÁNG
            if green_gap > d_req:
                # Màu xanh ngọc (Cyan) báo hiệu khe hở tốt
                return f"\033[96m✅ GAP {green_gap:.0f}m > {d_req:.0f}m\033[0m"
            else:
                # Màu đỏ báo hiệu khe quá hẹp
                return f"\033[91m⛔ GAP {green_gap:.0f}m < {d_req:.0f}m\033[0m"
        else:
            # Nếu không có xe trước mặt, so sánh d_side với d_req như bình thường
            if d_side > d_req: 
                return f"\033[92m✅ >{d_req:.0f}m\033[0m"
            else: 
                return f"\033[91m⛔ <{d_req:.0f}m\033[0m"

    l_stat = get_status(left_lane_idx)
    r_stat = get_status(right_lane_idx)
            
    return l_stat, r_stat

def evaluate():
    # --- ĐƯỜNG DẪN MODEL ---
    model_path = "./saved_models_aggressive_v2/best_model_highway.zip"
    
    print(f"🔍 Đang load model tại: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ KHÔNG TÌM THẤY MODEL! Hãy kiểm tra lại.")
        return

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"⚠️ Lỗi load model: {e}")
        return
    
    # [FIX] Khởi tạo pygame
    pygame.init()
    
    env = create_eval_env(render_mode="human")
    
    n_episodes = 5
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        print(f"\n{'='*40} BẮT ĐẦU EPISODE {ep+1} {'='*40}")
        # Header bảng
        print(f"{'STEP':<4} | {'ACTION':<10} | {'SPEED':<10} | {'L_SIDE (Green Gap)':<25} | {'ME':<3} | {'R_SIDE (Green Gap)':<25} | {'STATUS':<20}")
        print("-" * 120)
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            act_idx = int(action)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Lấy thông tin thật
            real_env = env.unwrapped
            ego_vehicle = real_env.vehicle
            
            real_speed_kmh = ego_vehicle.speed * 3.6 
            real_speed_ms = ego_vehicle.speed
            real_lane = ego_vehicle.lane_index[2]
            front_gap = get_front_gap(real_env, ego_vehicle)
            
            # Check sensor (Logic mới)
            l_stat, r_stat = check_lane_status(real_env, ego_vehicle)
            
            # --- Status Chung ---
            # Kẹt nếu: Gap < 60m VÀ Tốc độ < 126km/h (35m/s)
            is_impeded = (front_gap < 60.0) and (real_speed_ms < 35.0)
            
            gap_str = f"{front_gap:.1f}m" if front_gap < 100 else "Free"
            if is_impeded:
                status_str = f"\033[93m⚠️ STUCK ({gap_str})\033[0m"
            elif front_gap < 60.0 and real_speed_ms >= 35.0:
                status_str = f"\033[96m🚀 RACING\033[0m"
            else:
                status_str = f"\033[92m✅ CLEAR\033[0m"

            # Tô màu Action
            act_str = ACTION_MAP[act_idx]
            if act_idx == 0 or act_idx == 2: act_str = f"\033[1;93m{act_str}\033[0m"
            elif act_idx == 4: act_str = f"\033[91m{act_str}\033[0m"
            
            # Tô màu Speed
            speed_str = f"{real_speed_kmh:.0f} km/h"
            if real_speed_ms < 25.0: speed_str = f"\033[91m{speed_str}\033[0m"
            
            print(f"{step_count:03d}  | {act_str:<19} | {speed_str:<19} | {l_stat:<34} |  {real_lane}  | {r_stat:<34} | {status_str:<20}")
            
            env.render()
            
            # [FIX] Xử lý event pygame để không bị treo
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            time.sleep(0.05) 

        print(f"\n🛑 KẾT THÚC EPISODE {ep+1} - TỔNG ĐIỂM: {total_reward:.2f}")
        print("-" * 120)
            
    env.close()

if __name__ == "__main__":
    evaluate()