import time
import numpy as np
import pygame
import gymnasium as gym
from highway_env.envs.highway_env_custom import HighwayEnvCustom

# --- CẤU HÌNH ---
config = {
    "observation": {
        "type": "Tuple",
        "observation_configs": [
            {
                "type": "Kinematics", 
                "vehicles_count": 5, 
                "features": ["presence", "x", "y", "vx", "vy", "lane"], 
                "absolute": False, 
                "normalize": True
            },
            {
                "type": "Sensor", 
                "max_ttc": 2.5
            } 
        ]
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0, 15, 25, 35, 45, 55] 
    },
    "lanes_count": 4,
    "duration": 60,
    
    # --- TRAFFIC ĐÔNG ĐÚC ---
    "vehicles_count": 15,  
    "vehicles_density": 1.2, 

    "show_trajectories": True,
    "render_agent": True,
    
    # [QUAN TRỌNG] manual_control = False để code Python được quyền gửi action
    "manual_control": False 
}

def debug_manual_driving():
    # Khởi tạo môi trường
    env = HighwayEnvCustom(config=config, render_mode="human")
    env.reset()
    
    print("=== CHẾ ĐỘ TỰ LÁI & DEBUG ===")
    print("🎮 ĐIỀU KHIỂN: [⬅️ Trái] [➡️ Phải] [⬆️ Tăng tốc] [⬇️ Giảm tốc]")
    print("ℹ️  Màn hình sẽ hiển thị thông số Sensor thời gian thực.")

    running = True
    while running:
        done = False
        obs, info = env.reset()
        
        while not done and running:
            # --- 1. XỬ LÝ PHÍM BẤM ---
            action = 1 # Mặc định là IDLE (Giữ nguyên)
            
            # Lấy trạng thái bàn phím
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_LEFT]:
                action = 0 # LANE_LEFT
            elif keys[pygame.K_RIGHT]:
                action = 2 # LANE_RIGHT
            elif keys[pygame.K_UP]:
                action = 3 # FASTER
            elif keys[pygame.K_DOWN]:
                action = 4 # SLOWER
            
            # --- 2. GỬI ACTION VÀO GAME ---
            # obs là Tuple: (Kinematics, Sensor)
            obs, reward, done, truncated, info = env.step(action)
            
            sensor_obs = obs[1] # Lấy phần Sensor
            
            # --- 3. HIỂN THỊ DEBUG ---
            print("\033[H\033[J", end="") # Clear console
            
            # Hiển thị Action đang bấm
            act_name = env.action_type.actions[action]
            color_code = "\033[92m" if action != 1 else "\033[90m"
            print(f"🎮 ACTION: {color_code}{act_name}\033[0m")
            
            # Thông số xe
            speed_kmh = env.vehicle.speed * 3.6
            print(f"🚀 TỐC ĐỘ: {speed_kmh:.0f} km/h (Target: {env.vehicle.target_speed*3.6:.0f})")
            
            print("-" * 40)
            print("📡 SENSOR OBSERVATION (Giá trị Neural Network):")
            
            # Màu sắc cảnh báo
            def color_val(val, danger_threshold, inverse=False):
                is_danger = val < danger_threshold if not inverse else val > danger_threshold
                return "\033[91m" + f"{val:.4f}" + "\033[0m" if is_danger else f"{val:.4f}"

            # Giải mã Vector Sensor (7 giá trị)
            # [0:FrontDist, 1:FrontTTC, 2:FrontLeft, 3:FrontRight, 4:SideLeft, 5:SideRight, 6:Speed]
            
            print(f"   [0] Khoảng cách trước : {color_val(sensor_obs[0], 0.3)}  (1.0=Xa, 0.0=Gần)")
            print(f"   [1] TTC (Va chạm)     : {color_val(sensor_obs[1], 0.5)}  (1.0=An toàn)")
            
            print(f"   [2] Góc Trái Trước    : {sensor_obs[2]:.4f}")
            print(f"   [3] Góc Phải Trước    : {sensor_obs[3]:.4f}")
            
            # Check điểm mù (Side Sensors)
            s_left = sensor_obs[4]
            s_right = sensor_obs[5]
            str_sl = "🛑 CÓ XE" if s_left > 0.9 else "✅ Trống"
            str_sr = "🛑 CÓ XE" if s_right > 0.9 else "✅ Trống"
            
            print(f"   [4] Hông Trái (Blind) : {str_sl}")
            print(f"   [5] Hông Phải (Blind) : {str_sr}")

            print("-" * 40)
            print("Nhấn ESC để thoát.")

            env.render()
            
            # Xử lý sự kiện thoát cửa sổ
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            
            # Giới hạn tốc độ vòng lặp để dễ lái (20 FPS)
            time.sleep(0.05) 
            
    env.close()

if __name__ == "__main__":
    debug_manual_driving()