import time
import numpy as np
import pygame
import sys
from highway_env.envs.highway_env_custom import HighwayEnvCustom

# --- CẤU HÌNH DEBUG (ĐỒNG BỘ VỚI ENV "AMBULANCE" MỚI) ---
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
    
    # --- [1] LOGIC ANTI-CAMPING ---
    "stuck_penalty_coeff": 1.0, 
    
    # --- [2] ĐỘNG LỰC ---
    "overtake_reward": 5.0,        
    "better_lane_reward": 3.0,     
    "high_speed_reward": 1.5,      
    
    # --- [3] KỶ LUẬT (CHỐNG ZIGZAG) ---
    "lane_change_penalty": -2.0,   # [QUAN TRỌNG] Phạt nặng để chỉ rẽ khi Green Gap ngon
    "headway_penalty": -15.0,      
    "unsafe_lane_change_penalty": -10.0,
    "collision_reward": -30.0,
    
    # --- [4] CHIẾN THUẬT PHANH ---
    "low_speed_penalty": -0.2,     # Giữ thấp để dám phanh lách
    "overspeed_penalty": -1.0,
    
    "invalid_action_penalty": -3.0,
    "danger_penalty": -1.0,
    
    # TRAFFIC
    "vehicles_count": 50,
    "vehicles_density": 2.5,
    "normalize_reward": True,
    
    "render_agent": True,
    "manual_control": False, # Ta tự code manual control ở dưới
    "screen_width": 1000,
    "screen_height": 250,
}

def analyze_green_gap(env):
    """
    Hàm này tính toán lại Logic Green Gap để hiển thị lên Dashboard
    giúp bạn biết tại sao Agent chọn (hoặc không chọn) rẽ.
    """
    vehicle = env.vehicle
    current_lane = vehicle.lane_index[2]
    my_pos = vehicle.position[0]
    my_speed = vehicle.speed
    
    # 1. Tính d_req (Khoảng cách lách an toàn)
    d_req = 5.0 + (my_speed * 0.6)
    
    # 2. Tìm xe trước mặt (d_front)
    d_front = 999.0
    for v in env.road.vehicles:
        if v is vehicle: continue
        if v.lane_index[2] == current_lane:
            dist = v.position[0] - my_pos
            if 0 < dist < d_front: d_front = dist

    # Helper phân tích từng làn
    def analyze_lane(target_lane):
        if target_lane < 0 or target_lane >= 4:
            return "WALL", 0, 0, False

        d_side = 999.0
        blocked_parallel = False
        
        for v in env.road.vehicles:
            if v is vehicle: continue
            if v.lane_index[2] == target_lane:
                dist = v.position[0] - my_pos
                if -5.0 < dist < 5.0: blocked_parallel = True
                if dist > 0 and dist < d_side: d_side = dist
        
        if blocked_parallel:
            return "PARALLEL", d_side, -999, False # Kẹt cứng
            
        if d_side == 999.0:
            return "EMPTY", 999, 999, True # Trống trơn
            
        # Tính Green Gap
        if d_front != 999.0:
            green_gap = d_side - d_front
            is_passable = green_gap > d_req
            return "GAP", d_side, green_gap, is_passable
        else:
            # Không có xe trước mặt
            is_passable = d_side > d_req
            return "CLEAR", d_side, d_side, is_passable

    left_info = analyze_lane(current_lane - 1)
    right_info = analyze_lane(current_lane + 1)
    
    return d_req, d_front, left_info, right_info

def print_dashboard(env, action, rewards_dict, total_reward):
    # --- 1. LẤY DATA ---
    speed_kmh = env.vehicle.speed * 3.6
    speed_ms = env.vehicle.speed
    lane = env.vehicle.lane_index[2]
    stuck_steps = getattr(env.unwrapped, 'stuck_steps', 0)
    
    # Phân tích Green Gap
    d_req, d_front, l_info, r_info = analyze_green_gap(env)
    
    # --- 2. HIỂN THỊ ---
    print("\033[H\033[J", end="") # Clear screen
    print("="*80)
    
    # HEAD UP DISPLAY
    p_col = "\033[92m" if stuck_steps < 60 else "\033[91m"
    print(f"⏱️  PATIENCE: {p_col}{stuck_steps:<3}\033[0m | 🏎️  SPEED: {speed_kmh:.0f} km/h | 🛣️  LANE: {lane}")
    print("-" * 80)
    
    # GREEN GAP VISUALIZER (Cái bạn cần nhất) 
    print(f"\033[1m📐 GREEN GAP ANALYSIS (Req: {d_req:.1f}m | Front: {d_front:.1f}m)\033[0m")
    
    def fmt_lane(name, info):
        status, side_dist, gap, passable = info
        
        if status == "WALL": return f"{name}: \033[90m❌ WALL\033[0m"
        if status == "PARALLEL": return f"{name}: \033[91m⛔ PARALLEL (Kẹt sườn)\033[0m"
        if status == "EMPTY": return f"{name}: \033[92m✅ EMPTY (Thoải mái)\033[0m"
        
        # Logic hiển thị Gap
        gap_col = "\033[96m" if passable else "\033[91m" # Xanh ngọc hoặc Đỏ
        icon = "✅" if passable else "⛔"
        
        if status == "GAP":
            return f"{name}: {icon} Side:{side_dist:.0f}m - Front:{d_front:.0f}m = {gap_col}Gap {gap:.1f}m\033[0m"
        else:
            return f"{name}: {icon} Dist:{side_dist:.0f}m"

    print(f"   {fmt_lane('LEFT ', l_info)}")
    print(f"   {fmt_lane('RIGHT', r_info)}")
    print("-" * 80)

    # REWARD TABLE
    def color(val):
        if val > 0.01: return f"\033[92m+{val:.2f}\033[0m"
        if val < -0.01: return f"\033[91m{val:.2f}\033[0m"
        return f"\033[90m{val:.2f}\033[0m"

    print(f"{'REWARD NAME':<25} | {'RAW':<6} | {'WGT':<4} | {'SCORE'}")
    
    for key, raw in rewards_dict.items():
        wgt = config.get(key, 0)
        score = raw * wgt
        if abs(score) > 0.01: # Chỉ hiện cái nào có điểm
            print(f"{key:<25} | {raw:<6.1f} | {wgt:<4.1f} | {color(score)}")
            
    print("="*80)
    t_col = "\033[1;92m" if total_reward > 0 else "\033[1;91m"
    print(f"💰 TOTAL REWARD: {t_col}{total_reward:.4f}\033[0m")

    # Gợi ý hành động
    if l_info[3] and action != 0: print("\033[93m💡 GỢI Ý: Làn Trái đang có Green Gap ngon! Rẽ trái ngay!\033[0m")
    elif r_info[3] and action != 2: print("\033[93m💡 GỢI Ý: Làn Phải đang có Green Gap ngon! Rẽ phải ngay!\033[0m")

def main():
    # Khởi tạo Pygame
    pygame.init()
    pygame.display.set_caption("Highway Env - Debug Mode")
    
    env = HighwayEnvCustom(config=config, render_mode="human")
    env.reset()
    
    print("Sử dụng các phím mũi tên để điều khiển xe Agent.")
    
    running = True
    while running:
        done = False
        obs, info = env.reset()
        
        while not done and running:
            # Manual Control
            action = 1 # IDLE
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_LEFT]: action = 0
            elif keys[pygame.K_RIGHT]: action = 2
            elif keys[pygame.K_UP]: action = 3   
            elif keys[pygame.K_DOWN]: action = 4 
            
            obs, reward, done, truncated, info = env.step(action)
            
            # Lấy reward dict
            rewards_dict = info.get("rewards", {})
            total = sum(rewards_dict.get(k, 0) * config.get(k, 0) for k in rewards_dict)
            
            print_dashboard(env, action, rewards_dict, total)
            env.render()
            
            # Xử lý sự kiện Pygame (QUAN TRỌNG: Chống treo)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False
            
            time.sleep(0.05) 
            
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()