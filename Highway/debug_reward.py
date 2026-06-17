import time
import numpy as np
import pygame
from highway_env.envs.highway_env_custom import HighwayEnvCustom

# --- CẤU HÌNH DEBUG (ĐỒNG BỘ VỚI ENV MỚI) ---
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
    
    # --- [1] LOGIC ANTI-CAMPING & PATIENCE ---
    "stuck_penalty_coeff": 1.0,    # Kích hoạt phạt mũ
    
    # --- [2] ĐỘNG LỰC ---
    "overtake_reward": 5.0,        # Thưởng lớn khi vượt thành công
    "better_lane_reward": 3.0,     # Thưởng chọn làn thông minh
    "high_speed_reward": 1.5,      # Thưởng tốc độ cao (1.5)
    
    # --- [3] KỶ LUẬT ---
    "lane_change_penalty": -0.5,   
    "headway_penalty": -15.0,      
    "unsafe_lane_change_penalty": -10.0,
    "collision_reward": -30.0,
    
    # --- [4] CHIẾN THUẬT PHANH ---
    "low_speed_penalty": -0.2,     # [QUAN TRỌNG] Giảm phạt để dám phanh
    "overspeed_penalty": -1.0,
    
    "invalid_action_penalty": -3.0,
    "danger_penalty": -1.0,
    
    # TRAFFIC
    "vehicles_count": 50,
    "vehicles_density": 2.5,
    "normalize_reward": True,
    
    "render_agent": True,
    "manual_control": False, 
    "screen_width": 1000,
    "screen_height": 250,
}

def get_front_gap(env):
    """Tính khoảng cách xe gần nhất phía trước"""
    vehicle = env.vehicle
    dist = 999.0
    for v in env.road.vehicles:
        if v is vehicle: continue
        if v.lane_index[2] == vehicle.lane_index[2]: # Cùng làn
            d = v.position[0] - vehicle.position[0]
            if 0 < d < dist:
                dist = d
    return dist

def print_dashboard(env, action, rewards_dict, total_reward):
    # Helper tô màu
    def color(val, is_score=False):
        if is_score:
            if val > 0.001: return f"\033[92m+{val:.2f}\033[0m" # Xanh lá
            if val < -0.001: return f"\033[91m{val:.2f}\033[0m" # Đỏ
            return f"\033[90m{val:.2f}\033[0m" # Xám
        return f"{val:.2f}"

    # Lấy thông tin xe
    speed_kmh = env.vehicle.speed * 3.6
    speed_ms = env.vehicle.speed
    lane = env.vehicle.lane_index[2]
    
    # Lấy thông tin Stuck từ Env (dùng getattr để an toàn)
    stuck_steps = getattr(env.unwrapped, 'stuck_steps', 0)
    
    # Tính Gap & Trạng thái Camping
    front_gap = get_front_gap(env)
    
    # Logic Camping (Khớp với Env: Gap < 60m VÀ Speed < 25m/s)
    is_impeded = (front_gap < 60.0) and (speed_ms < 25.0)
    
    # Xóa màn hình
    print("\033[H\033[J", end="") 
    
    print("="*60)
    # Hiển thị Thanh Kiên Nhẫn
    patience_color = "\033[92m" # Xanh
    if stuck_steps > 60: patience_color = "\033[93m" # Vàng
    if stuck_steps > 150: patience_color = "\033[91m" # Đỏ
    
    print(f"⏱️   PATIENCE METER: {patience_color}{stuck_steps}\033[0m steps (Boiling Frog)")
    print(f"🏎️   Speed: {speed_kmh:.0f} km/h ({speed_ms:.1f} m/s) | Lane: {lane}")
    print(f"📏  Front Gap: {front_gap:.1f}m")
    
    # Hiển thị Trạng thái Chiến thuật
    if is_impeded:
        status = "\033[1;93m⚠️  BỊ KẸT (IMPEDED)\033[0m"
        hint = "-> Đang bị phạt nhẹ... Hãy tìm đường RẼ hoặc PHANH để lách!"
    elif front_gap < 60.0 and speed_ms >= 25.0:
        status = "\033[96m✅ BÁM ĐUÔI TỐC ĐỘ CAO\033[0m"
        hint = "-> Tốt! Giữ tốc độ này để không bị tính là kẹt."
    else:
        status = "\033[92m🌟 ĐƯỜNG THOÁNG (Free)\033[0m"
        hint = "-> Tăng tốc tối đa (High Speed Reward)!"

    print(f"Status: {status}")
    print(f"Hint:   {hint}")
    print("-" * 60)
    
    print(f"{'CATEGORY':<25} | {'RAW':<6} | {'WEIGHT':<6} | {'SCORE':<10}")
    print("-" * 60)

    def print_row(name, key):
        raw = rewards_dict.get(key, 0.0)
        weight = config.get(key, 0.0)
        score = raw * weight
        score_str = color(score, True)
        
        # Highlight các chỉ số quan trọng
        if key == "stuck_penalty" and score < -0.01:
             score_str = f"\033[93m{score:.2f}\033[0m" # Vàng
        if key == "stuck_penalty" and score < -2.0:
             score_str = f"\033[1;91m{score:.2f} !!!\033[0m" # Đỏ rực
        
        if key == "overtake_reward" and score > 0:
             score_str = f"\033[1;92m+{score:.2f} (WIN)\033[0m" # Xanh đậm
             
        if key == "low_speed_penalty" and raw > 0 and weight > -0.5:
             score_str = f"\033[96m{score:.2f} (BRAKE)\033[0m" # Xanh dương (Phanh chấp nhận được)

        print(f"{name:<25} | {raw:<6.2f} | {weight:<6.1f} | {score_str}")
        return score

    # --- NHÓM 1: PHẦN THƯỞNG ---
    print(f"\033[1m💰  GAINS (Động lực)\033[0m")
    t1 = print_row("Overtake (Vượt)", "overtake_reward")
    t2 = print_row("Smart Lane (Làn ngon)", "better_lane_reward")
    p1 = print_row("High Speed (Tốc độ)", "high_speed_reward")
    
    # --- NHÓM 2: CHI PHÍ THỜI GIAN ---
    print("-" * 60)
    print(f"\033[1m⏳  TIME PRESSURE (Áp lực)\033[0m")
    c_stuck = print_row("STUCK PENALTY (Mũ)", "stuck_penalty") 
    
    # --- NHÓM 3: CHI PHÍ VẬN HÀNH ---
    print("-" * 60)
    print(f"\033[1m⚙️  OPERATIONS (Vận hành)\033[0m")
    c_slow  = print_row("Braking Cost (Phanh)", "low_speed_penalty") 
    c_steer = print_row("Steering (Đánh lái)", "lane_change_penalty")
    
    # --- NHÓM 4: AN TOÀN ---
    print("-" * 60)
    print(f"\033[1m🛡️  SAFETY (An toàn)\033[0m")
    s1 = print_row("Collision (Va chạm)", "collision_reward")
    s2 = print_row("Unsafe Turn (Rẽ ẩu)", "unsafe_lane_change_penalty") 
    s3 = print_row("Headway (Bám đuôi)", "headway_penalty")

    print("="*60)
    total_color = "\033[1;92m" if total_reward > 0 else "\033[1;91m"
    print(f"💰  TOTAL REWARD: {total_color}{total_reward:.4f}\033[0m")
    
    # Cảnh báo cuối
    if t1 > 0:
        print(f"\n🎉 \033[1;92mVƯỢT XE THÀNH CÔNG! (+{t1:.1f})\033[0m")
    elif c_stuck < -1.0:
        print(f"\n🔥 \033[91mCẢNH BÁO: ĐỨNG QUÁ LÂU! HÃY RẼ NGAY!\033[0m")

def main():
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
            
            # Logic phím bấm
            if keys[pygame.K_LEFT]: action = 0
            elif keys[pygame.K_RIGHT]: action = 2
            elif keys[pygame.K_UP]: action = 3   
            elif keys[pygame.K_DOWN]: action = 4 
            
            obs, reward, done, truncated, info = env.step(action)
            
            # Lấy dictionary rewards từ info (được trả về từ Env custom)
            rewards_dict = info.get("rewards", {})
            
            # Tính tổng reward theo trọng số Config Debug (để test nhanh)
            # Lưu ý: Lúc train thì PPO sẽ dùng trọng số trong Env, ở đây ta giả lập lại
            total = sum(rewards_dict.get(k, 0) * config.get(k, 0) for k in rewards_dict)
            
            print_dashboard(env, action, rewards_dict, total)
            env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False
            
            time.sleep(0.05) # Giảm tốc độ để kịp nhìn log
            
    env.close()

if __name__ == "__main__":
    main()