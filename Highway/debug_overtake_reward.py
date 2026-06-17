import time
import numpy as np
from highway_env.envs.highway_env_custom import HighwayEnvCustom
from highway_env.vehicle.behavior import IDMVehicle

# --- CONFIG PHẢI KHỚP VỚI FILE TRAINING/ENV ---
config = {
    "observation": {
        "type": "Tuple",
        "observation_configs": [
            { "type": "Sensor" },
            { 
                "type": "Kinematics", 
                "vehicles_count": 5, 
                "features": ["x", "y", "vx", "vy", "lane"], 
                "normalize": False, 
                "absolute": False 
            }
        ]
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0, 15, 25, 35, 45, 55]
    },
    "lanes_count": 4,
    "vehicles_count": 0, # Tự setup xe thủ công
    "duration": 40,
    
    # Các trọng số reward (để tính toán hiển thị nếu cần)
    "overtake_reward": 1.5,
    "better_lane_reward": 2.0, 
}

def setup_smart_overtake_scenario(env):
    """
    Tình huống: Ego đang ở Làn 2. Bị kẹt sau xe chậm.
    - Làn 1 (Bên trái): TRỐNG TRẢI (Gap = 200m).
    - Làn 3 (Bên phải): CÓ XE (Gap = 30m).
    => Agent NÊN rẽ trái (Làn 1). Nếu rẽ trái sẽ được thưởng 'better_lane_reward'.
    """
    env.road.vehicles = [] 

    # 1. EGO VEHICLE (Làn 2, Speed 25)
    lane_2 = env.road.network.get_lane(("0", "1", 2))
    ego = env.action_type.vehicle_class(
        env.road, 
        position=lane_2.position(0, 0),
        heading=lane_2.heading_at(0),
        speed=25
    )
    ego.MAX_SPEED = 60
    env.vehicle = ego
    env.road.vehicles.append(ego)
    env.controlled_vehicles = [ego]

    # 2. BLOCKER (Làn 2, Ngay trước mặt Ego, Speed 15)
    blocker = IDMVehicle(env.road, position=lane_2.position(30, 0), heading=0, speed=15)
    blocker.color = (200, 0, 0) # Đỏ
    env.road.vehicles.append(blocker)

    # 3. OBSTACLE RIGHT (Làn 3, Hơi vướng, Speed 20)
    # Vị trí 40m (gần hơn làn trái)
    obs_right = IDMVehicle(env.road, position=env.road.network.get_lane(("0", "1", 3)).position(40, 0), heading=0, speed=20)
    env.road.vehicles.append(obs_right)

    # 4. OBSTACLE LEFT (Làn 1, Rất xa, Speed 30) -> LÀN NGON
    # Vị trí 150m (Rất xa)
    obs_left = IDMVehicle(env.road, position=env.road.network.get_lane(("0", "1", 1)).position(150, 0), heading=0, speed=30)
    env.road.vehicles.append(obs_left)
    
    print("-" * 60)
    print("--- KỊCH BẢN: CHỌN LÀN THÔNG MINH ---")
    print("🚗 EGO (Lane 2) bị kẹt xe trước.")
    print("❌ Lane 3 (Phải): Có xe cách 40m -> Khá chật.")
    print("✅ Lane 1 (Trái): Xe cách tận 150m -> RẤT THOÁNG.")
    print("👉 MONG ĐỢI: Khi Agent rẽ Trái, Better Lane Reward phải > 0.")
    print("-" * 60)

def test_overtake_logic():
    env = HighwayEnvCustom(config=config, render_mode="human")
    env.reset()
    setup_smart_overtake_scenario(env)
    
    # Kịch bản Action: 
    # 0: LEFT, 1: IDLE, 2: RIGHT
    # 5 bước đầu đi thẳng, bước thứ 6 (index 5) rẽ trái
    actions_script = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1] 

    print("\nBẮT ĐẦU CHẠY...")
    
    for step, action_idx in enumerate(actions_script):
        # Bước simulation
        obs, reward, done, truncated, info = env.step(action_idx)
        rewards_dict = info.get("rewards", {})
        
        # Lấy thông tin Gap thủ công để verify (gọi hàm nội bộ của env)
        current_lane = env.vehicle.lane_index[2]
        gap_left = env._get_lane_distance(current_lane - 1)
        gap_right = env._get_lane_distance(current_lane + 1)
        
        # Format in ấn
        print("\033[H\033[J", end="") # Clear màn hình
        act_name = env.action_type.actions[action_idx]
        
        print(f"STEP {step} | ACTION: \033[93m{act_name}\033[0m")
        print(f"📍 Ego Lane: {current_lane}")
        print(f"👀 GAP CHECK:")
        print(f"   - Left Gap  (Lane {current_lane-1}): {gap_left:.1f} m")
        print(f"   - Right Gap (Lane {current_lane+1}): {gap_right:.1f} m")
        
        # Check Reward
        r_overtake = rewards_dict.get("overtake_reward", 0) * config["overtake_reward"]
        r_better = rewards_dict.get("better_lane_reward", 0) * config["better_lane_reward"]
        
        print(f"\n💎 REWARD NHẬN ĐƯỢC:")
        print(f"   - Overtake Bonus : {r_overtake:.2f}")
        print(f"   - Smart Choice   : {r_better:.2f}")
        
        if r_better > 0:
            print("\n🌟🌟🌟 THÀNH CÔNG! SMART CHOICE TRIGGERED! 🌟🌟🌟")
            print("Lý do: Gap Left > Gap Right và Agent đã chọn rẽ Trái.")
        
        env.render()
        
        # Chậm lại khi thực hiện hành động rẽ để quan sát
        if action_idx == 0: 
            time.sleep(1.0)
        else:
            time.sleep(0.2)

    env.close()

if __name__ == "__main__":
    test_overtake_logic()