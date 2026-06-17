import gymnasium as gym
import os
import shutil
import cv2
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from highway_env.envs.highway_env_custom import HighwayEnvCustom
from highway_env.wrappers.tuple_flatten import TupleFlattenWrapper

# --- CẤU HÌNH ---
MODEL_PATH = "./saved_models_aggressive_v2/best_model_highway.zip"
VIDEO_FOLDER = "./recorded_videos_traj/"
VIDEO_FPS = 30  

# ======================================================
# 1. VISUALIZATION WRAPPER (LỚP VẼ TRAJECTORY)
# ======================================================
class VisualTrajectoryWrapper(gym.Wrapper):
    """
    Wrapper này sẽ can thiệp vào hàm render() để vẽ mũi tên hướng đi
    trước khi RecordVideo lưu lại thành video.
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_action = 1 # Mặc định là đi thẳng
        
    def step(self, action):
        # Lưu lại hành động để vẽ mũi tên tương ứng
        self.last_action = action
        return self.env.step(action)

    def render(self):
        # 1. Lấy ảnh gốc từ môi trường
        frame = self.env.render()
        
        # 2. Xử lý vẽ đè lên ảnh (Overlay)
        frame = self.draw_trajectory_overlay(frame, self.last_action)
        
        return frame

    def draw_trajectory_overlay(self, frame, action):
        # Chuyển sang BGR để vẽ bằng OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        
        # Vị trí vẽ (Cao hơn xe một chút)
        hud_y = center_y - 60
        
        # Cấu hình nét vẽ
        turn_color = (0, 255, 255) # Vàng (Yellow)
        brake_color = (0, 0, 255)  # Đỏ (Red)
        line_type = cv2.LINE_AA
        thickness = 3

        # --- LOGIC VẼ MŨI TÊN ---
        if action == 0: # LEFT (LANE_LEFT)
            # Vẽ mũi tên hướng trái <<
            pt1 = (center_x - 30, hud_y)
            pt2 = (center_x - 60, hud_y)
            cv2.arrowedLine(frame, pt1, pt2, turn_color, thickness, tipLength=0.5, line_type=line_type)
            cv2.putText(frame, "L", (center_x - 80, hud_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, turn_color, 2, line_type)
            
        elif action == 2: # RIGHT (LANE_RIGHT)
            # Vẽ mũi tên hướng phải >>
            pt1 = (center_x + 30, hud_y)
            pt2 = (center_x + 60, hud_y)
            cv2.arrowedLine(frame, pt1, pt2, turn_color, thickness, tipLength=0.5, line_type=line_type)
            cv2.putText(frame, "R", (center_x + 70, hud_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, turn_color, 2, line_type)
            
        elif action == 4: # BRAKE (SLOWER)
            # Vẽ cảnh báo phanh
            cv2.circle(frame, (center_x, hud_y), 15, brake_color, 2, line_type)
            cv2.putText(frame, "!", (center_x - 5, hud_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, brake_color, 2, line_type)

        # Chuyển lại về RGB để Gym hiển thị đúng màu
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ======================================================
# 2. KHỞI TẠO MÔI TRƯỜNG
# ======================================================
def create_env_for_recording():
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
        "vehicles_count": 40,
        "vehicles_density": 2.5,
        "normalize_reward": True,
        
        # Đồng bộ FPS
        "simulation_frequency": VIDEO_FPS, 
        "policy_frequency": int(VIDEO_FPS / 15),
        
        "screen_width": 800,  
        "screen_height": 350, # Cao một chút để thấy mũi tên
        "centering_position": [0.3, 0.6], # Hạ thấp xe xuống để có chỗ vẽ bên trên
        "scaling": 5.5,
        
        # Bật hiển thị lịch sử đường đi (các chấm nhỏ)
        "show_trajectories": True, 
        
        "render_agent": True,
        "offscreen_rendering": True 
    }
    
    env = HighwayEnvCustom(config=config, render_mode="rgb_array")
    
    # Gán Metadata FPS để RecordVideo không bị lỗi
    env.metadata["render_fps"] = VIDEO_FPS 
    
    # WRAPPER 1: Flatten Observation (Để Model hiểu)
    env = TupleFlattenWrapper(env)
    
    # WRAPPER 2: Visual Trajectory (Để vẽ mũi tên) - QUAN TRỌNG
    env = VisualTrajectoryWrapper(env)
    
    return env

# ======================================================
# 3. MAIN
# ======================================================
def main():
    if os.path.exists(VIDEO_FOLDER):
        shutil.rmtree(VIDEO_FOLDER)
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    print(f"🎥 Khởi tạo môi trường với Trajectory Overlay...")
    
    # Tạo môi trường đã có VisualTrajectoryWrapper
    env = create_env_for_recording()
    
    # WRAPPER 3: RecordVideo (Lưu kết quả cuối cùng)
    env = RecordVideo(
        env, 
        video_folder=VIDEO_FOLDER,
        episode_trigger=lambda e: True
    )
    
    # Cần set wrapper cho env gốc để render đúng (fix lỗi highway-env)
    # Lưu ý: env.unwrapped lấy env lõi, env lúc này đang là RecordVideo(Trajectory(Flatten(Highway)))
    env.unwrapped.set_record_video_wrapper(env)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model: {MODEL_PATH}")
        return
    model = PPO.load(MODEL_PATH)

    print(f"🔴 Đang quay video tại: {VIDEO_FOLDER}")
    
    for episode in range(3):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        step = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            # RecordVideo tự động gọi render() -> gọi VisualTrajectoryWrapper.render()
        
        print(f"✅ Xong Episode {episode+1}: Reward={total_reward:.1f}")

    env.close()
    print(f"\n🎉 Hoàn tất! Video (có mũi tên hướng) đã lưu tại: {VIDEO_FOLDER}")

if __name__ == "__main__":
    main()