import gymnasium as gym
import torch
import numpy as np
import os
import time
import sys
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.callbacks import BaseCallback

from highway_env.wrappers.tuple_flatten import TupleFlattenWrapper
from highway_env.envs.highway_env_custom import HighwayEnvCustom

# ==========================================
# 1. CALLBACK: SMART STOP
# ==========================================
class SmartStopCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, reward_threshold: float, length_threshold: float, verbose=1):
        super(SmartStopCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.reward_threshold = reward_threshold
        self.length_threshold = length_threshold
        self.consecutive_wins = 0 
        
        self.save_path_best = os.path.join(save_path, "best_model_highway")
        if save_path is not None: os.makedirs(save_path, exist_ok=True)
        self.last_time = time.time()
        self.last_step = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            current_time = time.time()
            steps_done = self.num_timesteps - self.last_step
            dt = current_time - self.last_time
            fps = steps_done / dt if dt > 0 else 0
            self.last_time = current_time
            self.last_step = self.num_timesteps

            if len(self.model.ep_info_buffer) > 0:
                mean_reward = safe_mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_len = safe_mean([ep["l"] for ep in self.model.ep_info_buffer])
            else:
                mean_reward = 0.0
                mean_len = 0.0

            # Lưu model tốt nhất
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"\n⭐ [NEW RECORD] Reward: {mean_reward:.2f} (Len: {mean_len:.1f}s) -> Saving...")
                self.model.save(self.save_path_best)

            # Kiểm tra điều kiện dừng
            if mean_reward >= self.reward_threshold and mean_len >= self.length_threshold:
                self.consecutive_wins += 1
                print(f"\n💎 [ĐẠT CHUẨN] Lần thứ {self.consecutive_wins}/3. Reward: {mean_reward:.1f} | Len: {mean_len:.1f}s")
            else:
                self.consecutive_wins = 0

            if self.consecutive_wins >= 3:
                print(f"\n\n🏆 TRAINING THÀNH CÔNG! ĐÃ ĐẠT MỤC TIÊU!")
                return False 

            logger_vals = self.model.logger.name_to_value
            val_loss = logger_vals.get("train/value_loss", 0.0)
            entropy = -logger_vals.get("train/entropy_loss", 0.0)
            
            print(f"\n[CONTINUE STATUS] Steps: {self.num_timesteps:07d} | FPS: {int(fps)}")
            print(f"📊 MỤC TIÊU: Reward > {self.reward_threshold} & Time > {self.length_threshold}s")
            print(f"   | Hiện tại: Reward = {mean_reward:.1f} | Time = {mean_len:.1f}s")
            print(f"🧠 PPO INFO: Ent={entropy:.4f} | Loss={val_loss:.4f}")
            print(f"{'-'*60}")
        return True

# ==========================================
# 2. CẤU HÌNH MÔI TRƯỜNG (ANTI-ZIGZAG & ANTI-PARKING)
# ==========================================
def create_one_env():
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
        
        # [CẬP NHẬT] Thưởng tốc độ cao khi đã thỏa mãn (đường thoáng)
        "high_speed_reward": 2.0,     
        
        # KỶ LUẬT (CHỐNG ZIGZAG)
        # Phạt nặng để không rẽ lung tung khi đã Satisfaction
        "lane_change_penalty": -2.0,  
        
        "headway_penalty": -15.0,     
        "unsafe_lane_change_penalty": -10.0, 

        # PHANH CHIẾN THUẬT & CHỐNG ĐẬU XE
        # Phạt nặng (-1.0) kết hợp logic trong Env để trị bệnh đậu xe
        "low_speed_penalty": -1.0,    
        
        "collision_reward": -30.0,    
        "vehicles_count": 40,  # Giữ 40 xe để có khe hở       
        "vehicles_density": 2.5,      
        
        "overspeed_penalty": -1.0,
        "invalid_action_penalty": -3.0,
        "danger_penalty": -1.0,
        "normalize_reward": True, 
    }
    
    env = HighwayEnvCustom(config=config, render_mode=None)
    env = TupleFlattenWrapper(env)
    return env

# ==========================================
# 3. MAIN TRAINING LOOP (CONTINUE)
# ==========================================
if __name__ == "__main__":
    # --- CẤU HÌNH HỆ THỐNG ---
    num_cpu = 12  
    save_dir = "./saved_models_aggressive_v2/"
    log_dir = "./ppo_logs_aggressive_v2/"
    
    # Model cần load (Ưu tiên load Best Model)
    model_path = os.path.join(save_dir, "model_final_aggressive_v2_continued.zip")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy model tại: {model_path}")
        # Thử tìm model final
        model_path = os.path.join(save_dir, "model_final_aggressive_v2.zip")
        if not os.path.exists(model_path):
            print("❌ LỖI: Không tìm thấy bất kỳ model cũ nào để train tiếp!")
            sys.exit()
    
    print(f"♻️  LOADING MODEL: {model_path}")

    # Mục tiêu
    TARGET_LENGTH = 59.0   
    TARGET_REWARD = 1500.0  

    print(f"=== CONTINUE TRAINING: AGGRESSIVE DRIVING AGENT ===")
    print(f"Mục tiêu: Mean Reward >= {TARGET_REWARD} | Max Length >= {TARGET_LENGTH}")
    
    # Tạo môi trường song song (Giống hệt lúc train đầu)
    env = make_vec_env(create_one_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecMonitor(env, filename=log_dir + "monitor_continue.csv")

    # LOAD MODEL
    # Lưu ý: Chúng ta truyền env mới vào để đảm bảo logic reward mới được áp dụng
    model = PPO.load(
        model_path,
        env=env,
        tensorboard_log=log_dir,
        print_system_info=True,
        # Nếu muốn fine-tune (giảm learning rate hoặc entropy), chỉnh ở đây:
        # learning_rate=1e-4, 
        # ent_coef=0.01 
    )
    
    callback = SmartStopCallback(
        check_freq=1000,    
        save_path=save_dir,
        reward_threshold=TARGET_REWARD,
        length_threshold=TARGET_LENGTH
    )

    print("--- TIẾP TỤC TRAINING ---")
    try:
        # reset_num_timesteps=False: Để Tensorboard nối tiếp vào biểu đồ cũ
        model.learn(total_timesteps=5_000_000, callback=callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nNgười dùng dừng thủ công.")
    
    # Lưu model final mới
    final_path = os.path.join(save_dir, "model_final_aggressive_v2_continued")
    model.save(final_path)
    print(f"Đã lưu model tiếp tục tại: {final_path}.zip")
    env.close()