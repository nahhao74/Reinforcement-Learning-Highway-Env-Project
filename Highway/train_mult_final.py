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
            
            print(f"\n[TRAINING STATUS] Steps: {self.num_timesteps:07d} | FPS: {int(fps)}")
            print(f"📊 MỤC TIÊU: Reward > {self.reward_threshold} & Time > {self.length_threshold}s")
            print(f"   | Hiện tại: Reward = {mean_reward:.1f} | Time = {mean_len:.1f}s")
            print(f"🧠 PPO INFO: Ent={entropy:.4f} | Loss={val_loss:.4f}")
            print(f"{'-'*60}")
        return True

# ==========================================
# 2. CẤU HÌNH MÔI TRƯỜNG (AGGRESSIVE MODE)
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
        "high_speed_reward": 2.0,     
        
        # KỶ LUẬT (CHỐNG ZIGZAG)
        "lane_change_penalty": -2.0,  
        "headway_penalty": -15.0,     
        "unsafe_lane_change_penalty": -10.0, 

        # PHANH CHIẾN THUẬT
        "low_speed_penalty": -1.0,    
        
        "collision_reward": -30.0,    
        # [TINH CHỈNH] Giảm xuống 40 xe để tạo khe hở (vì traffic không rẽ)
        "vehicles_count": 40,         
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
# 3. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    # --- CẤU HÌNH HỆ THỐNG ---
    num_cpu = 12  
    save_dir = "./saved_models_aggressive_v2/"
    log_dir = "./ppo_logs_aggressive_v2/"
    
    # Reset Folder
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Mục tiêu
    TARGET_LENGTH = 59.0   
    # [TINH CHỈNH] Tăng Target Reward vì cơ chế Boost điểm (x1.5)
    TARGET_REWARD = 800.0  

    print(f"=== TRAINING: AGGRESSIVE DRIVING AGENT ===")
    print(f"Mục tiêu: Mean Reward >= {TARGET_REWARD} | Max Length >= {TARGET_LENGTH}")
    
    # Tạo môi trường song song
    env = make_vec_env(create_one_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv, seed=42)
    env = VecMonitor(env, filename=log_dir + "monitor.csv")

    # Khởi tạo PPO
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[512, 512, 256]), 
        learning_rate=3e-4, 
        n_steps=2048,       
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        # [TINH CHỈNH] Giảm Entropy một chút để hội tụ nhanh hơn vào chiến thuật tốt
        ent_coef=0.02,      
        verbose=0,
        tensorboard_log=log_dir
    )
    
    callback = SmartStopCallback(
        check_freq=1000,    
        save_path=save_dir,
        reward_threshold=TARGET_REWARD,
        length_threshold=TARGET_LENGTH
    )

    print("--- BẮT ĐẦU TRAINING ---")
    try:
        model.learn(total_timesteps=10_000_000, callback=callback)
    except KeyboardInterrupt:
        print("\nNgười dùng dừng thủ công.")
    
    final_path = os.path.join(save_dir, "model_final_aggressive_v2")
    model.save(final_path)
    print(f"Đã lưu model cuối cùng tại: {final_path}.zip")
    env.close()