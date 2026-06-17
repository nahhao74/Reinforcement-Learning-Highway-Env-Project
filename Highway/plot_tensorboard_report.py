import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

# --- CẤU HÌNH ---
LOG_DIR = "./ppo_logs_aggressive_v2/PPO_1"  # Folder chứa nhiều file events
OUTPUT_DIR = "./academic_plots_merged/" # Nơi lưu ảnh
SMOOTHING_WEIGHT = 0.95                # Độ mượt (0.0 - 0.99). 0.95 là rất mượt.

# Các Metrics cần vẽ
METRICS_TO_PLOT = {
    # 1. Quan trọng nhất
    "rollout/ep_rew_mean": ("Average Episode Reward", "reward_mean", "royalblue"),
    "rollout/ep_len_mean": ("Average Episode Length", "episode_length", "green"),
    
    # 2. Hiệu suất mạng
    "train/value_loss": ("Value Function Loss", "value_loss", "orange"),
    "train/policy_gradient_loss": ("Policy Gradient Loss", "policy_loss", "purple"),
    "train/entropy_loss": ("Entropy (Exploration)", "entropy", "red"),
    
    # 3. Kỹ thuật (để debug/giải thích trong báo cáo)
    "train/approx_kl": ("Approximate KL Divergence", "approx_kl", "gray"),
    "train/explained_variance": ("Explained Variance", "explained_variance", "teal"),
}

def get_all_event_files(log_dir):
    """Tìm tất cả file events trong folder"""
    files = glob.glob(os.path.join(log_dir, "*tfevents*"))
    # Sắp xếp theo thời gian tạo để load đúng thứ tự
    files.sort(key=os.path.getmtime)
    return files

def extract_merged_data(log_dir):
    """Đọc và gộp dữ liệu từ TẤT CẢ các file logs"""
    files = get_all_event_files(log_dir)
    if not files:
        raise ValueError(f"❌ Không tìm thấy file log nào trong {log_dir}")
    
    print(f"📂 Tìm thấy {len(files)} file logs. Đang tiến hành gộp...")
    
    # Dictionary chứa list các dataframe con
    merged_data_buffer = {} 
    
    for f_idx, file_path in enumerate(files):
        print(f"   [{f_idx+1}/{len(files)}] Đọc: {os.path.basename(file_path)}")
        try:
            # size_guidance=0 để load full lịch sử
            ea = EventAccumulator(file_path, size_guidance={'scalars': 0})
            ea.Reload()
            
            tags = ea.Tags()['scalars']
            
            for tag in tags:
                if tag not in METRICS_TO_PLOT: continue # Chỉ lấy cái cần vẽ cho nhẹ
                
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                df_chunk = pd.DataFrame({"step": steps, "value": values})
                
                if tag not in merged_data_buffer:
                    merged_data_buffer[tag] = []
                merged_data_buffer[tag].append(df_chunk)
                
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {os.path.basename(file_path)}: {e}")

    # Gộp các chunk lại thành DataFrame hoàn chỉnh
    final_data = {}
    for tag, chunks in merged_data_buffer.items():
        if chunks:
            # Concat và sort theo step
            full_df = pd.concat(chunks).sort_values("step").drop_duplicates(subset="step", keep="last")
            final_data[tag] = full_df
            
    return final_data

def smooth_data(values, weight=0.6):
    """Hàm làm mượt (Smoothing)"""
    last = values.iloc[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_single_metric(df, title, filename, color, save_dir):
    """Vẽ 1 biểu đồ chuẩn Academic"""
    # Setup style
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "serif" # Font kiểu báo cáo khoa học
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Vẽ Raw Data (Mờ nhạt) để thấy độ dao động thực tế
    ax.plot(df["step"], df["value"], alpha=0.15, color=color, linewidth=1, label="Raw Data")
    
    # 2. Vẽ Smoothed Data (Đậm) để thấy xu hướng
    df["smooth"] = smooth_data(df["value"], SMOOTHING_WEIGHT)
    ax.plot(df["step"], df["smooth"], alpha=1.0, color=color, linewidth=2.5, label=f"Trend (Smooth {SMOOTHING_WEIGHT})")
    
    # Trang trí trục và tiêu đề
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Training Timesteps", fontsize=14, fontweight='bold')
    ax.set_ylabel("Value", fontsize=14, fontweight='bold')
    
    # Lưới và Legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc="best", frameon=True, fontsize=12)
    
    # Format số bước chân (ví dụ 1000000 -> 1M)
    def human_format(num, pos):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G'][magnitude])
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(human_format))
    
    plt.tight_layout()
    
    # Lưu file
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu biểu đồ: {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("🚀 Bắt đầu trích xuất và vẽ biểu đồ...")
    
    try:
        # 1. Lấy dữ liệu gộp
        data = extract_merged_data(LOG_DIR)
        
        # 2. Vẽ từng cái
        count = 0
        for tag, (title, fname, color) in METRICS_TO_PLOT.items():
            if tag in data:
                plot_single_metric(data[tag], title, fname, color, OUTPUT_DIR)
                count += 1
            else:
                print(f"⚠️ Không có dữ liệu cho tag: {tag}")
                
        print(f"\n🎉 HOÀN TẤT! Đã xuất {count} biểu đồ vào thư mục '{OUTPUT_DIR}'")
        
    except Exception as e:
        print(f"❌ CÓ LỖI XẢY RA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()