import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TupleFlattenWrapper(gym.ObservationWrapper):
    """
    Wrapper này làm phẳng Observation dạng Tuple thành một Box 1 chiều (Vector).
    Cần thiết để PPO (MlpPolicy) hiểu được dữ liệu đầu vào.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Kiểm tra xem env có phải Tuple không
        if not isinstance(env.observation_space, spaces.Tuple):
            raise ValueError("TupleFlattenWrapper chỉ dùng cho Tuple observation space")

        # Tính toán kích thước không gian mới sau khi gộp
        lows = []
        highs = []
        for s in env.observation_space.spaces:
            if not isinstance(s, spaces.Box):
                raise ValueError("Chỉ hỗ trợ Tuple của các Box spaces")
            lows.append(np.reshape(s.low, (-1,)))
            highs.append(np.reshape(s.high, (-1,)))
            
        low = np.concatenate(lows).astype(np.float32)
        high = np.concatenate(highs).astype(np.float32)
        
        # Định nghĩa observation space mới
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        """
        Hàm này tự động được gọi mỗi khi env trả về observation (reset hoặc step).
        Nó sẽ làm phẳng dữ liệu trước khi đưa cho Agent.
        """
        parts = []
        for o in observation:
            parts.append(np.asarray(o).flatten())
        return np.concatenate(parts).astype(np.float32)