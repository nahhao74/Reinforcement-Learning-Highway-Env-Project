from __future__ import annotations
import os
import numpy as np
import pygame
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.vehicle.graphics import VehicleGraphics

class EnvViewer:
    """
    Trình quản lý hiển thị đồ họa cho môi trường Highway-env sử dụng thư viện Pygame.
    """
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or env.config
        self.offscreen = self.config["offscreen_rendering"]
        
        # Khởi tạo Pygame và thiết lập tiêu đề cửa sổ
        pygame.init()
        pygame.display.set_caption("Highway-env - Autonomous Driving Project")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        # Thiết lập chế độ hiển thị (Cửa sổ hoặc Offscreen để quay video)
        if not self.offscreen:
            self.screen = pygame.display.set_mode(panel_size)
        
        # WorldSurface chịu trách nhiệm chuyển đổi tọa độ thực (m) sang tọa độ điểm ảnh (pixel)
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get("scaling", 5.5)
        self.sim_surface.centering_position = self.config.get("centering_position", [0.3, 0.5])
        self.clock = pygame.time.Clock()

    def handle_events(self):
        """Xử lý các sự kiện tương tác từ bàn phím hoặc chuột (ví dụ: đóng cửa sổ)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                self.env.close()

    def display(self):
        """Hàm vẽ chính, được gọi tại mỗi bước mô phỏng (step)."""
        # Di chuyển camera theo xe Ego
        self.sim_surface.move_display_window_to(self.env.vehicle.position)
        
        # Vẽ mặt đường và các làn đường
        RoadGraphics.display(self.env.road, self.sim_surface)
        
        # Vẽ các phương tiện giao thông khác trên đường
        RoadGraphics.display_traffic(self.env.road, self.sim_surface)
        
        # === HIỂN THỊ THÔNG SỐ TRẠNG THÁI (Dashboard) ===
        DashboardGraphics.display(self.env, self.sim_surface)

        if not self.offscreen:
            # Đẩy hình ảnh từ surface mô phỏng lên màn hình hiển thị
            self.screen.blit(self.sim_surface, (0, 0))
            # Giới hạn tốc độ khung hình theo tần số vật lý của môi trường
            self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

    def get_image(self) -> np.ndarray:
        """Xuất khung hình hiện tại dưới dạng mảng Numpy (RGB) để phục vụ việc lưu Video."""
        data = pygame.surfarray.array3d(self.sim_surface)
        return np.moveaxis(data, 0, 1)

    def close(self):
        """Giải phóng tài nguyên Pygame khi kết thúc mô phỏng."""
        pygame.quit()

class DashboardGraphics:
    """
    Lớp hỗ trợ vẽ các thông tin trạng thái trực tiếp lên màn hình mô phỏng.
    """
    @classmethod
    def display(cls, env, surface):
        # Thiết lập font chữ
        font = pygame.font.SysFont("Arial", 25, bold=True)
        
        # 1. Tính toán vận tốc xe Ego (km/h)
        speed_kmh = env.vehicle.speed * 3.6
        speed_text = font.render(f"Speed: {speed_kmh:.0f} km/h", True, (255, 255, 255))
        
        # 2. Hiển thị thông tin làn đường hiện tại
        lane_text = font.render(f"Lane: {env.vehicle.lane_index[2]}", True, (255, 255, 0))
        
        # Vẽ các thông số lên góc trái màn hình
        surface.blit(speed_text, (20, 20))
        surface.blit(lane_text, (20, 50))
        
        # 3. Vẽ chỉ báo hành động (Nếu Agent đang thực hiện rẽ)
        # Giúp báo cáo trực quan hơn về việc tại sao xe lại chuyển làn
        ego_pos = env.vehicle.position
        pix_pos = surface.pos2pix(ego_pos[0], ego_pos[1])
        
        # Nếu xe đang va chạm, hiển thị cảnh báo đỏ ngay trên xe
        if env.vehicle.crashed:
            crash_font = pygame.font.SysFont("Arial", 30, bold=True)
            crash_text = crash_font.render("COLLISION!", True, (255, 0, 0))
            surface.blit(crash_text, (pix_pos[0] - 60, pix_pos[1] - 80))