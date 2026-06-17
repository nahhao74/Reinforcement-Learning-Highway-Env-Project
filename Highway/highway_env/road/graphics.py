from __future__ import annotations
import numpy as np
import pygame
from highway_env.road.lane import AbstractLane, LineType
from highway_env.road.road import Road

class WorldSurface(pygame.Surface):
    """ Quản lý tọa độ Zoom/Scroll của cửa sổ Pygame """
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    
    def __init__(self, size, flags, surf):
        super().__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING

    def pix(self, length): return int(length * self.scaling)
    def pos2pix(self, x, y): return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])
    def vec2pix(self, vec): return self.pos2pix(vec[0], vec[1])

    def move_display_window_to(self, position):
        self.origin = position - np.array([
            self.centering_position[0] * self.get_width() / self.scaling,
            self.centering_position[1] * self.get_height() / self.scaling
        ])

    def is_visible(self, position, margin=None):
        """
        Kiểm tra xem một vị trí (world coordinates) có nằm trong vùng hiển thị của cửa sổ không.
        Dùng để tối ưu rendering: không vẽ những thứ nằm ngoài màn hình.
        """
        if margin is None:
            margin = 50 / self.scaling # 50 pixels margin
            
        x, y = self.pos2pix(position[0], position[1])
        # Kiểm tra xem pixel có nằm trong kích thước surface không (cộng thêm margin)
        # Lưu ý: get_width() trả về kích thước pixel của cửa sổ
        return -50 <= x <= self.get_width() + 50 and -50 <= y <= self.get_height() + 50

class LaneGraphics:
    STRIPE_SPACING: float = 4.33
    STRIPE_LENGTH: float = 3
    STRIPE_WIDTH: float = 0.3

    @classmethod
    def display(cls, lane: AbstractLane, surface: WorldSurface):
        stripes_count = int(2 * (surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling))
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING
        
        for side in range(2):
            if lane.line_types[side] != LineType.NONE:
                cls.draw_line(lane, surface, stripes_count, s0, side, lane.line_types[side])

    @classmethod
    def draw_line(cls, lane, surface, stripes_count, longitudinal, side, type_):
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        if type_ == LineType.STRIPED:
            ends = starts + cls.STRIPE_LENGTH
        else: # Continuous
            ends = starts + cls.STRIPE_SPACING
            
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        
        # Vẽ từng đoạn
        for k, _ in enumerate(starts):
            if abs(starts[k] - ends[k]) > 0.5:
                # Chỉ vẽ nếu đoạn thẳng nằm trong màn hình (gọi hàm vec2pix để lấy tọa độ pixel)
                p1 = surface.vec2pix(lane.position(starts[k], lats[k]))
                p2 = surface.vec2pix(lane.position(ends[k], lats[k]))
                
                # Check nhanh xem có cần vẽ không
                if (0 <= p1[0] <= surface.get_width() and 0 <= p1[1] <= surface.get_height()) or \
                   (0 <= p2[0] <= surface.get_width() and 0 <= p2[1] <= surface.get_height()):
                    pygame.draw.line(
                        surface, (255, 255, 255),
                        p1,
                        p2,
                        max(surface.pix(cls.STRIPE_WIDTH), 1)
                    )

class RoadGraphics:
    @staticmethod
    def display(road: Road, surface: WorldSurface):
        surface.fill((100, 100, 100)) # Grey Road
        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for l in road.network.graph[_from][_to]:
                    LaneGraphics.display(l, surface)

    @staticmethod
    def display_traffic(road: Road, surface: WorldSurface):
        # Import ở đây để tránh circular import
        from highway_env.vehicle.graphics import VehicleGraphics 
        for v in road.vehicles:
            VehicleGraphics.display(v, surface)