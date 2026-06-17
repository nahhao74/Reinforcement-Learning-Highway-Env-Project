from __future__ import annotations
import copy
from typing import List, Optional, Tuple, Union
import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle

class ControlledVehicle(Vehicle):
    """
    Xe được điều khiển bởi 2 bộ controller PID:
    - Longitudinal: Điều khiển tốc độ (Gas/Brake)
    - Lateral: Điều khiển hướng (Steering) để giữ tâm làn
    """

    target_speed: float
    """ Desired velocity."""

    # --- CÁC HỆ SỐ PID GỐC ---
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
        )
        return v

    

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Thực hiện hành động mức cao (High-level action).
        Logic: Chỉ chấp nhận chuyển làn khi xe đã ổn định (target == current).
        """
        self.follow_road()
        
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
            
        elif action == "LANE_RIGHT":
            # --- LOGIC QUAN TRỌNG: CHẶN RẼ LIÊN TỤC ---
            # Chỉ cho phép rẽ nếu xe đang ổn định tại làn mục tiêu hiện tại
            if self.target_lane_index == self.lane_index:
                _from, _to, _id = self.target_lane_index
                target_lane_index = (
                    _from,
                    _to,
                    np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
                )
                if self.road.network.get_lane(target_lane_index).is_reachable_from(
                    self.position
                ):
                    self.target_lane_index = target_lane_index
                    
        elif action == "LANE_LEFT":
            # --- LOGIC QUAN TRỌNG: CHẶN RẼ LIÊN TỤC ---
            if self.target_lane_index == self.lane_index:
                _from, _to, _id = self.target_lane_index
                target_lane_index = (
                    _from,
                    _to,
                    np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
                )
                if self.road.network.get_lane(target_lane_index).is_reachable_from(
                    self.position
                ):
                    self.target_lane_index = target_lane_index

        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

    def follow_road(self) -> None:
        """Tự động chuyển sang đoạn đường tiếp theo khi hết đường hiện tại"""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(
                self.target_lane_index,
                # route=self.route,  <--- ĐÃ XÓA DÒNG NÀY ĐỂ SỬA LỖI
                position=self.position,
                np_random=self.road.np_random,
            )

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """Tính toán góc lái (Steering Angle) dùng PID"""
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        )
        heading_ref = lane_future_heading + np.clip(
            heading_command, -np.pi / 4, np.pi / 4
        )
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        # Heading rate to steering angle
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """Tính toán gia tốc (Acceleration) dùng PID"""
        return self.KP_A * (target_speed - self.speed)


class MDPVehicle(ControlledVehicle):
    """
    Xe Agent với không gian hành động rời rạc (Discrete Action Space)
    """

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)

    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
    ) -> None:
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Thực hiện hành động:
        - Nếu là thay đổi tốc độ: Cập nhật target_speed từ danh sách rời rạc.
        - Nếu là chuyển làn: Gọi hàm act của lớp cha (ControlledVehicle).
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            # Chuyển tiếp hành động (LANE_LEFT, LANE_RIGHT, IDLE) cho lớp cha xử lý
            super().act(action)
            return
            
        # Xử lý giới hạn tốc độ và cập nhật target_speed
        self.speed_index = int(
            np.clip(self.speed_index, 0, self.target_speeds.size - 1)
        )
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        x = (speed - self.target_speeds[0]) / (
            self.target_speeds[-1] - self.target_speeds[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (self.target_speeds.size - 1)),
                0,
                self.target_speeds.size - 1,
            )
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (
            cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)),
                0,
                cls.DEFAULT_TARGET_SPEEDS.size - 1,
            )
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )