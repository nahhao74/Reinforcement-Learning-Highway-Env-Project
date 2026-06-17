from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Tuple, Dict
import numpy as np

# Chỉ import những thứ cơ bản về làn đường
from highway_env.road.lane import AbstractLane, LineType, StraightLane, lane_from_config

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

class RoadNetwork:
    """
    Mạng lưới đường bộ dạng Graph.
    Đã tinh giản: Bỏ BFS/ShortestPath, chỉ giữ logic kết nối làn đường cơ bản.
    """
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        _from, _to, _id = index
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position: np.ndarray, heading: float | None = None) -> LaneIndex:
        """Tìm làn đường gần nhất với vị trí xe (Quan trọng để xác định xe đang ở đâu)."""
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index: LaneIndex, position: np.ndarray = None, np_random: np.random.RandomState = np.random) -> LaneIndex:
        """
        Tìm làn đường tiếp theo khi hết đoạn đường hiện tại.
        (Dùng cho infinite road hoặc nối tiếp các đoạn đường).
        """
        _from, _to, _id = current_index
        next_to = None
        
        # Tìm node tiếp theo trong graph
        if _to in self.graph:
            next_to_candidates = list(self.graph[_to].keys())
            if next_to_candidates:
                # Mặc định đi thẳng tiếp (lấy node đầu tiên tìm thấy)
                # Với highway thẳng, thường chỉ có 1 node tiếp theo
                next_to = next_to_candidates[0]

        if not next_to:
            return current_index # Không có đường tiếp, giữ nguyên

        # Tìm làn tương ứng ở đoạn đường tiếp theo (cùng ID hoặc gần nhất)
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)
        
        return self._next_lane_given_next_road(_from, _to, _id, next_to, projected_position)

    def _next_lane_given_next_road(self, _from: str, _to: str, _id: int, next_to: str, position: np.ndarray) -> LaneIndex:
        # Nếu số lượng làn bằng nhau, giữ nguyên ID làn (đi thẳng)
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        else:
            # Nếu số làn thay đổi (thu hẹp/mở rộng), chọn làn gần nhất về mặt hình học
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes, key=lambda l: self.get_lane((_to, next_to, l)).distance(position))
        
        return _to, next_to, next_id

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """Lấy danh sách các làn bên cạnh (Trái/Phải) -> QUAN TRỌNG CHO VIỆC RẼ."""
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0: # Làn trái
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1: # Làn phải
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def straight_road_network(lanes: int = 4, start: float = 0.0, length: float = 10000.0, angle: float = 0.0, speed_limit: float = 30.0) -> RoadNetwork:
        net = RoadNetwork()
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane("0", "1", StraightLane(origin, end, line_types=line_types, speed_limit=speed_limit))
        return net

    def lanes_list(self) -> List[AbstractLane]:
        return [lane for to in self.graph.values() for ids in to.values() for lane in ids]

    @classmethod
    def from_config(cls, config: dict) -> RoadNetwork:
        net = cls()
        for _from, to_dict in config.items():
            net.graph[_from] = {}
            for _to, lanes_dict in to_dict.items():
                net.graph[_from][_to] = []
                for lane_dict in lanes_dict:
                    net.graph[_from][_to].append(lane_from_config(lane_dict))
        return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(lane.to_config())
        return graph_dict


class Road:
    """
    Quản lý các thực thể trên đường (Vehicles, Objects).
    """
    def __init__(self, network: RoadNetwork = None, vehicles: List[kinematics.Vehicle] = None, road_objects: List[objects.RoadObject] = None, np_random: np.random.RandomState = None, record_history: bool = False):
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_objects_to(self, vehicle: kinematics.Vehicle, distance: float, count: int | None = None, see_behind: bool = True, sort: bool = True, vehicles_only: bool = False) -> object:
        """Tìm các vật thể gần xe Ego (Dùng cho Sensor/Observation)."""
        vehicles = [v for v in self.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]
        
        obstacles = []
        if not vehicles_only:
            obstacles = [o for o in self.objects
                         if np.linalg.norm(o.position - vehicle.position) < distance
                         and -2 * vehicle.LENGTH < vehicle.lane_distance_to(o)]

        objects_ = vehicles + obstacles

        if sort:
            objects_ = sorted(objects_, key=lambda o: abs(vehicle.lane_distance_to(o)))
        if count:
            objects_ = objects_[:count]
        return objects_

    def act(self) -> None:
        """Quyết định hành động cho mọi xe trên đường."""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """Cập nhật vật lý cho mọi xe."""
        for vehicle in self.vehicles:
            vehicle.step(dt)
        
        # Xử lý va chạm
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i + 1:]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)

    def neighbour_vehicles(self, vehicle: kinematics.Vehicle, lane_index: LaneIndex = None) -> Tuple[kinematics.Vehicle | None, kinematics.Vehicle | None]:
        """
        Tìm xe liền trước và liền sau (Dùng cho IDM/MOBIL).
        """
        # [FIX] Import Landmark ở đây để tránh Circular Import
        from highway_env.vehicle.objects import Landmark

        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
            
        lane = self.network.get_lane(lane_index)
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark):
                s_v, lat_v = lane.local_coordinates(v.position)
                # Chỉ xét xe nằm trên cùng làn (hoặc gần làn đó)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                
                # Tìm xe trước gần nhất
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                # Tìm xe sau gần nhất
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
                    
        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()