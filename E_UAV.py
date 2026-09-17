import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


global_n0 = 3.98e-21
global_rho = 1.0
global_min_uav2uav_trans_rate = 60.0
global_min_uav_energy = 0.1
Energy_Threshold = 0.0


class UAVType(Enum):
    ClusterHead = auto()
    BackupClusterHead = auto()
    Linker = auto()
    Explorer = auto()
    Normal = auto()
    BASE = 900


@dataclass
class UavNeighbor:
    uavID: int
    uav_type: UAVType
    uav_position: Tuple[float, float, float]
    uav_energy: float
    uav_load: float
    uav_speed: float


class UAV:
    def __init__(
        self,
        uavID=1,
        uav_position=(10, 10, 999),
        uav_speed=10,
        uav_transpower=0,
        uav_bandwidth=0,
        MAX_LOAD=100,
        MAX_ENERGY=100,
        uav_type=UAVType.Normal,
    ):
        self.uavID = uavID
        self.uav_position = tuple(float(v) for v in uav_position)
        self.uav_speed = float(uav_speed)
        self.uav_TransPower = float(uav_transpower)  # 发射功率
        self.uav_Bandwidth = float(uav_bandwidth)
        self.uav2uav_TransRate: dict[int, float] = {}
        self.MAX_LOAD = MAX_LOAD
        self.MAX_ENERGY = MAX_ENERGY
        self.uav_type = uav_type
        self.uav_energy = MAX_ENERGY
        self.uav_load = 0
        self.communication_radius = 0.0
        self.serving_GroundUser_set = []
        self.num_UAV_neighbor = 0
        self.uav_neighbor_set: dict[int, UavNeighbor] = {}
        self.num_serving_GroundUser = 0
        self.is_active = True

    def uavMove(self, moving_duration: float, destination: Tuple[float, float, float]):
        """
        旧版兼容接口：按照 UAV 速度和 moving_duration 朝 destination 移动。

        新的强化学习环境不再使用该函数执行 Actor 动作，而改用
        ``moveByDelta``，从而避免 Actor 给出的多个不同 delta 被速度限制
        压缩成相同的实际位移（动作别名问题）。
        """
        if not self.is_active:
            print(f"\033[91m [ERROR] UAV {self.uavID} is not active, can not move \033[0m")
            return self.uav_position

        destination = tuple(float(v) for v in destination)
        dx = destination[0] - self.uav_position[0]
        dy = destination[1] - self.uav_position[1]
        dz = destination[2] - self.uav_position[2]

        distance_to_destination = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        moving_distance = max(0.0, self.uav_speed * float(moving_duration))

        if distance_to_destination <= 1e-12:
            return self.uav_position

        if moving_distance >= distance_to_destination:
            self.uav_position = destination
        else:
            ratio = moving_distance / distance_to_destination
            self.uav_position = (
                self.uav_position[0] + dx * ratio,
                self.uav_position[1] + dy * ratio,
                self.uav_position[2] + dz * ratio,
            )

        return self.uav_position

    def moveByDelta(
        self,
        delta: Tuple[float, float, float],
        position_bounds=None,
    ):
        """
        强化学习专用动作接口：Actor 输出的 delta 就是本 decision step 的
        实际三维位移，不再二次经过 ``uav_speed * moving_duration`` 截断。

        Parameters
        ----------
        delta:
            (dx, dy, dz)，单位与环境坐标一致。
        position_bounds:
            可选，格式：
            ((x_min, x_max), (y_min, y_max), (z_min, z_max))。
            如果目标位置越界，会裁剪到边界；函数返回实际执行的位移，
            因而 Replay Buffer / Reward Model 可以知道真正发生了什么。

        Returns
        -------
        new_position, executed_delta
        """
        if not self.is_active:
            return self.uav_position, (0.0, 0.0, 0.0)

        dx, dy, dz = (float(delta[0]), float(delta[1]), float(delta[2]))
        old_position = tuple(float(v) for v in self.uav_position)

        new_position = [
            old_position[0] + dx,
            old_position[1] + dy,
            old_position[2] + dz,
        ]

        if position_bounds is not None:
            if len(position_bounds) != 3:
                raise ValueError("position_bounds must contain x/y/z ranges.")
            for axis in range(3):
                low, high = position_bounds[axis]
                if low > high:
                    raise ValueError("Each position bound must satisfy low <= high.")
                new_position[axis] = min(max(new_position[axis], float(low)), float(high))

        self.uav_position = tuple(new_position)
        executed_delta = (
            self.uav_position[0] - old_position[0],
            self.uav_position[1] - old_position[1],
            self.uav_position[2] - old_position[2],
        )
        return self.uav_position, executed_delta

    def getUav2UavTransRate(
        self,
        uavID,
        uav_position,
        n0=global_n0,
        rho=global_rho,
        min_uav2uav_trans_rate: float = global_min_uav2uav_trans_rate,
    ):
        if self.uav_Bandwidth <= 0 or self.uav_TransPower <= 0:
            self.uav2uav_TransRate[uavID] = 0.0
            return 0.0

        dx = float(uav_position[0]) - self.uav_position[0]
        dy = float(uav_position[1]) - self.uav_position[1]
        dz = float(uav_position[2]) - self.uav_position[2]
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # 两节点坐标完全重合时避免除零。这里用一个很小的正数代替距离。
        distance = max(distance, 1e-6)

        h = float(rho) / distance ** 2
        sinr = (h * self.uav_TransPower) / (float(n0) * self.uav_Bandwidth)
        uav2uav_trans_rate = self.uav_Bandwidth * math.log2(1.0 + sinr)

        # 每次都写入，避免字典中残留上一时刻的旧链路速率。
        self.uav2uav_TransRate[uavID] = float(uav2uav_trans_rate)
        return float(uav2uav_trans_rate)

    def appendServingGroundUser(self, groundUserID):
        if groundUserID not in self.serving_GroundUser_set:
            self.serving_GroundUser_set.append(groundUserID)
            self.num_serving_GroundUser += 1
        else:
            print(
                f"\033[91m [ERROR] UAV{self.uavID} has been serving "
                f"GroundUser{groundUserID}, but append again\033[0m"
            )

    def appendUAVNeighbor(self, uav_neighbor: UavNeighbor):
        if uav_neighbor.uavID not in self.uav_neighbor_set:
            self.num_UAV_neighbor += 1
        self.uav_neighbor_set[uav_neighbor.uavID] = uav_neighbor

    def isActive(self):
        """用于设置和返回 UAV 是否 active。"""
        if self.is_active and self.uav_energy >= Energy_Threshold:
            self.is_active = True
            return True
        self.is_active = False
        return False

    def countDistance(self, position=(100, 100, 100)):
        dx = float(position[0]) - self.uav_position[0]
        dy = float(position[1]) - self.uav_position[1]
        dz = float(position[2]) - self.uav_position[2]
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def infoState(self):
        print(
            f"\033[96m[UAV STATE] UAV {self.uavID} "
            f"\nPosition{self.uav_position} "
            f"\nEnergy{self.uav_energy}\033[0m"
        )
