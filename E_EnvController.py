import random
from typing import Optional

from E_Monitor import Monitor, base
from E_UAV import UAV, UAVType


COMMUNICATION_RADIUS = 2000.0
UAV_TRANS_POWER = 1.0
UAV_BANDWIDTH = 1e6
MIN_TRANS_RATE = 60.0
ALTITUDE = 0.0
COVERAGE_RADIUS = 500.0
BASE_ID = base.uavID

# 与 Monitor.getState() 的默认归一化范围保持一致。
WORLD_BOUNDS = (
    (0.0, 10000.0),
    (0.0, 10000.0),
    (0.0, 1000.0),
)


class EnvController:
    def __init__(self):
        print("EnvController is working")

    def make_uav(self, uav_id, position, speed=20.0):
        """
        创建一架 UAV。

        注意：UAV 类默认 communication_radius / 发射功率 / 带宽均可能为 0，
        因而场景构造时必须在这里显式设置。
        """
        uav = UAV(
            uavID=int(uav_id),
            uav_position=tuple(float(v) for v in position),
            uav_speed=float(speed),
            uav_transpower=UAV_TRANS_POWER,
            uav_bandwidth=UAV_BANDWIDTH,
            MAX_LOAD=100,
            MAX_ENERGY=10000,
            uav_type=UAVType.Normal,
        )
        uav.communication_radius = COMMUNICATION_RADIUS
        return uav

    def _new_monitor(self):
        return Monitor(
            position_bounds=WORLD_BOUNDS,
            coverage_radius=COVERAGE_RADIUS,
        )

    def update_network_without_base(self, monitor):
        """
        更新普通 UAV 网络并确保 base=900 不参与本训练场景。
        """
        monitor.updateEachUavNeighbors(min_uav2uav_trans_rate=MIN_TRANS_RATE)

        monitor.uav_net.pop(BASE_ID, None)
        for neighbors in monitor.uav_net.values():
            neighbors.discard(BASE_ID)

        for uav in monitor.uav_set.values():
            if BASE_ID in uav.uav_neighbor_set:
                del uav.uav_neighbor_set[BASE_ID]
                uav.num_UAV_neighbor = max(0, uav.num_UAV_neighbor - 1)

        return monitor.uav_net

    def print_network_state(self, title, monitor):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

        print("UAV positions:")
        for uav_id in sorted(monitor.uav_set.keys()):
            print(f"  UAV{uav_id}: {monitor.uav_set[uav_id].uav_position}")

        print("\nUAV network adjacency list:")
        for uav_id in sorted(monitor.uav_net.keys()):
            print(f"  UAV{uav_id}: {sorted(monitor.uav_net[uav_id])}")

        is_connected, components = monitor.checkUavnetConnectivity()
        print(f"\nIs connected? {is_connected}")
        print(f"Connected components: {[sorted(c) for c in components]}")
        return is_connected, components

    def build_monitor_scene(self):
        """
        构造用于验证 LLM-preference -> Reward Model -> SAC 闭环的固定 8-UAV 场景。

        当前 sanity-check 任务只有一个目标：让所有 UAV 最终移动到 (0, 0, 0)。
        因此初始位置刻意限制在离原点较近的区域，避免任务难度被长距离移动主导，
        从而更直接地检查偏好标注、Reward Model 和 Actor 是否能够正确学习。
        """
        # base 在本任务中不参与训练，把它放远只作为额外保险。
        base.uav_position = (100000.0, 100000.0, ALTITUDE)

        monitor = self._new_monitor()
        positions = {
            0: (200.0, 200.0, ALTITUDE),
            1: (400.0, 100.0, ALTITUDE),
            2: (100.0, 400.0, ALTITUDE),
            3: (400.0, 400.0, ALTITUDE),
            4: (600.0, 200.0, ALTITUDE),
            5: (800.0, 100.0, ALTITUDE),
            6: (600.0, 500.0, ALTITUDE),
            7: (800.0, 500.0, ALTITUDE),
        }

        # positions = {
        #     0: (0.0, 0.0, ALTITUDE),
        #     1: (0.0, 0.0, ALTITUDE),
        #     2: (0.0, 0.0, ALTITUDE),
        #     3: (0.0, 0.0, ALTITUDE),
        #     4: (0.0, 0.0, ALTITUDE),
        #     5: (0.0, 0.0, ALTITUDE),
        #     6: (0.0, 0.0, ALTITUDE),
        #     7: (800.0, 500.0, ALTITUDE),
        # }

        for uav_id, pos in positions.items():
            monitor.addUAV(self.make_uav(uav_id, pos))

        self.update_network_without_base(monitor)
        return monitor

    def build_random_scene(
        self,
        num_uav: int = 8,
        seed: Optional[int] = None,
        num_clusters: int = 3,
        cluster_spread: float = 500.0,
        min_cluster_center_distance: float = 2600.0,
        altitude: float = ALTITUDE,
        max_attempts: int = 500,
    ):
        """
        构造用于后续泛化训练/验证的随机多簇场景。

        该函数不是用随机坐标简单撒点，而是先生成相互较远的 cluster center，
        再在每个 cluster 周围生成 UAV，从而更容易产生“多个内部连通、簇间不连通”
        的初始网络，与当前固定训练问题保持同类结构。

        Parameters
        ----------
        num_uav:
            UAV 数量。后续 Actor 若固定 MAX_NUM_UAV=8，则应满足 num_uav <= 8。
        seed:
            随机种子，便于复现实验。
        num_clusters:
            初始簇数量。
        cluster_spread:
            UAV 相对簇中心在 XY 平面的最大随机偏移。
        min_cluster_center_distance:
            不同簇中心之间的最小距离。
        """
        if num_uav <= 0:
            raise ValueError("num_uav must be positive.")
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive.")
        num_clusters = min(int(num_clusters), int(num_uav))

        rng = random.Random(seed)
        base.uav_position = (100000.0, 100000.0, float(altitude))
        monitor = self._new_monitor()

        x_min, x_max = WORLD_BOUNDS[0]
        y_min, y_max = WORLD_BOUNDS[1]
        margin = max(COMMUNICATION_RADIUS, cluster_spread) + 100.0

        center_x_min = x_min + margin
        center_x_max = x_max - margin
        center_y_min = y_min + margin
        center_y_max = y_max - margin
        if center_x_min >= center_x_max or center_y_min >= center_y_max:
            raise ValueError("WORLD_BOUNDS are too small for the requested random scene.")

        centers = []
        attempts = 0
        while len(centers) < num_clusters and attempts < max_attempts:
            attempts += 1
            candidate = (
                rng.uniform(center_x_min, center_x_max),
                rng.uniform(center_y_min, center_y_max),
            )
            if all(
                ((candidate[0] - cx) ** 2 + (candidate[1] - cy) ** 2) ** 0.5
                >= min_cluster_center_distance
                for cx, cy in centers
            ):
                centers.append(candidate)

        if len(centers) < num_clusters:
            raise RuntimeError(
                "Could not place enough separated cluster centers. "
                "Reduce min_cluster_center_distance or num_clusters."
            )

        # 尽量平均地把 UAV 分配给各个簇。
        cluster_sizes = [num_uav // num_clusters] * num_clusters
        for idx in range(num_uav % num_clusters):
            cluster_sizes[idx] += 1

        positions = {}
        uav_id = 0
        for cluster_idx, cluster_size in enumerate(cluster_sizes):
            cx, cy = centers[cluster_idx]
            for _ in range(cluster_size):
                x = min(max(cx + rng.uniform(-cluster_spread, cluster_spread), x_min), x_max)
                y = min(max(cy + rng.uniform(-cluster_spread, cluster_spread), y_min), y_max)
                z = min(max(float(altitude), WORLD_BOUNDS[2][0]), WORLD_BOUNDS[2][1])
                positions[uav_id] = (x, y, z)
                uav_id += 1

        for current_id, pos in positions.items():
            monitor.addUAV(self.make_uav(current_id, pos))

        self.update_network_without_base(monitor)
        return monitor


if __name__ == "__main__":
    controller = EnvController()
    scene = controller.build_monitor_scene()
    controller.print_network_state("Fixed 8-UAV scene", scene)
