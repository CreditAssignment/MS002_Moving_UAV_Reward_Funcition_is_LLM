import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from E_UAV import UAV, UAVType, UavNeighbor, global_min_uav2uav_trans_rate, global_min_uav_energy
from E_GroundUser import GroundUser

base = UAV(uavID=900,
           uav_position=(10000, 10000, 0),
           uav_speed=0,
           uav_type=UAVType.BASE,
           MAX_ENERGY=100000)  # base作为UAV的实例对象只是为了方便，但是base不是UAV


# -----------------------------------------------------------------------------
# Sanity-check task: move every UAV to the origin.
# The optimization target is exactly (0, 0, 0); tolerance is used only to
# decide whether an episode has successfully finished.
# -----------------------------------------------------------------------------
TARGET_POSITION = (0.0, 0.0, 0.0)
TARGET_TOLERANCE = 30.0


class Monitor:
    def __init__(
        self,
        position_bounds=((0.0, 10000.0), (0.0, 10000.0), (0.0, 1000.0)),
        coverage_radius=500.0,
    ):
        self.uav_set: dict[int, UAV] = {}  # 不包括 Base
        self.ground_user_set: dict = {}
        self.num_ALL_UAV = 0
        self.num_Active_UAV = 0
        self.num_Damaged_UAV = 0
        self.num_ground_user = 0
        self.uav_net = {}  # UAV 网络图的邻接表表示
        self.remaining_uav_ids: dict[int, float] = {}
        self.damaged_uav_ids = {}

        # 强化学习动作执行边界。Actor 输出的是实际 delta；只有在越界时
        # 才会发生裁剪，并把实际 executed_delta 写入 transition info。
        self.position_bounds = tuple(
            (float(bound[0]), float(bound[1])) for bound in position_bounds
        )
        if len(self.position_bounds) != 3:
            raise ValueError("position_bounds must contain x/y/z ranges.")
        self.coverage_radius = float(coverage_radius)

    def addUAV(self, uav):
        self.uav_set[uav.uavID] = uav
        self.remaining_uav_ids[uav.uavID] = uav.uav_energy
        self.num_ALL_UAV += 1
        self.num_Active_UAV += 1
        self.updateAllAboutUAVnet()

    def addGroundUser(self, groundUser):
        self.num_ground_user += 1
        self.ground_user_set[groundUser.userID] = groundUser

    def deleteUAV(self, uav):
        # del self.uav_set[uav.uavID]
        del self.remaining_uav_ids[uav.uavID]
        self.damaged_uav_ids[uav.uavID] = uav.uav_energy
        self.num_Active_UAV -= 1
        self.num_Damaged_UAV += 1
        self.updateAllAboutUAVnet()

    def deleteGroundUser(self, groundUser):
        del self.ground_user_set[groundUser.userID]
        self.num_ground_user -= 1

    def setUavEnergy(self, uavID, uav_energy):
        if uavID not in self.uav_set:
            print(f"\033[91m UAV {uavID} not exists \033[0m")
        elif uav_energy < global_min_uav_energy:
            print("设置UAV能量低于阈值")
            self.uav_set[uavID].uav_energy = uav_energy
            if uavID in self.remaining_uav_ids.keys():
                del self.remaining_uav_ids[uavID]
            self.damaged_uav_ids[uavID] = uav_energy
            self.num_Active_UAV = len(self.remaining_uav_ids)
            self.num_Damaged_UAV = len(self.damaged_uav_ids)
            self.updateAllAboutUAVnet()
        else:
            print("设置UAV能量高于阈值")
            self.uav_set[uavID].uav_energy = uav_energy
            self.remaining_uav_ids[uavID] = uav_energy
            if uavID in self.damaged_uav_ids.keys():
                del self.remaining_uav_ids[uavID]
            self.num_Active_UAV = len(self.remaining_uav_ids)
            self.num_Damaged_UAV = len(self.damaged_uav_ids)
            self.updateAllAboutUAVnet()

    def moveUAV(self, uavID: int, destination: tuple[float, float, float], moving_duration: float):
        """旧版兼容接口：按速度和持续时间朝 destination 移动。"""
        self.uav_set[uavID].uavMove(
            moving_duration=moving_duration,
            destination=destination,
        )
        self.updateAllAboutUAVnet()

    def moveUAVByDelta(self, uavID: int, delta):
        """
        强化学习专用接口。Actor 给出的 delta 直接作为本 decision step 的
        实际位移；仅在位置越出环境边界时裁剪。

        Returns
        -------
        new_position, executed_delta
        """
        if uavID not in self.uav_set:
            raise KeyError(f"UAV {uavID} does not exist.")

        new_position, executed_delta = self.uav_set[uavID].moveByDelta(
            delta=delta,
            position_bounds=self.position_bounds,
        )
        self.updateAllAboutUAVnet()
        return new_position, executed_delta

    def getUavState(self, uavID):
        uav = self.uav_set[uavID]
        print(
            f"\033[96m [UAV STATE] UAV{uav.uavID} Position{uav.uav_position} Energy{uav.uav_energy} Load{uav.uav_load}\033[0m")

    def getUavServingGroundUsersState(self, uavID):
        uav = self.uav_set[uavID]
        for groundUserID in uav.serving_GroundUser_set:
            self.ground_user_set[groundUserID].infoState()

    def updateAllUavActivities(self, min_energy=global_min_uav_energy):
        # UAV的能量大于等于min_energy，则UAV是active的
        self.remaining_uav_ids.clear()
        self.damaged_uav_ids.clear()
        for uav in self.uav_set.values():
            if uav.uav_energy >= min_energy:
                uav.is_active = True
                self.remaining_uav_ids[uav.uavID] = uav.uav_energy
            else:
                uav.is_active = False
                self.damaged_uav_ids[uav.uavID] = uav.uav_position
        self.num_Active_UAV = len(self.remaining_uav_ids)
        self.num_Damaged_UAV = len(self.damaged_uav_ids)

    def updateEachUavNeighborsWithBase(self, min_uav2uav_trans_rate: float = global_min_uav2uav_trans_rate):
        """
        更新每个 UAV 的 uavNeighborSet，同时构建 Monitor 的 UAV 网络图 self.uavNet。

        判定 UAV j 是否可以成为 UAV i 的邻居：
        1. i 和 j 不是同一架 UAV；
        2. 二者距离 < UAV i 的 communicationRadius；
        3. 由 UAV i 调用 get_uav2uavTransRate(j) 得到的通信速率 > min_trans_rate。

        self.uavNet 是邻接表形式：
        {
            uav_i_id: {neighbor_1_id, neighbor_2_id, ...},
            ...
        }
        """

        # 每次更新前，先清空旧邻居信息
        self.uav_net.clear()
        self.uav_net = {uavID: set() for uavID in self.uav_set.keys() if self.uav_set[uavID].is_active}
        self.uav_net[base.uavID] = set()

        for uav in self.uav_set.values():
            uav.uav_neighbor_set.clear()
            uav.num_UAV_neighbor = 0

        # 遍历每一对 UAV
        for src_uav_id, src_uav in self.uav_set.items():
            src_uav.uav2uav_TransRate.clear()  # 清空src_uav到其他uav的传输速率字典
            # base要特殊处理
            dx = base.uav_position[0] - src_uav.uav_position[0]
            dy = base.uav_position[1] - src_uav.uav_position[1]
            dz = base.uav_position[2] - src_uav.uav_position[2]
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            trans_rate = src_uav.getUav2UavTransRate(base.uavID, base.uav_position)
            if distance <= src_uav.communication_radius and trans_rate >= min_uav2uav_trans_rate:
                uav_neighbor = UavNeighbor(
                    uavID=base.uavID,
                    uav_type=base.uav_type,
                    uav_position=base.uav_position,
                    uav_energy=base.uav_energy,
                    uav_load=base.uav_load,
                    uav_speed=base.uav_speed
                )

                src_uav.appendUAVNeighbor(uav_neighbor)
                self.uav_net[src_uav_id].add(base.uavID)
                self.uav_net[base.uavID].add(src_uav_id)

            for dst_uav_id, dst_uav in self.uav_set.items():

                # UAV 不能把自己作为邻居
                if src_uav_id == dst_uav_id:
                    continue
                # dst必须是active的
                if not self.uav_set[dst_uav_id].is_active:
                    continue

                dx = dst_uav.uav_position[0] - src_uav.uav_position[0]
                dy = dst_uav.uav_position[1] - src_uav.uav_position[1]
                dz = dst_uav.uav_position[2] - src_uav.uav_position[2]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # 条件 1：距离必须小于当前 UAV 的通信半径
                if distance > src_uav.communication_radius:
                    continue

                # 条件 2：通信速率必须大于 60 bps
                trans_rate = src_uav.getUav2UavTransRate(dst_uav_id, dst_uav.uav_position)
                if trans_rate < min_uav2uav_trans_rate:
                    continue

                # 构造邻居信息
                uav_neighbor = UavNeighbor(
                    uavID=dst_uav.uavID,
                    uav_type=dst_uav.uav_type,
                    uav_position=dst_uav.uav_position,
                    uav_energy=dst_uav.uav_energy,
                    uav_load=dst_uav.uav_load,
                    uav_speed=dst_uav.uav_speed
                )

                src_uav.appendUAVNeighbor(uav_neighbor)
                self.uav_net[src_uav_id].add(dst_uav_id)

    def updateEachUavNeighbors(self, min_uav2uav_trans_rate: float = global_min_uav2uav_trans_rate):
        """
        不考虑 Base，更新每个 UAV 的 uav_neighbor_set，
        同时构建 Monitor 的 UAV 网络图 self.uav_net。

        判定 UAV j 是否可以成为 UAV i 的邻居：
        1. i 和 j 不是同一架 UAV；
        2. 二者距离 < UAV i 的 communication_radius；
        3. 由 UAV i 调用 get_uav2uavTransRate(j) 得到的通信速率 > min_uav2uav_trans_rate。

        self.uav_net 是邻接表形式：
        {
            uav_i_id: {neighbor_1_id, neighbor_2_id, ...},
            ...
        }

        注意：
        该函数不会把 base.uavID=900 加入 self.uav_net。
        """

        # 每次更新前，先清空旧网络图
        # 这里只加入 self.uav_set 中的 UAV，不加入 Base，uav_net里只考虑active的UAV
        self.uav_net.clear()
        self.uav_net = {uavID: set() for uavID in self.uav_set.keys() if self.uav_set[uavID].is_active}

        # 清空每架 UAV 的旧邻居信息
        for uav in self.uav_set.values():
            uav.uav_neighbor_set.clear()
            uav.num_UAV_neighbor = 0

        # 遍历每一对 UAV
        for src_uav_id, src_uav in self.uav_set.items():
            if not self.uav_set[src_uav_id].is_active:
                continue
            # 清空 src_uav 到其他 UAV 的传输速率字典
            src_uav.uav2uav_TransRate.clear()

            for dst_uav_id, dst_uav in self.uav_set.items():

                # UAV 不能把自己作为邻居
                if src_uav_id == dst_uav_id:
                    continue
                # dst必须是active的
                if not self.uav_set[dst_uav_id].is_active:
                    continue

                dx = dst_uav.uav_position[0] - src_uav.uav_position[0]
                dy = dst_uav.uav_position[1] - src_uav.uav_position[1]
                dz = dst_uav.uav_position[2] - src_uav.uav_position[2]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                # 条件 1：距离必须小于当前 UAV 的通信半径
                if distance > src_uav.communication_radius:
                    continue

                # 条件 2：通信速率必须大于阈值
                trans_rate = src_uav.getUav2UavTransRate(
                    dst_uav_id,
                    dst_uav.uav_position
                )

                if trans_rate < min_uav2uav_trans_rate:
                    continue

                # 构造邻居信息
                uav_neighbor = UavNeighbor(
                    uavID=dst_uav.uavID,
                    uav_type=dst_uav.uav_type,
                    uav_position=dst_uav.uav_position,
                    uav_energy=dst_uav.uav_energy,
                    uav_load=dst_uav.uav_load,
                    uav_speed=dst_uav.uav_speed
                )

                # 更新 src_uav 的邻居集合
                src_uav.appendUAVNeighbor(uav_neighbor)

                # 更新 Monitor 中的邻接表
                self.uav_net[src_uav_id].add(dst_uav_id)

    def updateAllAboutUAVnet(self,
                             min_energy=global_min_uav_energy,
                             has_bas=False,
                             min_uav2uav_trans_rate=global_min_uav2uav_trans_rate):
        self.updateAllUavActivities(min_energy)
        if has_bas:
            self.updateEachUavNeighborsWithBase(min_uav2uav_trans_rate)
        else:
            self.updateEachUavNeighbors(min_uav2uav_trans_rate)

    def checkUavnetConnectivity(self):
        """
                检查 UAV 网络是否全连通，并找出所有连通子图。

                返回:
                    is_connected: bool
                        True 表示整个 UAV 网络全连通，False 表示不全连通。

                    connected_components: list[set]
                        每个 set 是一个连通子图中的 UAV ID 集合。
                        例如:
                        [
                            {900, 1, 2, 3},
                            {4, 5},
                            {6}
                        ]
        """

        if not self.uav_net:
            return True, []

        # 如果希望按“无向图”检查连通性，先构造无向邻接表
        undirected_net = {uav_id: set(neighbors) for uav_id, neighbors in self.uav_net.items()}

        for uav_id, neighbors in self.uav_net.items():  # 有UAV1->UAV2，但是没有UAV2->UAV1，则添加UAV2->UAV1
            for neighbor_id in neighbors:
                if neighbor_id not in undirected_net:
                    undirected_net[neighbor_id] = set()
                undirected_net[neighbor_id].add(uav_id)  # python的集合里，相同添加已有元素是不会重复添加的，{1, 2, 2}就是{1, 2}

        visited = set()
        connected_components = []

        for start_uav_id in undirected_net.keys():
            if start_uav_id in visited:
                continue

            # BFS 搜索一个连通子图
            component = set()
            queue = [start_uav_id]
            visited.add(start_uav_id)

            while queue:
                current_id = queue.pop(0)
                component.add(current_id)

                for neighbor_id in undirected_net[current_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

            connected_components.append(component)

        is_connected = len(connected_components) == 1

        if is_connected:
            print(f"\033[92m[UAV NET] 网络全连通，共 {len(undirected_net)} 个节点。\033[0m")
        # else:
        #     print(f"\033[91m[UAV NET] 网络不全连通，共有 {len(connected_components)} 个连通子图。\033[0m")
        #     for idx, component in enumerate(connected_components):
        #         print(f"  连通子图 {idx + 1}: {sorted(component)}")

        return is_connected, connected_components

    def getActiveMask(self):
        """
        按 ``sorted(self.uav_set.keys())`` 的固定槽位顺序返回 active mask。
        该顺序与 Actor 的 uav_slot 以及 getState()/getAllState() 保持一致。
        """
        return [
            bool(self.uav_set[uav_id].isActive())
            for uav_id in sorted(self.uav_set.keys())
        ]

    def getActionMask(
            self,
            target_position=TARGET_POSITION,
            tolerance=TARGET_TOLERANCE,
    ):
        """
        返回 Actor 的 UAV 选择 mask。

        True:
            UAV 正常工作，并且尚未进入目标区域，可以被 Actor 选择。

        False:
            UAV 已失效，或者已经进入目标区域，不允许 Actor 再次选择。
        """

        target_metrics = self.getTargetMetrics(
            target_position=target_position,
            tolerance=tolerance,
        )

        action_mask = []

        for uav_id in sorted(self.uav_set.keys()):
            uav = self.uav_set[uav_id]

            is_active = bool(uav.isActive())

            distance_to_target = target_metrics["distances"][int(uav_id)]

            has_reached_target = distance_to_target <= tolerance

            selectable = is_active and (not has_reached_target)

            action_mask.append(selectable)

        return action_mask

    def getSelectAbleMask(self):
        return self.getActionMask()

    def getState(self,
                 max_num_uav=None,
                 x_range=(0.0, 10000.0),
                 y_range=(0.0, 10000.0),
                 z_range=(0.0, 1000.0),
                 max_speed=1000.0,
                 max_comm_radius=2000.0,
                 include_distance_matrix=True,
                 auto_update_neighbors=False,
                 min_uav2uav_trans_rate=global_min_uav2uav_trans_rate,
                 return_info=False):
        """
        返回当前 UAV 网络状态，作为强化学习智能体的神经网络输入。

        返回:
            state: list[float]
                一维向量，可直接转换为 torch.tensor(state, dtype=torch.float32)

        state 结构:
            [global_features,
             node_features,
             adjacency_matrix_flatten,
             distance_matrix_flatten 可选]

        注意:
            1. 智能体不需要访问 Monitor。
            2. 智能体只接收本函数返回的 state。
            3. max_num_uav 应在训练过程中保持固定，否则神经网络输入维度会变化。
        """

        # 是否在取状态前自动更新邻接关系
        # 如果你在 monitor 执行动作后已经手动调用 updateEachUavNeighbors()，
        # 这里可以保持 False。
        if auto_update_neighbors:
            self.updateEachUavNeighbors(
                min_uav2uav_trans_rate=min_uav2uav_trans_rate
            )

        # ---------- 基本工具函数 ----------

        def safe_div(a, b):
            if b == 0:
                return 0.0
            return float(a) / float(b)

        def norm_value(v, value_range):
            low, high = value_range
            if high == low:
                return 0.0
            value = (float(v) - float(low)) / (float(high) - float(low))
            # 裁剪到 [0, 1]，避免异常坐标导致神经网络输入过大
            return max(0.0, min(1.0, value))

        def get_uav_position(uav):
            x, y, z = uav.uav_position
            return float(x), float(y), float(z)

        def is_uav_active(uav_id):
            """
            判断 UAV 是否仍然有效。
            damaged_uav_ids 中的 UAV 不参与连通性状态。
            """
            if uav_id not in self.uav_set:
                return False

            if uav_id in self.damaged_uav_ids:
                return False

            if self.remaining_uav_ids and uav_id not in self.remaining_uav_ids:
                return False

            uav = self.uav_set[uav_id]

            if hasattr(uav, "getIsActive"):
                return bool(uav.isActive())

            return bool(getattr(uav, "is_active", True))

        # ---------- UAV 顺序与最大数量 ----------

        all_uav_ids = sorted(self.uav_set.keys())

        if max_num_uav is None:
            max_num_uav = len(all_uav_ids)

        if len(all_uav_ids) > max_num_uav:
            raise ValueError(
                f"当前 UAV 数量为 {len(all_uav_ids)}，超过 max_num_uav={max_num_uav}。"
                f"请增大 max_num_uav，或者保持训练场景 UAV 数量固定。"
            )

        # slot_uav_ids 决定 state 中每个 UAV 的槽位顺序
        # 后续 monitor 执行动作时也应使用相同的 sorted(self.uav_set.keys()) 顺序
        slot_uav_ids = all_uav_ids + [None] * (max_num_uav - len(all_uav_ids))
        id_to_slot = {uav_id: idx for idx, uav_id in enumerate(all_uav_ids)}

        active_uav_ids = [
            uav_id for uav_id in all_uav_ids
            if is_uav_active(uav_id)
        ]

        active_uav_id_set = set(active_uav_ids)

        # ---------- 构造无向邻接矩阵 ----------

        adjacency_matrix = [
            [0.0 for _ in range(max_num_uav)]
            for _ in range(max_num_uav)
        ]

        for src_id in all_uav_ids:
            if src_id not in active_uav_id_set:
                continue

            src_slot = id_to_slot[src_id]

            for dst_id in self.uav_net.get(src_id, set()):
                if dst_id not in id_to_slot:
                    continue
                if dst_id not in active_uav_id_set:
                    continue
                if dst_id == src_id:
                    continue

                dst_slot = id_to_slot[dst_id]

                # 连通性判断通常按无向图处理，所以这里也转成无向邻接矩阵
                adjacency_matrix[src_slot][dst_slot] = 1.0
                adjacency_matrix[dst_slot][src_slot] = 1.0

        # ---------- 静默计算连通子图，不调用 checkUavnetConnectivity，避免训练时频繁 print ----------

        visited = set()
        connected_components = []

        for start_id in active_uav_ids:
            if start_id in visited:
                continue

            component = set()
            queue = [start_id]
            visited.add(start_id)

            while queue:
                current_id = queue.pop(0)
                component.add(current_id)

                current_slot = id_to_slot[current_id]

                for other_id in active_uav_ids:
                    if other_id in visited:
                        continue

                    other_slot = id_to_slot[other_id]

                    if adjacency_matrix[current_slot][other_slot] > 0.5:
                        visited.add(other_id)
                        queue.append(other_id)

            connected_components.append(component)

        component_size_by_id = {}
        for component in connected_components:
            component_size = len(component)
            for uav_id in component:
                component_size_by_id[uav_id] = component_size

        num_active_uav = len(active_uav_ids)
        num_components = len(connected_components)
        largest_component_size = 0

        if connected_components:
            largest_component_size = max(len(c) for c in connected_components)

        is_connected = 1.0 if num_active_uav > 0 and num_components == 1 else 0.0

        # ---------- 全局网络特征 ----------

        edge_count = 0
        for i in range(max_num_uav):
            for j in range(i + 1, max_num_uav):
                if adjacency_matrix[i][j] > 0.5:
                    edge_count += 1

        max_possible_edges = num_active_uav * (num_active_uav - 1) / 2
        edge_density = safe_div(edge_count, max_possible_edges)

        avg_degree = safe_div(2 * edge_count, num_active_uav)

        global_features = [
            safe_div(num_active_uav, max_num_uav),
            safe_div(len(self.damaged_uav_ids), max_num_uav),
            safe_div(num_components, max_num_uav),
            safe_div(largest_component_size, max_num_uav),
            is_connected,
            edge_density,
            safe_div(avg_degree, max(max_num_uav - 1, 1)),
        ]

        # ---------- 节点特征 ----------

        node_features = []

        for slot_idx, uav_id in enumerate(slot_uav_ids):
            if uav_id is None:
                # padding UAV
                node_features.extend([
                    0.0,  # exists
                    0.0,  # active
                    0.0,  # x
                    0.0,  # y
                    0.0,  # z
                    0.0,  # energy
                    0.0,  # load
                    0.0,  # speed
                    0.0,  # communication radius
                    0.0,  # degree
                    0.0,  # component size
                ])
                continue

            uav = self.uav_set[uav_id]
            x, y, z = get_uav_position(uav)

            exists = 1.0
            active = 1.0 if uav_id in active_uav_id_set else 0.0

            degree = sum(adjacency_matrix[slot_idx])
            component_size = component_size_by_id.get(uav_id, 0)

            energy_norm = safe_div(
                getattr(uav, "uav_energy", 0.0),
                max(getattr(uav, "MAX_ENERGY", 1.0), 1.0)
            )

            load_norm = safe_div(
                getattr(uav, "uav_load", 0.0),
                max(getattr(uav, "MAX_LOAD", 1.0), 1.0)
            )

            speed_norm = safe_div(
                getattr(uav, "uav_speed", 0.0),
                max(max_speed, 1.0)
            )

            radius_norm = safe_div(
                getattr(uav, "communication_radius", 0.0),
                max(max_comm_radius, 1.0)
            )

            degree_norm = safe_div(
                degree,
                max(max_num_uav - 1, 1)
            )

            component_size_norm = safe_div(
                component_size,
                max_num_uav
            )

            node_features.extend([
                exists,
                active,
                norm_value(x, x_range),
                norm_value(y, y_range),
                norm_value(z, z_range),
                energy_norm,
                load_norm,
                speed_norm,
                radius_norm,
                degree_norm,
                component_size_norm,
            ])

        # ---------- 邻接矩阵展平 ----------

        adjacency_features = []
        for i in range(max_num_uav):
            for j in range(max_num_uav):
                adjacency_features.append(adjacency_matrix[i][j])

        # ---------- 距离矩阵展平，可选 ----------

        distance_features = []

        if include_distance_matrix:
            x_len = x_range[1] - x_range[0]
            y_len = y_range[1] - y_range[0]
            z_len = z_range[1] - z_range[0]
            max_distance = math.sqrt(x_len ** 2 + y_len ** 2 + z_len ** 2)

            for i in range(max_num_uav):
                uav_i_id = slot_uav_ids[i]

                for j in range(max_num_uav):
                    uav_j_id = slot_uav_ids[j]

                    if uav_i_id is None or uav_j_id is None:
                        distance_features.append(0.0)
                        continue

                    if uav_i_id not in active_uav_id_set or uav_j_id not in active_uav_id_set:
                        distance_features.append(0.0)
                        continue

                    uav_i = self.uav_set[uav_i_id]
                    uav_j = self.uav_set[uav_j_id]

                    xi, yi, zi = get_uav_position(uav_i)
                    xj, yj, zj = get_uav_position(uav_j)

                    dx = xi - xj
                    dy = yi - yj
                    dz = zi - zj

                    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    distance_features.append(safe_div(distance, max_distance))

        # ---------- 合并成最终 state ----------

        state = (
                global_features
                + node_features
                + adjacency_features
                + distance_features
        )

        if return_info:
            info = {
                "uav_order": all_uav_ids,
                "slot_uav_ids": slot_uav_ids,
                "num_active_uav": num_active_uav,
                "num_components": num_components,
                "largest_component_size": largest_component_size,
                "is_connected": bool(is_connected),
                "edge_count": edge_count,
                "state_dim": len(state),
                "node_feature_dim": 11,
                "global_feature_dim": len(global_features),
                "include_distance_matrix": include_distance_matrix,
            }
            return state, info

        return state

    def getAllState(self,
                    max_num_uav=None,
                    x_range=(0.0, 10000.0),
                    y_range=(0.0, 10000.0),
                    z_range=(0.0, 1000.0),
                    max_speed=1000.0,
                    max_comm_radius=2000.0,
                    include_distance_matrix=True,
                    auto_update_neighbors=False,
                    min_uav2uav_trans_rate=global_min_uav2uav_trans_rate):
        """
        返回当前 UAV 网络的完整结构化状态。

        返回:
            {
                "flat_state": flat_state,
                "global_features": global_features,
                "node_features": node_features,
                "adjacency_matrix": adjacency_matrix,
                "distance_matrix": distance_matrix,
            }

        其中:
            global_features: list[float]
                全局网络特征，形状为 [7]

            node_features: list[list[float]]
                每架 UAV 的节点特征，形状为 [max_num_uav, 11]

            adjacency_matrix: list[list[float]]
                UAV 网络邻接矩阵，形状为 [max_num_uav, max_num_uav]

            distance_matrix: list[list[float]]
                UAV 间距离矩阵，形状为 [max_num_uav, max_num_uav]

            flat_state: list[float]
                展平后的一维状态，可直接输入 MLP 强化学习网络
        """

        # 是否自动更新邻接关系
        if auto_update_neighbors:
            self.updateEachUavNeighbors(
                min_uav2uav_trans_rate=min_uav2uav_trans_rate
            )

        # ---------- 基本工具函数 ----------

        def safe_div(a, b):
            if b == 0:
                return 0.0
            return float(a) / float(b)

        def norm_value(v, value_range):
            low, high = value_range
            if high == low:
                return 0.0

            value = (float(v) - float(low)) / (float(high) - float(low))

            # 裁剪到 [0, 1]
            return max(0.0, min(1.0, value))

        def get_uav_position(uav):
            x, y, z = uav.uav_position
            return float(x), float(y), float(z)

        def is_uav_active(uav_id):
            """
            判断 UAV 是否仍然有效。
            damaged_uav_ids 中的 UAV 不参与连通性状态。
            """
            if uav_id not in self.uav_set:
                return False

            if uav_id in self.damaged_uav_ids:
                return False

            if self.remaining_uav_ids and uav_id not in self.remaining_uav_ids:
                return False

            uav = self.uav_set[uav_id]

            if hasattr(uav, "getIsActive"):
                return bool(uav.isActive())

            return bool(getattr(uav, "is_active", True))

        # ---------- UAV 顺序与最大数量 ----------

        all_uav_ids = sorted(self.uav_set.keys())

        if max_num_uav is None:
            max_num_uav = len(all_uav_ids)

        if len(all_uav_ids) > max_num_uav:
            raise ValueError(
                f"当前 UAV 数量为 {len(all_uav_ids)}，超过 max_num_uav={max_num_uav}。"
                f"增大 max_num_uav，或者保持训练场景中UAV数量固定。"
            )

        # 每个 UAV 的固定槽位
        slot_uav_ids = all_uav_ids + [None] * (max_num_uav - len(all_uav_ids))
        id_to_slot = {uav_id: idx for idx, uav_id in
                      enumerate(all_uav_ids)}  # UAV的ID，不一定连续、不一定由小到大，所以这里做了一个UAV的ID到索引的映射。

        active_uav_ids = [
            uav_id for uav_id in all_uav_ids
            if is_uav_active(uav_id)
        ]

        active_uav_id_set = set(active_uav_ids)

        # ---------- 构造无向邻接矩阵 ----------

        adjacency_matrix = [
            [0.0 for _ in range(max_num_uav)]
            for _ in range(max_num_uav)
        ]

        for src_id in all_uav_ids:
            if src_id not in active_uav_id_set:  # 非active的UAV，不计入。
                continue

            src_slot = id_to_slot[src_id]

            for dst_id in self.uav_net.get(src_id, set()):
                if dst_id not in id_to_slot:
                    continue
                if dst_id not in active_uav_id_set:
                    continue
                if dst_id == src_id:
                    continue

                dst_slot = id_to_slot[dst_id]

                # 按无向图处理
                adjacency_matrix[src_slot][dst_slot] = 1.0
                adjacency_matrix[dst_slot][src_slot] = 1.0

        # ---------- 静默计算连通子图 ----------
        # 只统计active节点间的连通子图
        visited = set()
        connected_components = []

        for start_id in active_uav_ids:
            if start_id in visited:
                continue

            component = set()
            queue = [start_id]
            visited.add(start_id)

            while queue:
                current_id = queue.pop(0)
                component.add(current_id)

                current_slot = id_to_slot[current_id]

                for other_id in active_uav_ids:
                    if other_id in visited:
                        continue

                    other_slot = id_to_slot[other_id]

                    if adjacency_matrix[current_slot][other_slot] > 0.5:
                        visited.add(other_id)
                        queue.append(other_id)

            connected_components.append(component)

        component_size_by_id = {}

        for component in connected_components:
            component_size = len(component)
            for uav_id in component:
                component_size_by_id[uav_id] = component_size

        num_active_uav = len(active_uav_ids)
        num_components = len(connected_components)

        if connected_components:
            largest_component_size = max(len(c) for c in connected_components)
        else:
            largest_component_size = 0

        is_connected = 1.0 if num_active_uav > 0 and num_components == 1 else 0.0

        # ---------- 全局网络特征 ----------

        edge_count = 0

        for i in range(max_num_uav):
            for j in range(i + 1, max_num_uav):
                if adjacency_matrix[i][j] > 0.5:
                    edge_count += 1

        max_possible_edges = num_active_uav * (num_active_uav - 1) / 2
        edge_density = safe_div(edge_count, max_possible_edges)
        avg_degree = safe_div(2 * edge_count, num_active_uav)

        global_features = [
            safe_div(num_active_uav, max_num_uav),
            safe_div(len(self.damaged_uav_ids), max_num_uav),
            safe_div(num_components, max_num_uav),
            safe_div(largest_component_size, max_num_uav),
            is_connected,
            edge_density,
            safe_div(avg_degree, max(max_num_uav - 1, 1)),
        ]

        # ---------- 节点特征矩阵 [max_num_uav, 11] ----------

        node_features = []

        for slot_idx, uav_id in enumerate(slot_uav_ids):
            if uav_id is None:
                # padding UAV
                node_features.append([
                    0.0,  # exists
                    0.0,  # active
                    0.0,  # x
                    0.0,  # y
                    0.0,  # z
                    0.0,  # energy
                    0.0,  # load
                    0.0,  # speed
                    0.0,  # communication radius
                    0.0,  # degree
                    0.0,  # component size
                ])
                continue

            uav = self.uav_set[uav_id]
            x, y, z = get_uav_position(uav)

            exists = 1.0
            active = 1.0 if uav_id in active_uav_id_set else 0.0

            degree = sum(adjacency_matrix[slot_idx])
            component_size = component_size_by_id.get(uav_id, 0)

            energy_norm = safe_div(
                getattr(uav, "uav_energy", 0.0),
                max(getattr(uav, "MAX_ENERGY", 1.0), 1.0)
            )

            load_norm = safe_div(
                getattr(uav, "uav_load", 0.0),
                max(getattr(uav, "MAX_LOAD", 1.0), 1.0)
            )

            speed_norm = safe_div(
                getattr(uav, "uav_speed", 0.0),
                max(max_speed, 1.0)
            )

            radius_norm = safe_div(
                getattr(uav, "communication_radius", 0.0),
                max(max_comm_radius, 1.0)
            )

            degree_norm = safe_div(
                degree,
                max(max_num_uav - 1, 1)
            )

            component_size_norm = safe_div(
                component_size,
                max_num_uav
            )

            node_features.append([
                exists,
                active,
                norm_value(x, x_range),
                norm_value(y, y_range),
                norm_value(z, z_range),
                energy_norm,
                load_norm,
                speed_norm,
                radius_norm,
                degree_norm,
                component_size_norm,
            ])

        # ---------- 距离矩阵 [max_num_uav, max_num_uav] ----------

        distance_matrix = [
            [0.0 for _ in range(max_num_uav)]
            for _ in range(max_num_uav)
        ]

        if include_distance_matrix:
            x_len = x_range[1] - x_range[0]
            y_len = y_range[1] - y_range[0]
            z_len = z_range[1] - z_range[0]
            max_distance = math.sqrt(x_len ** 2 + y_len ** 2 + z_len ** 2)

            for i in range(max_num_uav):
                uav_i_id = slot_uav_ids[i]

                for j in range(max_num_uav):
                    uav_j_id = slot_uav_ids[j]

                    if uav_i_id is None or uav_j_id is None:
                        distance_matrix[i][j] = 0.0
                        continue

                    if uav_i_id not in active_uav_id_set or uav_j_id not in active_uav_id_set:
                        distance_matrix[i][j] = 0.0
                        continue

                    uav_i = self.uav_set[uav_i_id]
                    uav_j = self.uav_set[uav_j_id]

                    xi, yi, zi = get_uav_position(uav_i)
                    xj, yj, zj = get_uav_position(uav_j)

                    dx = xi - xj
                    dy = yi - yj
                    dz = zi - zj

                    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    distance_matrix[i][j] = safe_div(distance, max_distance)

        # ---------- 构造 flat_state ----------

        flat_node_features = []
        for row in node_features:
            flat_node_features.extend(row)

        flat_adjacency_matrix = []
        for row in adjacency_matrix:
            flat_adjacency_matrix.extend(row)

        flat_distance_matrix = []
        for row in distance_matrix:
            flat_distance_matrix.extend(row)

        flat_state = (
                global_features
                + flat_node_features
                + flat_adjacency_matrix
                + flat_distance_matrix
        )

        return {
            "flat_state": flat_state,
            "global_features": global_features,
            "node_features": node_features,
            "adjacency_matrix": adjacency_matrix,  # 只考虑active的UAV间的邻接矩阵
            "distance_matrix": distance_matrix,
        }

    def getCoverageScore(self, coverage_radius=None):
        """
        根据 UAV 在 XY 平面上的圆形覆盖区域计算 [0, 1] 的覆盖分散度。

        1.0：任意两个 UAV 的覆盖范围都不重叠；
        越小：平均成对覆盖重叠越严重。

        注意：该量是一个客观环境指标，不再直接作为人工 reward。
        """
        metrics = self.getCoverageMetrics(coverage_radius=coverage_radius)
        return metrics["coverage_score"]

    def getCoverageMetrics(self, coverage_radius=None):
        """返回供评估、偏好标注和 Reward Model 使用的客观覆盖指标。"""
        if coverage_radius is None:
            coverage_radius = self.coverage_radius
        coverage_radius = float(coverage_radius)
        if coverage_radius <= 0:
            raise ValueError("coverage_radius must be positive.")

        active_uavs = [
            self.uav_set[uav_id]
            for uav_id in sorted(self.uav_set.keys())
            if self.uav_set[uav_id].is_active
        ]
        n = len(active_uavs)
        single_circle_area = math.pi * coverage_radius ** 2
        nominal_total_area = n * single_circle_area

        if n <= 1:
            return {
                "coverage_radius": coverage_radius,
                "num_active_uavs": n,
                "single_uav_area": single_circle_area,
                "nominal_total_area": nominal_total_area,
                "pairwise_overlap_area": 0.0,
                "average_pair_overlap_ratio": 0.0,
                "coverage_score": 1.0,
            }

        total_overlap = 0.0
        num_pairs = n * (n - 1) / 2.0

        for i in range(n):
            xi, yi, _ = active_uavs[i].uav_position
            for j in range(i + 1, n):
                xj, yj, _ = active_uavs[j].uav_position
                d = math.hypot(xi - xj, yi - yj)

                if d >= 2.0 * coverage_radius:
                    overlap_area = 0.0
                elif d <= 1e-12:
                    overlap_area = single_circle_area
                else:
                    value = max(-1.0, min(1.0, d / (2.0 * coverage_radius)))
                    overlap_area = (
                        2.0 * coverage_radius ** 2 * math.acos(value)
                        - 0.5 * d * math.sqrt(
                            max(0.0, 4.0 * coverage_radius ** 2 - d ** 2)
                        )
                    )
                total_overlap += overlap_area

        average_overlap_ratio = total_overlap / (num_pairs * single_circle_area)
        coverage_score = 1.0 - average_overlap_ratio
        coverage_score = max(0.0, min(1.0, coverage_score))

        return {
            "coverage_radius": coverage_radius,
            "num_active_uavs": n,
            "single_uav_area": single_circle_area,
            "nominal_total_area": nominal_total_area,
            # 这是所有 UAV 对的重叠面积之和；存在三重重叠时会重复计数，
            # 因而不把它错误命名成“真实 union area”。
            "pairwise_overlap_area": float(total_overlap),
            "average_pair_overlap_ratio": float(average_overlap_ratio),
            "coverage_score": float(coverage_score),
        }

    @staticmethod
    def _delta_to_tuple(delta):
        """兼容 torch.Tensor / numpy.ndarray / list 的 [3] 或 [1, 3] delta。"""
        if hasattr(delta, "detach"):
            delta = delta.detach()
        if hasattr(delta, "cpu"):
            delta = delta.cpu()
        if hasattr(delta, "numpy"):
            delta = delta.numpy()
        arr = np.asarray(delta, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"delta must contain exactly 3 values, got shape {arr.shape}.")
        return float(arr[0]), float(arr[1]), float(arr[2])

    def _get_undirected_edge_set(self):
        """以 ``(min_id, max_id)`` 的形式返回当前 active UAV 的无向边集合。"""
        active_ids = {
            uav_id for uav_id, uav in self.uav_set.items()
            if uav.is_active
        }
        edges = set()
        for src_id, neighbors in self.uav_net.items():
            if src_id not in active_ids:
                continue
            for dst_id in neighbors:
                if dst_id not in active_ids or dst_id == src_id:
                    continue
                edges.add(tuple(sorted((int(src_id), int(dst_id)))))
        return edges

    def _get_connected_components_silent(self, edges=None):
        """静默计算 active UAV 的无向连通分量，避免训练时频繁打印。"""
        active_ids = sorted(
            uav_id for uav_id, uav in self.uav_set.items()
            if uav.is_active
        )
        if not active_ids:
            return []

        if edges is None:
            edges = self._get_undirected_edge_set()

        adjacency = {uav_id: set() for uav_id in active_ids}
        for u, v in edges:
            if u in adjacency and v in adjacency:
                adjacency[u].add(v)
                adjacency[v].add(u)

        visited = set()
        components = []
        for start_id in active_ids:
            if start_id in visited:
                continue
            queue = [start_id]
            visited.add(start_id)
            component = set()
            while queue:
                current = queue.pop(0)
                component.add(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    def _get_min_inter_component_distance(self, components):
        """
        返回不同连通分量之间最近 UAV 对的三维距离以及“通信缺口”。
        gap = max(0, distance - min(comm_radius_i, comm_radius_j))。
        """
        if len(components) <= 1:
            return 0.0, 0.0, None

        best_distance = float("inf")
        best_gap = float("inf")
        best_pair = None

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                for u_id in components[i]:
                    for v_id in components[j]:
                        u = self.uav_set[u_id]
                        v = self.uav_set[v_id]
                        distance = u.countDistance(v.uav_position)
                        effective_radius = min(
                            float(u.communication_radius),
                            float(v.communication_radius),
                        )
                        gap = max(0.0, distance - effective_radius)
                        if distance < best_distance:
                            best_distance = float(distance)
                            best_gap = float(gap)
                            best_pair = (int(u_id), int(v_id))

        return best_distance, best_gap, best_pair

    def getNetworkMetrics(self, coverage_radius=None):
        """
        返回不带人工权重的客观网络指标。

        这些网络指标仍保留在环境中，便于兼容原有状态表示和调试；
        当前 sanity-check 的训练目标不再由这些指标决定。
        """
        edges = self._get_undirected_edge_set()
        components = self._get_connected_components_silent(edges)
        active_ids = sorted(
            uav_id for uav_id, uav in self.uav_set.items()
            if uav.is_active
        )

        num_active = len(active_ids)
        num_components = len(components)
        largest_component_size = max((len(c) for c in components), default=0)
        is_connected = bool(num_active > 0 and num_components == 1)
        edge_count = len(edges)
        max_edges = num_active * (num_active - 1) / 2.0
        edge_density = edge_count / max_edges if max_edges > 0 else 0.0
        avg_degree = (2.0 * edge_count / num_active) if num_active > 0 else 0.0

        min_inter_distance, min_inter_gap, closest_pair = (
            self._get_min_inter_component_distance(components)
        )
        coverage = self.getCoverageMetrics(coverage_radius=coverage_radius)

        return {
            "num_active_uavs": num_active,
            "num_damaged_uavs": int(self.num_Damaged_UAV),
            "num_components": num_components,
            "component_sizes": sorted([len(c) for c in components], reverse=True),
            "connected_components": [sorted(int(v) for v in c) for c in components],
            "largest_component_size": largest_component_size,
            "is_connected": is_connected,
            "edge_count": edge_count,
            "edge_density": float(edge_density),
            "average_degree": float(avg_degree),
            "edge_list": [list(edge) for edge in sorted(edges)],
            "min_inter_component_distance": float(min_inter_distance),
            "min_inter_component_gap": float(min_inter_gap),
            "closest_inter_component_pair": (
                list(closest_pair) if closest_pair is not None else None
            ),
            "coverage": coverage,
        }

    def getTargetMetrics(
        self,
        target_position=TARGET_POSITION,
        tolerance=TARGET_TOLERANCE,
    ):
        """
        计算当前 sanity-check 任务的客观目标指标。

        真正的优化目标始终是让每架 UAV 的三维坐标趋近 (0, 0, 0)。
        tolerance 只用于定义 episode 是否已经成功完成，不用于替代连续距离目标。
        """
        tx, ty, tz = (float(v) for v in target_position)
        tolerance = float(tolerance)

        distances = {}
        for uav_id in sorted(self.uav_set.keys()):
            uav = self.uav_set[uav_id]
            x, y, z = (float(v) for v in uav.uav_position)
            distance = math.sqrt(
                (x - tx) ** 2
                + (y - ty) ** 2
                + (z - tz) ** 2
            )
            distances[int(uav_id)] = float(distance)

        distance_values = list(distances.values())
        if distance_values:
            sum_distance = float(sum(distance_values))
            mean_distance = float(sum_distance / len(distance_values))
            max_distance = float(max(distance_values))
        else:
            sum_distance = 0.0
            mean_distance = 0.0
            max_distance = 0.0

        num_within_tolerance = sum(
            int(distance <= tolerance) for distance in distance_values
        )
        all_reached = bool(
            distance_values
            and num_within_tolerance == len(distance_values)
        )

        return {
            "target_position": [tx, ty, tz],
            "tolerance": tolerance,
            "distances": distances,
            "sum_distance": sum_distance,
            "mean_distance": mean_distance,
            "max_distance": max_distance,
            "num_within_tolerance": int(num_within_tolerance),
            "all_reached": all_reached,
        }

    def getPreferenceSnapshot(self, coverage_radius=None):
        """
        生成适合序列化为 JSON 并输入 LLM 的结构化快照。

        network 字段为兼容原环境而保留；target 字段才是当前 sanity-check
        偏好判断和固定验证使用的任务指标。
        """
        metrics = self.getNetworkMetrics(coverage_radius=coverage_radius)
        target_metrics = self.getTargetMetrics()
        components = metrics["connected_components"]
        component_index = {}
        for idx, component in enumerate(components):
            for uav_id in component:
                component_index[int(uav_id)] = idx

        uavs = []
        for slot, uav_id in enumerate(sorted(self.uav_set.keys())):
            uav = self.uav_set[uav_id]
            neighbors = sorted(
                int(v) for v in self.uav_net.get(uav_id, set())
                if v in self.uav_set and self.uav_set[v].is_active
            )
            uavs.append({
                "slot": int(slot),
                "uav_id": int(uav_id),
                "active": bool(uav.is_active),
                "position": [float(v) for v in uav.uav_position],
                "distance_to_target": float(target_metrics["distances"][int(uav_id)]),
                "energy": float(uav.uav_energy),
                "load": float(uav.uav_load),
                "communication_radius": float(uav.communication_radius),
                "degree": len(neighbors),
                "neighbors": neighbors,
                "component_index": component_index.get(int(uav_id)),
            })

        return {
            "network": metrics,
            "target": target_metrics,
            "uavs": uavs,
        }

    def step(
        self,
        uav_slot,
        delta,
        max_num_uav=None,
        terminate_on_connected=False,
        terminate_on_target=False,
        time_step_counter=None,
    ):
        """
        执行 Hybrid SAC 动作，但不直接产生人工 reward。

        当前 sanity-check 任务使用 target 指标描述动作前后的客观变化；
        LLM 根据 trajectory preference 监督 Reward Model，SAC 再使用 Reward Model reward。

        Parameters
        ----------
        uav_slot:
            Actor 输出的离散槽位。槽位映射为
            ``sorted(self.uav_set.keys())[uav_slot]``。
        delta:
            Actor 输出的实际三维位移 (dx, dy, dz)。
        max_num_uav:
            用于构造 next_state 的固定最大 UAV 数；训练时建议始终传 8。
        terminate_on_connected:
            兼容旧任务的终止开关，当前 sanity-check 保持 False。
        terminate_on_target:
            若为 True，当所有 UAV 均进入 TARGET_TOLERANCE 范围时终止 episode。
        time_step_counter:
            仅写入 info 供调试/LLM 使用，不参与奖励计算。

        Returns
        -------
        next_state, done, info
        """
        all_uav_ids = sorted(self.uav_set.keys())
        uav_slot = int(uav_slot)
        if not 0 <= uav_slot < len(all_uav_ids):
            raise ValueError(
                f"uav_slot={uav_slot} is invalid for {len(all_uav_ids)} UAVs."
            )

        moving_uav_id = all_uav_ids[uav_slot]
        moving_uav = self.uav_set[moving_uav_id]
        if not moving_uav.is_active:
            raise ValueError(
                f"Actor selected inactive UAV slot={uav_slot}, id={moving_uav_id}."
            )

        command_delta = self._delta_to_tuple(delta)
        before_snapshot = self.getPreferenceSnapshot()
        before_edges = {
            tuple(edge) for edge in before_snapshot["network"]["edge_list"]
        }
        old_position = tuple(float(v) for v in moving_uav.uav_position)

        new_position, executed_delta = self.moveUAVByDelta(
            uavID=moving_uav_id,
            delta=command_delta,
        )

        after_snapshot = self.getPreferenceSnapshot()
        after_edges = {
            tuple(edge) for edge in after_snapshot["network"]["edge_list"]
        }

        new_edges = sorted(after_edges - before_edges)
        broken_edges = sorted(before_edges - after_edges)
        before_network = before_snapshot["network"]
        after_network = after_snapshot["network"]
        before_cov = before_network["coverage"]
        after_cov = after_network["coverage"]
        before_target = before_snapshot["target"]
        after_target = after_snapshot["target"]

        info = {
            "time_step": None if time_step_counter is None else int(time_step_counter),
            "action": {
                "uav_slot": uav_slot,
                "uav_id": int(moving_uav_id),
                "command_delta": [float(v) for v in command_delta],
                "executed_delta": [float(v) for v in executed_delta],
                "position_before": [float(v) for v in old_position],
                "position_after": [float(v) for v in new_position],
            },
            "before": before_snapshot,
            "after": after_snapshot,
            "changes": {
                # 当前 sanity-check 真正关心的目标变化。
                # delta < 0 表示相应距离指标变小，即朝正确方向进步。
                "sum_distance_to_target_delta": (
                    float(after_target["sum_distance"])
                    - float(before_target["sum_distance"])
                ),
                "mean_distance_to_target_delta": (
                    float(after_target["mean_distance"])
                    - float(before_target["mean_distance"])
                ),
                "max_distance_to_target_delta": (
                    float(after_target["max_distance"])
                    - float(before_target["max_distance"])
                ),
                "num_within_tolerance_delta": (
                    int(after_target["num_within_tolerance"])
                    - int(before_target["num_within_tolerance"])
                ),
                "all_reached": bool(after_target["all_reached"]),

                # 以下网络/覆盖变化仅为兼容与调试保留，不参与当前 LLM 偏好目标。
                "num_components_delta": (
                    int(after_network["num_components"])
                    - int(before_network["num_components"])
                ),
                "largest_component_size_delta": (
                    int(after_network["largest_component_size"])
                    - int(before_network["largest_component_size"])
                ),
                "edge_count_delta": (
                    int(after_network["edge_count"])
                    - int(before_network["edge_count"])
                ),
                "new_edges": [list(edge) for edge in new_edges],
                "broken_edges": [list(edge) for edge in broken_edges],
                "min_inter_component_distance_delta": (
                    float(after_network["min_inter_component_distance"])
                    - float(before_network["min_inter_component_distance"])
                ),
                "min_inter_component_gap_delta": (
                    float(after_network["min_inter_component_gap"])
                    - float(before_network["min_inter_component_gap"])
                ),
                "coverage_score_delta": (
                    float(after_cov["coverage_score"])
                    - float(before_cov["coverage_score"])
                ),
                "pairwise_overlap_area_delta": (
                    float(after_cov["pairwise_overlap_area"])
                    - float(before_cov["pairwise_overlap_area"])
                ),
                "became_connected": bool(
                    (not before_network["is_connected"])
                    and after_network["is_connected"]
                ),
                "lost_connectivity": bool(
                    before_network["is_connected"]
                    and (not after_network["is_connected"])
                ),
            },
        }

        if max_num_uav is None:
            max_num_uav = len(all_uav_ids)
        next_state = self.getState(max_num_uav=max_num_uav)

        no_active_uav = not any(self.getSelectAbleMask())
        done = bool(
            no_active_uav
            or (terminate_on_connected and after_network["is_connected"])
            or (terminate_on_target and after_target["all_reached"])
        )
        return next_state, done, info

    def draw_env_2D(self, positions, iteration=1, time_step=1):
        fig, ax = plt.subplots(figsize=(8, 8))

        x_list = []
        y_list = []

        for key in positions.keys():
            if key != 7:
                x_list.append(positions[key][0])
                y_list.append(positions[key][1])

        ax.scatter(
            x_list,
            y_list,
            color="dodgerblue",
            s=150,
            marker="o",
            edgecolors="black",
        )

        x_list.clear()
        y_list.clear()

        for key in positions.keys():
            if key == 7:
                x_list.append(positions[key][0])
                y_list.append(positions[key][1])

        ax.scatter(
            x_list,
            y_list,
            color="red",
            s=150,
            marker="o",
            edgecolors="black",
        )

        # for index, (x, y, z) in enumerate(positions.values()):
        #     ax.text(
        #         x,
        #         y + 5.0,
        #         f"UAV {index}\nz={z:.1f} m",
        #         ha="center",
        #         fontsize=9,
        #     )

        # ax.set_xlabel("X coordinate (m)")
        # ax.set_ylabel("Y coordinate (m)")
        ax.set_title(f"UAV PositionsTop View: ITE {iteration:03d} time step {time_step:02d}")

        # 设置坐标轴范围
        ax.set_xlim(-500, 4500)
        ax.set_ylim(-500, 4500)

        # 设置主刻度间隔
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_major_locator(MultipleLocator(500))

        # 设置次刻度间隔
        # ax.xaxis.set_minor_locator(MultipleLocator(5))
        # ax.yaxis.set_minor_locator(MultipleLocator(5))

        # 主网格和次网格
        ax.grid(True, which="major", linewidth=0.8)
        # ax.grid(True, which="minor", linewidth=0.4, linestyle="--")

        # 使 x、y 方向单位长度相同
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(f"./R_images/env_2D_ITE{iteration:04d}_TIM{time_step:03d}.jpeg", dpi=100)
        plt.close()
        # plt.show()



