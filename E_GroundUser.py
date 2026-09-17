import math


class GroundUser:
    def __init__(self,
                 userID=0,
                 userPosition=(0, 0, 0),
                 userBandwidth=0,
                 userTraffic=0,
                 userWaitingTime=0,
                 userTransPower=0,
                 servedByUAV=9999):
        self.userID = userID
        self.user_position = userPosition
        self.user_bandwidth = userBandwidth
        self.user_traffic = userTraffic
        self.user_waiting_time = userWaitingTime
        self.user_TransPower = userTransPower
        self.user_TransRate = 0
        self.served_by_UAV = servedByUAV

    def get_uav2userTransRate(self, uavID, uav_position,
                              fc=2e9,
                              c=3e8,
                              n0_dbm_per_hz=-174,
                              n0=3.98e-21,
                              eta_L=0.1,
                              eta_N=20,
                              a=4.88,
                              b=0.429):
        if uavID != self.served_by_UAV:
            print(
                f"\033[91m GroundUser {self.userID} served by UAV {self.served_by_UAV}, but denote served by {uavID}.\033[0m")
            self.user_TransRate = 0
            return 0
        if self.user_bandwidth == 0:
            print(f"\033[91m GroundUser has no bandwidth. \033[0m")
            return 0
        dx = uav_position[0] - self.user_position[0]
        dy = uav_position[1] - self.user_position[1]
        dz = uav_position[2] - self.user_position[2]
        horizontal_distance = math.sqrt(dx ** 2 + dy ** 2)
        d_3d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # 地面用户和无人机间的仰角，
        if horizontal_distance == 0:
            theta_rad = math.pi / 2
        else:
            theta_rad = math.atan(abs(dz) / horizontal_distance)

        theta_deg = math.degrees(theta_rad)
        p_L = 1 / (1 + a * math.exp(-b * (theta_deg - a)))
        p_N = 1 - p_L

        # 公式(7)：LoS / NLoS 路径损耗，单位 dB
        phi_spl_db = 20 * math.log10(4 * math.pi * fc * d_3d / c)
        phi_LoS_db = phi_spl_db + eta_L
        phi_NLoS_db = phi_spl_db + eta_N
        phi_LoS_linear = 10 ** (phi_LoS_db / 10)
        phi_NLoS_linear = 10 ** (phi_NLoS_db / 10)
        phi_avg_linear = p_L * phi_LoS_linear + p_N * phi_NLoS_linear
        sinr = self.user_TransPower / (phi_avg_linear * n0 * self.user_bandwidth)
        self.user_TransRate = self.user_bandwidth * math.log2(1 + sinr)
        return self.user_TransRate

    def infoState(self):
        print(f"\033[96m [GroundUser STATE] GroundUser{self.userID} Position{self.user_position} ServedByUAV{self.served_by_UAV} "
              f"TransRate{self.user_TransRate}\033[0m")
