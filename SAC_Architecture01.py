import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


DEFAULT_MAX_DELTA = (20.0, 20.0, 20.0)


class HybridSACActor(nn.Module):
    """
    Hybrid SAC Actor.

    动作由两部分组成：
        1. uav_slot：离散动作，选择哪一架 UAV；
        2. delta=(dx, dy, dz)：连续动作，表示本 decision step 的实际位移。

    注意：
        uav_slot 不是原始 UAV ID，而是 sorted(uav_set.keys()) 中的槽位。
        环境执行时由 Monitor.step() 映射到真实 UAV ID。

    与新版 E_Monitor.py 配合时，Actor 输出的 delta 会直接作为实际位移执行，
    不再被 UAV speed * moving_duration 二次截断，从而消除“不同动作得到同一环境结果”
    的动作别名问题。
    """

    def __init__(
        self,
        state_dim: int,
        num_uav: int,
        hidden_dim: int = 256,
        max_delta=DEFAULT_MAX_DELTA,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.num_uav = int(num_uav)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.register_buffer(
            "delta_scale",
            torch.tensor(max_delta, dtype=torch.float32),
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.uav_logits_layer = nn.Linear(self.hidden_dim, self.num_uav)
        self.delta_mu_layer = nn.Linear(self.hidden_dim, self.num_uav * 3)
        self.delta_log_std_layer = nn.Linear(self.hidden_dim, self.num_uav * 3)

    def _prepare_select_able_mask(self, select_able_mask, logits):
        if select_able_mask is None:
            return None

        mask = torch.as_tensor(
            select_able_mask,
            dtype=torch.bool,
            device=logits.device,
        )

        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        if mask.shape[-1] != self.num_uav:
            raise ValueError(
                f"select_able_mask last dimension should be {self.num_uav}, "
                f"but got {tuple(mask.shape)}."
            )

        if mask.shape[0] == 1 and logits.shape[0] > 1:
            mask = mask.expand(logits.shape[0], -1)

        if mask.shape != logits.shape:
            raise ValueError(
                f"select_able_mask shape {tuple(mask.shape)} is incompatible with "
                f"logits shape {tuple(logits.shape)}."
            )

        if (~mask).all(dim=-1).any():
            raise ValueError("At least one sample has no active UAV available.")

        return mask

    def forward(self, state, select_able_mask=None):
        if state.ndim == 1:
            state = state.unsqueeze(0)

        h = self.encoder(state)
        uav_logits = self.uav_logits_layer(h)

        mask = self._prepare_select_able_mask(select_able_mask, uav_logits)
        if mask is not None:
            uav_logits = uav_logits.masked_fill(~mask, -1e9)

        delta_mu = self.delta_mu_layer(h).view(-1, self.num_uav, 3)
        delta_log_std = self.delta_log_std_layer(h).view(-1, self.num_uav, 3)
        delta_log_std = torch.clamp(
            delta_log_std,
            self.log_std_min,
            self.log_std_max,
        )

        return uav_logits, delta_mu, delta_log_std

    def sample_delta(self, mu, log_std):
        """
        tanh-squashed Gaussian 连续动作采样。
        mu log_std的形状是[B, 3]
        每个坐标轴都被限制在 [-delta_scale, +delta_scale]。
        """
        std = log_std.exp()
        normal = Normal(mu, std)
        raw_action = normal.rsample()  # 形状[B, 3]
        tanh_action = torch.tanh(raw_action)  # 形状[B, 3]
        delta = tanh_action * self.delta_scale  # 形状[B, 3]

        log_prob = normal.log_prob(raw_action)  # 形状[B, 3]
        log_prob = log_prob - torch.log(  # 形状[B, 3]
            self.delta_scale * (1.0 - tanh_action.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1) # 形状[B]
        return delta, log_prob

    def sample(self, state, select_able_mask=None):
        """采样一个完整混合动作。"""
        uav_logits, delta_mu, delta_log_std = self.forward(  # 形状分别是[B, NUM_UAV] [B, NUM_UAV, 3] [B, NUM_UAV, 3]
            state,
            select_able_mask=select_able_mask,
        )

        uav_dist = Categorical(logits=uav_logits)
        uav_slot = uav_dist.sample()  # 形状是[B]
        log_prob_uav = uav_dist.log_prob(uav_slot)

        gather_index = uav_slot.view(-1, 1, 1).expand(-1, 1, 3)  # 形状[B, 1, 3]
        selected_mu = delta_mu.gather(1, gather_index).squeeze(1)  # 形状[B, 3], gather_index和delta_mu的维度数必须一样
        selected_log_std = delta_log_std.gather(1, gather_index).squeeze(1)  # 形状[B, 3]

        delta, log_prob_delta = self.sample_delta(  # 形状[B ,3]
            selected_mu,
            selected_log_std,
        )

        log_prob = log_prob_uav + log_prob_delta
        action = torch.cat(  # 形状是[B, 4]
            [uav_slot.float().unsqueeze(-1), delta],
            dim=-1,
        )
        return action, uav_slot, delta, log_prob

    def deterministic_action(self, state, select_able_mask=None):
        """验证/测试时使用：离散动作取 argmax，连续动作取均值。"""
        uav_logits, delta_mu, _ = self.forward(
            state,
            select_able_mask=select_able_mask,
        )

        uav_slot = torch.argmax(uav_logits, dim=-1)
        gather_index = uav_slot.view(-1, 1, 1).expand(-1, 1, 3)
        selected_mu = delta_mu.gather(1, gather_index).squeeze(1)
        delta = torch.tanh(selected_mu) * self.delta_scale

        action = torch.cat(
            [uav_slot.float().unsqueeze(-1), delta],
            dim=-1,
        )
        return action, uav_slot, delta

    def sample_all(self, state, select_able_mask=None):
        """
        为每个离散 UAV 选择分别采样一组连续位移。

        用于 Hybrid SAC 的 soft value 与 Actor loss：
            V(s) = sum_d pi(d|s) [Q(s,d,c_d) - alpha log pi(d,c_d|s)]
        """
        uav_logits, delta_mu, delta_log_std = self.forward(
            state,
            select_able_mask=select_able_mask,
        )

        uav_probs = torch.softmax(uav_logits, dim=-1)  # 形状[B, NUM_UAV]
        uav_log_probs = F.log_softmax(uav_logits, dim=-1)  # 形状[B, NUM_UAV]
        delta_all, delta_log_probs = self.sample_delta(  # 形状[B, NUM_UAV, 3]
            delta_mu,
            delta_log_std,
        )
        total_log_prob = uav_log_probs + delta_log_probs
        return uav_probs, total_log_prob, delta_all


class HybridSACCritic(nn.Module):
    """
    Twin-Q SAC 中的单个 Q 网络。

    输入：state, uav_slot, delta
    输出：Q(state, uav_slot, delta)
    """

    def __init__(
        self,
        state_dim: int,
        num_uav: int,
        hidden_dim: int = 256,
        uav_embed_dim: int = 32,
        max_delta=DEFAULT_MAX_DELTA,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.num_uav = int(num_uav)
        self.hidden_dim = int(hidden_dim)
        self.uav_embed_dim = int(uav_embed_dim)

        self.register_buffer(
            "delta_scale",
            torch.tensor(max_delta, dtype=torch.float32),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.uav_embedding = nn.Embedding(self.num_uav, self.uav_embed_dim)

        self.q_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.uav_embed_dim + 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, state, uav_slot, delta):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        if uav_slot.ndim == 0:
            uav_slot = uav_slot.unsqueeze(0)

        state_feature = self.state_encoder(state)
        uav_feature = self.uav_embedding(uav_slot.long())
        delta_norm = delta / self.delta_scale

        x = torch.cat(
            [state_feature, uav_feature, delta_norm],
            dim=-1,
        )
        return self.q_net(x).squeeze(-1)

    def q_all_uavs(self, state, delta_all):
        """对所有 UAV slot 分别计算 Q。"""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if delta_all.ndim != 3:
            raise ValueError(
                f"delta_all should have [batch, num_uav, 3], got {tuple(delta_all.shape)}"
            )

        batch_size = state.shape[0]
        device = state.device

        state_feature = self.state_encoder(state)
        state_feature = state_feature.unsqueeze(1).expand(
            batch_size,
            self.num_uav,
            self.hidden_dim,
        )

        uav_slots = torch.arange(
            self.num_uav,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, self.num_uav)
        uav_feature = self.uav_embedding(uav_slots)

        delta_norm = delta_all / self.delta_scale
        x = torch.cat(
            [state_feature, uav_feature, delta_norm],
            dim=-1,
        )
        x = x.reshape(batch_size * self.num_uav, -1)
        return self.q_net(x).view(batch_size, self.num_uav)
