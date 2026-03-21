# offline_online_dt_single_file.py
# Offline + Online Decision Transformer (single-file)
# - CORL-style DT backbone
# - online-dt style stochastic policy head (optional)
# - online-dt style trajectory ReplayBuffer container (no separate trajectory store class)

import os
import math
import uuid
import random
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import gym
import d4rl  # noqa: F401
import pyrallis
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # logging
    project: str = "CORL"
    group: str = "DT-Off2On"
    name: str = "DT-OfflineOnline"

    # env
    env_name: str = "hopper-medium-v2"
    device: str = "cuda"
    train_seed: int = 0
    eval_seed: int = 42
    deterministic_torch: bool = False

    # model (CORL DT style)
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0

    # stochastic policy extension (online-dt style head)
    stochastic_policy: bool = True
    log_std_bounds_min: float = -5.0
    log_std_bounds_max: float = 2.0
    init_temperature: float = 0.1
    target_entropy: Optional[float] = None
    learn_temperature: bool = True
    policy_entropy_bonus: float = 0.0

    # optimization
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    warmup_steps: int = 10_000
    updates_per_iter: int = 1

    # offline + online schedule
    offline_iterations: int = 100_000
    online_iterations: int = 100_000
    reward_scale: float = 0.001

    # RTG semantics (DT usually uses undiscounted RTG => gamma=1.0)
    rtg_gamma: float = 1.0

    # trajectory replay buffer (trajectory-level)
    traj_buffer_capacity: int = 5000  # capacity in number of trajectories

    # action sampling behavior
    online_sample_actions: bool = True
    eval_sample_actions: bool = False

    # evaluation
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_every: int = 10_000
    eval_episodes: int = 10

    # checkpoint
    checkpoints_path: Optional[str] = None
    save_every: int = 50_000
    load_model: str = ""

    # normalization
    normalize_states: bool = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# ============================================================
# Utils
# ============================================================

ENVS_WITH_GOAL_PREFIX = ("antmaze", "pen", "door", "hammer", "relocate")


def wandb_init_if_needed(config: dict):
    wandb.init(project=config["project"], group=config["group"], name=config["name"], config=config)


def wandb_log_if_needed(data: dict, step: Optional[int] = None):
    wandb.log(data, step=step)


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if env is not None:
        env.seed(seed=seed)

    try:
        torch.use_deterministic_algorithms(deterministic_torch)
    except Exception:
        pass


def reset_env(env, seed: Optional[int] = None):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return next_obs, reward, done, info, terminated, truncated
    else:
        next_obs, reward, done, info = out
        return next_obs, reward, done, info, done, False


def is_goal_env(env_name: str) -> bool:
    return env_name.lower().startswith(ENVS_WITH_GOAL_PREFIX)


def is_goal_reached(reward, info) -> bool:
    if isinstance(info, dict):
        if "goal_achieved" in info:
            return bool(info["goal_achieved"])
        if "is_success" in info:
            return bool(info["is_success"])
    return reward > 0.0


def discounted_cumsum(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    if len(x) == 0:
        return y
    y[-1] = x[-1]
    for t in reversed(range(len(x) - 1)):
        y[t] = x[t] + gamma * y[t + 1]
    return y.astype(np.float32)


def pad_along_axis(arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, pad_size)
    return np.pad(arr, pad_width, mode="constant", constant_values=fill_value)


class TransformRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


# class TransformObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env, mean, std):
#         super().__init__(env)
#         self.mean = mean
#         self.std = std

#     def observation(self, observation):
#         return (observation - self.mean) / self.std

class TransformObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, mean, std):
        super().__init__(env)
        # force 1D so wrapped env outputs stay shape [state_dim], not [1, state_dim]
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        return (observation - self.mean) / self.std


def wrap_env(env, state_mean=None, state_std=None, reward_scale=1.0):
    if state_mean is not None and state_std is not None:
        env = TransformObservationWrapper(env, state_mean, state_std)
    if reward_scale != 1.0:
        env = TransformRewardWrapper(env, reward_scale)
    return env


# ============================================================
# D4RL dataset -> trajectories
# ============================================================

def qlearning_to_trajectories(dataset: Dict[str, np.ndarray], gamma: float = 1.0):
    """
    Convert D4RL qlearning dataset into trajectory list for DT.
    Each trajectory dict contains observations/actions/rewards/returns (RTG).
    """
    trajectories = []
    data_ = defaultdict(list)

    N = dataset["rewards"].shape[0]
    has_timeouts = "timeouts" in dataset

    for i in range(N):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        done = bool(dataset["terminals"][i])
        timeout = bool(dataset["timeouts"][i]) if has_timeouts else False

        if done or timeout:
            traj = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            if len(traj["rewards"]) > 0:
                traj["returns"] = discounted_cumsum(traj["rewards"], gamma=gamma)
                trajectories.append(traj)
            data_ = defaultdict(list)

    # flush tail
    if len(data_["rewards"]) > 0:
        traj = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
        traj["returns"] = discounted_cumsum(traj["rewards"], gamma=gamma)
        trajectories.append(traj)

    obs = dataset["observations"].astype(np.float32)
    info = {
        "obs_mean": obs.mean(0, keepdims=True).astype(np.float32),
        "obs_std": (obs.std(0, keepdims=True) + 1e-6).astype(np.float32),
    }
    return trajectories, info


# ============================================================
# online-dt style ReplayBuffer (trajectory container)
# ============================================================

class ReplayBuffer(object):
    """
    Trajectory-level replay buffer inspired by facebookresearch/online-dt replay_buffer.py
    - fixed capacity in # trajectories
    - if initial trajectories exceed capacity, keep top-return trajectories
    - add_new_trajs supports online appending
    """
    def __init__(self, capacity, trajectories=None):
        if trajectories is None:
            trajectories = []
        self.capacity = int(capacity)

        if len(trajectories) <= self.capacity:
            self.trajectories = list(trajectories)
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # ascending
            self.trajectories = [trajectories[ii] for ii in sorted_inds[-self.capacity :]]

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        if len(new_trajs) == 0:
            return

        # We usually add one trajectory at a time in online loop (safe with slice overwrite).
        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]
        else:
            self.trajectories[self.start_idx : self.start_idx + len(new_trajs)] = new_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        assert len(self.trajectories) <= self.capacity


# ============================================================
# DT batch sampling from ReplayBuffer (no TrajectoryStore class)
# ============================================================

def sample_batch_from_replay_buffer(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    seq_len: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    reward_scale: float,
    device: str,
    rtg_gamma: float = 1.0,
):
    assert len(replay_buffer) > 0, "ReplayBuffer is empty"

    # length-weighted trajectory sampling
    traj_lens = np.array(
        [max(1, traj["actions"].shape[0]) for traj in replay_buffer.trajectories],
        dtype=np.float64,
    )
    p = traj_lens / traj_lens.sum()

    batch = []
    for _ in range(batch_size):
        traj_idx = np.random.choice(len(replay_buffer.trajectories), p=p)
        traj = replay_buffer.trajectories[traj_idx]
        T = traj["rewards"].shape[0]
        start_idx = random.randint(0, max(0, T - 1))

        states = traj["observations"][start_idx : start_idx + seq_len]
        actions = traj["actions"][start_idx : start_idx + seq_len]

        if "returns" in traj:
            returns = traj["returns"][start_idx : start_idx + seq_len]
        else:
            rtg_full = discounted_cumsum(traj["rewards"], gamma=rtg_gamma)
            returns = rtg_full[start_idx : start_idx + seq_len]

        time_steps = np.arange(start_idx, start_idx + states.shape[0], dtype=np.int64)

        # normalize states and scale rewards/RTG as in CORL DT style
        states = (states - state_mean) / state_std
        returns = returns * reward_scale

        mask = np.hstack([
            np.ones(states.shape[0], dtype=np.float32),
            np.zeros(seq_len - states.shape[0], dtype=np.float32),
        ])

        if states.shape[0] < seq_len:
            states = pad_along_axis(states, seq_len, axis=0)
            actions = pad_along_axis(actions, seq_len, axis=0)
            returns = pad_along_axis(returns, seq_len, axis=0)

            if time_steps.shape[0] < seq_len:
                pad_ts = np.zeros((seq_len - time_steps.shape[0],), dtype=np.int64)
                time_steps = np.concatenate([time_steps, pad_ts], axis=0)

        batch.append((
            states.astype(np.float32),
            actions.astype(np.float32),
            returns.astype(np.float32),
            time_steps.astype(np.int64),
            mask.astype(np.float32),
        ))

    states, actions, returns, time_steps, mask = map(np.stack, zip(*batch))
    return (
        torch.tensor(states, device=device, dtype=torch.float32),
        torch.tensor(actions, device=device, dtype=torch.float32),
        torch.tensor(returns, device=device, dtype=torch.float32),
        torch.tensor(time_steps, device=device, dtype=torch.long),
        torch.tensor(mask, device=device, dtype=torch.float32),
    )


# ============================================================
# online-dt style stochastic policy components
# ============================================================

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        y = y.clamp(-0.999999, 0.999999)
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        base_dist = pyd.Normal(loc, std)
        super().__init__(base_dist, [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        x = self.rsample((N,))
        log_p = self.log_prob(x)  # [N, B, T, A]
        return -log_p.mean(dim=0).sum(dim=-1)  # [B, T]

    def log_likelihood(self, x):
        return self.log_prob(x).sum(dim=-1)  # [B, T]


class DiagGaussianActor(nn.Module):
    """
    Stochastic policy head over DT hidden states (online-dt style).
    Input:  [B, T, D]
    Output: SquashedNormal over [-1,1]^A
    """
    def __init__(self, hidden_dim, act_dim, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds
        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h):
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.tanh(log_std)
        low, high = self.log_std_bounds
        log_std = low + 0.5 * (high - low) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)


# ============================================================
# Transformer block (CORL style)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        self.register_buffer("causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).bool())

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        y = self.norm1(x)
        y = self.attn(
            query=y,
            key=y,
            value=y,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        x = x + self.drop(y)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Decision Transformer (keep CORL style; only action head is extended)
# ============================================================

class DecisionTransformer(nn.Module):
    """
    CORL-style Decision Transformer with minimal stochastic-policy extension:
    - backbone/tokenization/forward indexing stays CORL-style
    - only action head becomes optional stochastic policy head
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 20,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        max_action: float = 1.0,
        # stochastic-policy extension only
        stochastic_policy: bool = False,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        init_temperature: float = 0.1,
        target_entropy: Optional[float] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        self.stochastic_policy = stochastic_policy

        # === CORL-style embeddings ===
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.out_norm = nn.LayerNorm(embedding_dim)

        # === CORL-style blocks ===
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # === only this part differs ===
        if self.stochastic_policy:
            self.action_head = DiagGaussianActor(
                hidden_dim=embedding_dim,
                act_dim=action_dim,
                log_std_bounds=log_std_bounds,
            )
            self.log_temperature = nn.Parameter(
                torch.tensor(np.log(init_temperature), dtype=torch.float32)
            )
            self.target_entropy = -float(action_dim) if target_entropy is None else float(target_entropy)
        else:
            self.action_head = nn.Sequential(
                nn.Linear(embedding_dim, action_dim),
                nn.Tanh(),
            )
            self.register_parameter("log_temperature", None)
            self.target_entropy = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def temperature(self):
        if self.log_temperature is None:
            return None
        return self.log_temperature.exp()

    def _forward_impl(
        self,
        states: torch.Tensor,         # [B, T, state_dim]
        actions: torch.Tensor,        # [B, T, action_dim]
        returns_to_go: torch.Tensor,  # [B, T]
        time_steps: torch.Tensor,     # [B, T]
        padding_mask: Optional[torch.Tensor] = None,  # [B, T], True means PAD
    ) -> torch.Tensor:
        B, T = states.shape[0], states.shape[1]

        max_t = self.timestep_emb.num_embeddings - 1
        time_steps = time_steps.clamp(min=0, max=max_t)

        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        action_emb = self.action_emb(actions) + time_emb
        return_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # CORL/original token order: (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        x = (
            torch.stack([return_emb, state_emb, action_emb], dim=1)   # [B, 3, T, D]
            .permute(0, 2, 1, 3)                                      # [B, T, 3, D]
            .reshape(B, 3 * T, self.embedding_dim)                    # [B, 3T, D]
        )

        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(B, 3 * T)
            )

        x = self.emb_norm(x)
        x = self.emb_drop(x)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        x = self.out_norm(x)

        # action predicted from state token hidden states (positions 1,4,7,...)
        h = x[:, 1::3]  # [B, T, D]
        return h

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,  # only used when stochastic_policy=True
    ):
        h = self._forward_impl(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        if self.stochastic_policy:
            dist = self.action_head(h)  # squashed Gaussian in [-1,1]
            pred_actions = dist.mean if deterministic else dist.rsample()
            pred_actions = pred_actions * self.max_action
            return pred_actions, dist
        else:
            pred_actions = self.action_head(h) * self.max_action
            return pred_actions

    @torch.no_grad()
    def act_from_history(
        self,
        states_hist: np.ndarray,     # [L, state_dim], normalized if env wrapped
        actions_hist: np.ndarray,    # [L, action_dim], last slot can be dummy zeros
        returns_hist: np.ndarray,    # [L]
        timesteps_hist: np.ndarray,  # [L]
        device: str = "cpu",
        sample: bool = True,
    ) -> np.ndarray:
        states = torch.tensor(states_hist[None], dtype=torch.float32, device=device)[:, -self.seq_len :]
        actions = torch.tensor(actions_hist[None], dtype=torch.float32, device=device)[:, -self.seq_len :]
        returns = torch.tensor(returns_hist[None], dtype=torch.float32, device=device)[:, -self.seq_len :]
        timesteps = torch.tensor(timesteps_hist[None], dtype=torch.long, device=device)[:, -self.seq_len :]

        if self.stochastic_policy:
            pred_actions, _ = self.forward(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=timesteps,
                padding_mask=None,
                deterministic=not sample,
            )
        else:
            pred_actions = self.forward(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=timesteps,
                padding_mask=None,
            )

        action = pred_actions[0, -1]
        return action.clamp(-self.max_action, self.max_action).cpu().numpy()


# ============================================================
# Trainer
# ============================================================

class DTTrainer:
    def __init__(
        self,
        model: DecisionTransformer,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
        warmup_steps: int,
        clip_grad: Optional[float],
        learn_temperature: bool = True,
        policy_entropy_bonus: float = 0.0,
    ):
        self.model = model
        self.clip_grad = clip_grad
        self.total_it = 0
        self.learn_temperature = bool(learn_temperature and model.stochastic_policy)
        self.policy_entropy_bonus = float(policy_entropy_bonus)

        self.optim = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lambda step: min((step + 1) / max(1, warmup_steps), 1.0)
        )
        self.temp_optim = torch.optim.Adam([model.log_temperature], lr=lr) if self.learn_temperature else None

    def train_step(self, batch):
        self.model.train()
        self.total_it += 1

        states, actions, returns, time_steps, mask = batch
        padding_mask = ~mask.bool()  # True => PAD ignore

        log = {}

        if self.model.stochastic_policy:
            pred_actions, dist = self.model(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_mask,
                deterministic=False,
            )

            # dist is over normalized action range [-1,1], so rescale target
            target_actions = (actions.detach() / self.model.max_action).clamp(-1.0, 1.0)

            logp = dist.log_likelihood(target_actions)  # [B, T]
            actor_loss_tok = -logp

            if self.policy_entropy_bonus != 0.0:
                ent = dist.entropy()  # [B, T]
                actor_loss_tok = actor_loss_tok - self.policy_entropy_bonus * ent
                log["policy_entropy"] = float((ent * mask).sum().item() / (mask.sum().item() + 1e-8))

            actor_loss = (actor_loss_tok * mask).sum() / (mask.sum() + 1e-8)

            self.optim.zero_grad()
            actor_loss.backward(retain_graph=self.learn_temperature)
            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optim.step()
            self.scheduler.step()

            log["train_loss"] = float(actor_loss.item())
            log["nll_loss"] = float(actor_loss.item())

            mse = F.mse_loss(pred_actions, actions.detach(), reduction="none")
            mse = (mse * mask.unsqueeze(-1)).sum() / ((mask.sum() * actions.shape[-1]) + 1e-8)
            log["train_mse"] = float(mse.item())

            if self.learn_temperature:
                # detach entropy for temperature update (standard alpha update style)
                ent = dist.entropy().detach()
                temp_loss_tok = self.model.temperature() * ((-ent) - self.model.target_entropy)
                temp_loss = (temp_loss_tok * mask).sum() / (mask.sum() + 1e-8)

                self.temp_optim.zero_grad()
                temp_loss.backward()
                self.temp_optim.step()

                log["temp_loss"] = float(temp_loss.item())
                log["temperature"] = float(self.model.temperature().item())
            else:
                log["temperature"] = float(self.model.temperature().item())

        else:
            pred_actions = self.model(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_mask,
            )
            loss = F.mse_loss(pred_actions, actions.detach(), reduction="none")
            loss = (loss * mask.unsqueeze(-1)).sum() / ((mask.sum() * actions.shape[-1]) + 1e-8)

            self.optim.zero_grad()
            loss.backward()
            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optim.step()
            self.scheduler.step()

            log["train_loss"] = float(loss.item())
            log["train_mse"] = float(loss.item())

        log["learning_rate"] = float(self.scheduler.get_last_lr()[0])
        return log

    def state_dict(self):
        out = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "total_it": self.total_it,
        }
        if self.temp_optim is not None:
            out["temp_optim"] = self.temp_optim.state_dict()
        return out

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])
        self.optim.load_state_dict(sd["optim"])
        self.scheduler.load_state_dict(sd["scheduler"])
        self.total_it = sd.get("total_it", 0)
        if self.temp_optim is not None and "temp_optim" in sd:
            self.temp_optim.load_state_dict(sd["temp_optim"])


# ============================================================
# Online episode builder
# ============================================================

class OnlineEpisodeBuilder:
    def __init__(self, rtg_gamma: float = 1.0):
        self.rtg_gamma = rtg_gamma
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []

    def add_transition(self, obs, action, reward):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(np.float32(reward))

    def to_traj(self):
        traj = {
            "observations": np.array(self.obs, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
        }
        traj["returns"] = discounted_cumsum(traj["rewards"], gamma=self.rtg_gamma)
        return traj


# ============================================================
# Evaluation rollout (autoregressive DT)
# ============================================================

@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    rtg_gamma: float = 1.0,
    device: str = "cpu",
    sample_actions: bool = False,
):
    obs = reset_env(env)

    states_hist_list = []
    actions_hist_list = []
    rewards_hist_list = []

    ep_return = 0.0
    ep_len = 0
    success = False

    for _ in range(model.episode_len):
        states_hist_list.append(np.array(obs, dtype=np.float32))  # already normalized if env wrapped

        states_hist = np.array(states_hist_list, dtype=np.float32)
        actions_arr = (
            np.array(actions_hist_list, dtype=np.float32)
            if len(actions_hist_list) > 0
            else np.zeros((0, model.action_dim), dtype=np.float32)
        )
        if actions_arr.shape[0] < states_hist.shape[0]:
            actions_arr = np.concatenate(
                [actions_arr, np.zeros((1, model.action_dim), dtype=np.float32)], axis=0
            )

        # Build RTG history with same gamma semantics
        if rtg_gamma == 1.0:
            rtg_list = [target_return]
            for r in rewards_hist_list:
                rtg_list.append(rtg_list[-1] - float(r))
        else:
            rtg_list = [target_return]
            for r in rewards_hist_list:
                if abs(rtg_gamma) < 1e-8:
                    rtg_list.append(0.0)
                else:
                    rtg_list.append((rtg_list[-1] - float(r)) / rtg_gamma)
        returns_hist = np.array(rtg_list, dtype=np.float32)

        timesteps_hist = np.arange(states_hist.shape[0], dtype=np.int64)

        action = model.act_from_history(
            states_hist=states_hist,
            actions_hist=actions_arr,
            returns_hist=returns_hist,
            timesteps_hist=timesteps_hist,
            device=device,
            sample=(sample_actions and model.stochastic_policy),
        )

        next_obs, reward, done, info, _, _ = step_env(env, action)
        actions_hist_list.append(action.astype(np.float32))
        rewards_hist_list.append(np.float32(reward))

        if not success:
            success = is_goal_reached(reward, info)

        ep_return += reward
        ep_len += 1
        obs = next_obs
        if done:
            break

    return float(ep_return), int(ep_len), bool(success)


# ============================================================
# Main train
# ============================================================

def train_impl(config: TrainConfig):
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)
    set_seed(config.train_seed, env=env, deterministic_torch=config.deterministic_torch)
    set_seed(config.eval_seed, env=eval_env, deterministic_torch=config.deterministic_torch)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps = getattr(env, "_max_episode_steps", config.episode_len)
    env_with_goal = is_goal_env(config.env_name)

    # Offline dataset -> trajectories (DT usually uses undiscounted RTG => gamma=1.0)
    qdataset = d4rl.qlearning_dataset(env)
    offline_trajs, ds_info = qlearning_to_trajectories(qdataset, gamma=config.rtg_gamma)

    if config.normalize_states:
        state_mean, state_std = ds_info["obs_mean"], ds_info["obs_std"]
    else:
        state_mean = np.zeros((1, state_dim), dtype=np.float32)
        state_std = np.ones((1, state_dim), dtype=np.float32)

    # Wrap envs: normalize observations, scale rewards
    env = wrap_env(env, state_mean=state_mean, state_std=state_std, reward_scale=config.reward_scale)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std, reward_scale=config.reward_scale)

    # Trajectory container: online-dt style ReplayBuffer
    traj_buffer = ReplayBuffer(
        capacity=config.traj_buffer_capacity,
        trajectories=offline_trajs,
    )

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=max_action,
        stochastic_policy=config.stochastic_policy,
        log_std_bounds=(config.log_std_bounds_min, config.log_std_bounds_max),
        init_temperature=config.init_temperature,
        target_entropy=config.target_entropy,
    ).to(config.device)

    trainer = DTTrainer(
        model=model,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        warmup_steps=config.warmup_steps,
        clip_grad=config.clip_grad,
        learn_temperature=config.learn_temperature,
        policy_entropy_bonus=config.policy_entropy_bonus,
    )

    if config.load_model:
        ckpt = torch.load(config.load_model, map_location=config.device)
        trainer.load_state_dict(ckpt["trainer"] if "trainer" in ckpt else ckpt)

    if config.checkpoints_path is not None:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print("-" * 90)
    print(f"Training Offline+Online DT | env={config.env_name}")
    print(f"Offline trajectories: {len(offline_trajs)}")
    print(f"Trajectory replay buffer capacity: {config.traj_buffer_capacity}")
    print(f"Replay buffer current size: {len(traj_buffer)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 90)

    wandb_init_if_needed(asdict(config))

    # Online rollout state
    state = reset_env(env, seed=config.train_seed)
    episode_return = 0.0  # scaled return (because env wrapper scales reward)
    episode_step = 0
    goal_achieved = False
    train_success_flags = []
    ep_builder = OnlineEpisodeBuilder(rtg_gamma=config.rtg_gamma)

    total_iters = int(config.offline_iterations) + int(config.online_iterations)

    for t in range(total_iters):
        if t == config.offline_iterations:
            print("Switching to online finetuning ...")

        online_log = {}

        # ------------------------------------------------------
        # Online collection (after offline pretraining iterations)
        # ------------------------------------------------------
        if t >= config.offline_iterations:
            # current partial episode history is already normalized by wrapped env
            curr_states = ep_builder.obs + [np.array(state, dtype=np.float32)]
            states_hist = np.array(curr_states, dtype=np.float32)

            actions_hist = (
                np.array(ep_builder.actions, dtype=np.float32)
                if len(ep_builder.actions) > 0
                else np.zeros((0, action_dim), dtype=np.float32)
            )
            if actions_hist.shape[0] < states_hist.shape[0]:
                actions_hist = np.concatenate(
                    [actions_hist, np.zeros((1, action_dim), dtype=np.float32)], axis=0
                )

            # build current desired RTG history (scaled RTG because rewards are scaled in wrapped env)
            if config.rtg_gamma == 1.0:
                rtg_hist = [config.target_returns[0] * config.reward_scale]
                for r in ep_builder.rewards:
                    rtg_hist.append(rtg_hist[-1] - float(r))
            else:
                rtg_hist = [config.target_returns[0] * config.reward_scale]
                for r in ep_builder.rewards:
                    rtg_hist.append((rtg_hist[-1] - float(r)) / config.rtg_gamma)
            returns_hist = np.array(rtg_hist, dtype=np.float32)

            timesteps_hist = np.arange(states_hist.shape[0], dtype=np.int64)

            action = model.act_from_history(
                states_hist=states_hist,
                actions_hist=actions_hist,
                returns_hist=returns_hist,
                timesteps_hist=timesteps_hist,
                device=config.device,
                sample=(config.online_sample_actions and config.stochastic_policy),
            )

            next_state, reward, done, info, _, _ = step_env(env, action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)

            # `state` is normalized because env is wrapped
            ep_builder.add_transition(state, action, reward)
            episode_return += reward
            episode_step += 1
            state = next_state

            timeout = episode_step >= max_steps
            if done or timeout:
                new_traj = ep_builder.to_traj()
                traj_buffer.add_new_trajs([new_traj])  # one trajectory at a time

                if env_with_goal:
                    train_success_flags.append(float(goal_achieved))
                    online_log["train/is_success"] = float(goal_achieved)
                    online_log["train/regret"] = float(
                        np.mean(1.0 - np.array(train_success_flags, dtype=np.float32))
                    )

                # unscale to raw return for D4RL normalized score logging
                unscaled_ret = episode_return / max(config.reward_scale, 1e-8)
                try:
                    norm_ret = float(eval_env.get_normalized_score(unscaled_ret) * 100.0)
                except Exception:
                    norm_ret = float("nan")

                online_log["train/episode_return"] = float(unscaled_ret)
                online_log["train/d4rl_normalized_episode_return"] = norm_ret
                online_log["train/episode_length"] = int(episode_step)
                online_log["train/traj_buffer_size"] = int(len(traj_buffer))

                # reset episode
                state = reset_env(env)
                episode_return = 0.0
                episode_step = 0
                goal_achieved = False
                ep_builder.reset()

        # ------------------------------------------------------
        # Training updates (single sampler from trajectory replay buffer)
        # ------------------------------------------------------
        train_log = {}
        for _ in range(config.updates_per_iter):
            batch = sample_batch_from_replay_buffer(
                replay_buffer=traj_buffer,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                state_mean=state_mean,
                state_std=state_std,
                reward_scale=config.reward_scale,
                device=config.device,
                rtg_gamma=config.rtg_gamma,
            )
            train_log = trainer.train_step(batch)

        if t < config.offline_iterations:
            train_log["offline_iter"] = t
        else:
            train_log["online_iter"] = t - config.offline_iterations

        train_log.update(online_log)
        wandb_log_if_needed(train_log, step=trainer.total_it)

        # ------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------
        if ((t + 1) % config.eval_every == 0) or (t == total_iters - 1):
            model.eval()
            eval_log = {}

            print(f"[Iter {t + 1}/{total_iters}]")
            for target_return in config.target_returns:
                eval_returns = []
                eval_lens = []
                eval_success = []

                for ep in range(config.eval_episodes):
                    try:
                        reset_env(eval_env, seed=config.eval_seed + ep)
                    except Exception:
                        pass

                    ep_ret_scaled, ep_len, ep_succ = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        rtg_gamma=config.rtg_gamma,
                        device=config.device,
                        sample_actions=(config.eval_sample_actions and config.stochastic_policy),
                    )
                    ep_ret = ep_ret_scaled / max(config.reward_scale, 1e-8)

                    eval_returns.append(ep_ret)
                    eval_lens.append(ep_len)
                    eval_success.append(ep_succ)

                eval_returns_np = np.array(eval_returns, dtype=np.float32)
                try:
                    normalized = eval_env.get_normalized_score(eval_returns_np) * 100.0
                    norm_mean = float(np.nanmean(normalized))
                    norm_std = float(np.nanstd(normalized))
                except Exception:
                    norm_mean, norm_std = float("nan"), float("nan")

                eval_log[f"eval/{target_return}_return_mean"] = float(eval_returns_np.mean())
                eval_log[f"eval/{target_return}_return_std"] = float(eval_returns_np.std())
                eval_log[f"eval/{target_return}_episode_len_mean"] = float(np.mean(eval_lens))
                eval_log[f"eval/{target_return}_normalized_score_mean"] = norm_mean
                eval_log[f"eval/{target_return}_normalized_score_std"] = norm_std

                msg = (
                    f"  target={target_return:.1f} | "
                    f"return={eval_returns_np.mean():.2f} | d4rl={norm_mean:.2f}"
                )

                if env_with_goal:
                    succ_rate = float(np.mean(eval_success))
                    eval_log["eval/success_rate"] = succ_rate
                    msg += f" | success={succ_rate:.3f}"
                    if len(train_success_flags) > 0:
                        eval_log["eval/regret"] = float(
                            np.mean(1.0 - np.array(train_success_flags, dtype=np.float32))
                        )

                print(msg)

            wandb_log_if_needed(eval_log, step=trainer.total_it)

            if config.checkpoints_path is not None and (
                ((t + 1) % config.save_every == 0) or (t == total_iters - 1)
            ):
                ckpt = {
                    "trainer": trainer.state_dict(),
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "config": asdict(config),
                }
                torch.save(ckpt, os.path.join(config.checkpoints_path, f"checkpoint_{t+1}.pt"))

            model.train()


# ============================================================
# Entrypoint
# ============================================================

@pyrallis.wrap()
def main(config: TrainConfig):
    train_impl(config)


if __name__ == "__main__":
    main()
