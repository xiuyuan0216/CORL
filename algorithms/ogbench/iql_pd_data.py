# file: algorithms/finetune/iql_policy_decorator_ogbench_mixed.py

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ogbench
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

TensorBatch = List[torch.Tensor]

# ============================================================
# Utils
# ============================================================

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state, env.observation_space)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.action_space.seed(seed)


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


# ============================================================
# IQL modules
# ============================================================

EXP_ADV_MAX = 100.0
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    device: str = "cuda"
    env: str = "cube-double-play-singletask-task2-v0"
    seed: int = 0
    eval_seed: int = 42
    online_steps: int = int(1e6)
    eval_freq: int = int(5e4)
    n_episodes: int = 10
    checkpoints_path: Optional[str] = None

    iql_checkpoint: str = ""
    strict_iql_load: bool = True
    base_iql_deterministic: bool = True

    iql_deterministic_actor: bool = False
    actor_dropout: float = 0.0
    hidden_dim: int = 256
    n_hidden: int = 2

    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7

    normalize: bool = True
    normalize_reward: bool = False

    buffer_size: int = 2_000_000
    batch_size: int = 256
    learning_starts: int = 5000
    training_freq: int = 1
    utd: float = 1.0
    gamma: float = 0.99

    # NEW
    offline_mixing_ratio: float = 0.5
    min_offline_per_batch: int = 0

    actor_input: str = "obs"   # ['obs', 'obs_base_action']
    critic_input: str = "sum"  # ['res', 'sum', 'concat']
    res_scale: float = 0.1
    policy_frequency: int = 2
    target_network_frequency: int = 1
    q_lr: float = 3e-4
    policy_lr: float = 3e-4
    sac_alpha: float = 0.2
    autotune: bool = True
    max_grad_norm: float = 50.0
    alpha_init: float = None

    progressive_residual: bool = False
    prog_explore: int = int(1e5)
    critic_layer_norm: bool = False

    project: str = "CORL"
    group: str = "IQL-PolicyDecorator-ogbench"
    name: str = "IQL-PD-ogbench"
    log_wandb: bool = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# ============================================================
# Replay buffer
# Only store current env action a_t
# ============================================================

class DecoratorReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self._device)

    def add(self, obs, next_obs, action, reward, done):
        i = self._pointer
        self._states[i] = self._to_tensor(obs)
        self._next_states[i] = self._to_tensor(next_obs)
        self._actions[i] = self._to_tensor(action)
        self._rewards[i] = self._to_tensor([reward])
        self._dones[i] = self._to_tensor([done])

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._size, size=batch_size)
        return SimpleNamespace(
            observations=self._states[idx],
            next_observations=self._next_states[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            dones=self._dones[idx],
        )

    def size(self):
        return self._size


# ============================================================
# Policy Decorator modules
# ============================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, env, critic_layer_norm: bool = False):
        super().__init__()
        in_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)

        if critic_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                layer_init(nn.Linear(256, 1), std=0.01),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                layer_init(nn.Linear(256, 1), std=0.01),
            )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        input_dim = obs_dim if args.actor_input == "obs" else obs_dim + np.prod(env.single_action_space.shape)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(256, np.prod(env.single_action_space.shape)), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(256, np.prod(env.single_action_space.shape)), std=0.01)

        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


# ============================================================
# Utilities
# ============================================================

def make_dummy_env_for_pd(obs_dim: int, act_space: gym.Space, critic_input: str):
    class DummyObject:
        pass

    dummy_env_actor = DummyObject()
    dummy_env_actor.single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    dummy_env_actor.single_action_space = act_space

    dummy_env_critic = DummyObject()
    dummy_env_critic.single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    act_dim = int(np.prod(act_space.shape))
    q_action_dim = act_dim * 2 if critic_input == "concat" else act_dim
    dummy_env_critic.single_action_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(q_action_dim,), dtype=np.float32
    )

    return dummy_env_actor, dummy_env_critic


def build_iql_agent_for_loading(
    cfg: TrainConfig,
    state_dim: int,
    action_dim: int,
    max_action: float,
    device: str,
):
    if cfg.iql_deterministic_actor:
        iql_actor = DeterministicPolicy(
            state_dim=state_dim,
            act_dim=action_dim,
            max_action=max_action,
            hidden_dim=cfg.hidden_dim,
            n_hidden=cfg.n_hidden,
            dropout=cfg.actor_dropout,
        ).to(device)
    else:
        iql_actor = GaussianPolicy(
            state_dim=state_dim,
            act_dim=action_dim,
            max_action=max_action,
            hidden_dim=cfg.hidden_dim,
            n_hidden=cfg.n_hidden,
            dropout=cfg.actor_dropout,
        ).to(device)

    iql_qf = TwinQ(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
    ).to(device)

    iql_vf = ValueFunction(
        state_dim=state_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
    ).to(device)

    iql_actor_optimizer = torch.optim.Adam(iql_actor.parameters(), lr=cfg.actor_lr)
    iql_q_optimizer = torch.optim.Adam(iql_qf.parameters(), lr=cfg.qf_lr)
    iql_v_optimizer = torch.optim.Adam(iql_vf.parameters(), lr=cfg.vf_lr)

    iql = ImplicitQLearning(
        max_action=max_action,
        actor=iql_actor,
        actor_optimizer=iql_actor_optimizer,
        q_network=iql_qf,
        q_optimizer=iql_q_optimizer,
        v_network=iql_vf,
        v_optimizer=iql_v_optimizer,
        iql_tau=cfg.iql_tau,
        beta=cfg.beta,
        max_steps=max(1, cfg.online_steps),
        discount=cfg.gamma,
        tau=cfg.tau,
        device=device,
    )
    return iql


@torch.no_grad()
def get_base_action_tensor(iql: ImplicitQLearning, obs_tensor: torch.Tensor, max_action: float):
    policy_out = iql.actor(obs_tensor)
    if isinstance(policy_out, torch.distributions.Distribution):
        act = policy_out.mean
        act = torch.clamp(max_action * act, -max_action, max_action)
    else:
        act = torch.clamp(policy_out * max_action, -max_action, max_action)
    return act


@torch.no_grad()
def get_base_actions_numpy(
    iql: ImplicitQLearning,
    observations: np.ndarray,
    device: str,
    max_action: float,
    batch_size: int = 8192,
):
    outs = []
    for start in range(0, len(observations), batch_size):
        end = start + batch_size
        obs_t = torch.tensor(observations[start:end], dtype=torch.float32, device=device)
        act_t = get_base_action_tensor(iql, obs_t, max_action=max_action)
        outs.append(act_t.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_ogbench(
    env: gym.Env,
    iql: ImplicitQLearning,
    res_actor: Actor,
    cfg: TrainConfig,
    device: str,
    n_episodes: int,
) -> Dict[str, float]:
    episode_rewards = []
    successes = []

    res_actor.eval()
    iql.actor.eval()

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            base_a_np = iql.actor.act(obs, device=device)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            base_a_t = torch.tensor(base_a_np, dtype=torch.float32, device=device).unsqueeze(0)

            if cfg.actor_input == "obs":
                actor_in = obs_t
            else:
                actor_in = torch.cat([obs_t, base_a_t], dim=1)

            res_a_t = res_actor.get_eval_action(actor_in)
            final_a = base_a_t + cfg.res_scale * res_a_t
            final_a = final_a.squeeze(0).detach().cpu().numpy()
            final_a = np.clip(final_a, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(final_a)
            done = terminated | truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        successes.append(float(info["success"]))

    res_actor.train()
    return np.asarray(episode_rewards), float(np.mean(successes))


def build_offline_replay_buffer_from_dataset(
    dataset: Dict[str, np.ndarray],
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
    action_dim: int,
    device: str,
):
    offline_rb = DecoratorReplayBuffer(
        state_dim=dataset["observations"].shape[1],
        action_dim=action_dim,
        buffer_size=len(dataset["observations"]),
        device=device,
    )

    obs_np = dataset["observations"].astype(np.float32)
    next_obs_np = dataset["next_observations"].astype(np.float32)
    act_np = dataset["actions"].astype(np.float32)
    rew_np = dataset["rewards"].astype(np.float32)
    done_np = dataset["terminals"].astype(np.float32)

    if isinstance(state_mean, np.ndarray):
        obs_np = normalize_states(obs_np, state_mean, state_std)
        next_obs_np = normalize_states(next_obs_np, state_mean, state_std)

    for i in range(len(obs_np)):
        offline_rb.add(
            obs_np[i],
            next_obs_np[i],
            act_np[i],
            float(rew_np[i]),
            float(done_np[i]),
        )

    return offline_rb


def concat_samples(samples: List[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        observations=torch.cat([s.observations for s in samples], dim=0),
        next_observations=torch.cat([s.next_observations for s in samples], dim=0),
        actions=torch.cat([s.actions for s in samples], dim=0),
        rewards=torch.cat([s.rewards for s in samples], dim=0),
        dones=torch.cat([s.dones for s in samples], dim=0),
    )


def sample_mixed_batch(
    offline_rb: DecoratorReplayBuffer,
    online_rb: DecoratorReplayBuffer,
    batch_size: int,
    offline_mixing_ratio: float,
    min_offline_per_batch: int = 0,
) -> SimpleNamespace:
    offline_bs = int(batch_size * offline_mixing_ratio)
    # offline_bs = max(offline_bs, min_offline_per_batch)
    # offline_bs = min(offline_bs, batch_size)
    online_bs = batch_size - offline_bs

    parts = []
    if offline_bs > 0:
        parts.append(offline_rb.sample(offline_bs))
    if online_bs > 0:
        if online_rb.size() <= 0:
            raise RuntimeError("Trying to sample online data, but online replay buffer is empty.")
        parts.append(online_rb.sample(online_bs))

    batch = concat_samples(parts)
    perm = torch.randperm(batch.observations.shape[0], device=batch.observations.device)

    return SimpleNamespace(
        observations=batch.observations[perm],
        next_observations=batch.next_observations[perm],
        actions=batch.actions[perm],
        rewards=batch.rewards[perm],
        dones=batch.dones[perm],
    )


# ============================================================
# Main training
# ============================================================

@pyrallis.wrap()
def train(cfg: TrainConfig):
    assert cfg.iql_checkpoint != "", "--iql_checkpoint is required"

    device = cfg.device if torch.cuda.is_available() else "cpu"

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(cfg.env)
    eval_env = ogbench.make_env_and_datasets(cfg.env, env_only=True)

    max_steps = env._max_episode_steps

    set_seed(cfg.seed, env)
    set_seed(cfg.eval_seed, eval_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    reward_mod_dict = {}
    if cfg.normalize_reward:
        reward_mod_dict = modify_reward(train_dataset, cfg.env, max_episode_steps=max_steps)

    if cfg.normalize:
        state_mean, state_std = compute_mean_std(train_dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    env = wrap_env(env, state_mean=state_mean, state_std=state_std, reward_scale=1.0)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std, reward_scale=1.0)

    iql = build_iql_agent_for_loading(cfg, state_dim, action_dim, max_action, device)
    ckpt = torch.load(cfg.iql_checkpoint, map_location=device)
    iql.load_state_dict(ckpt)

    for p in iql.qf.parameters():
        p.requires_grad_(False)
    for p in iql.vf.parameters():
        p.requires_grad_(False)
    for p in iql.actor.parameters():
        p.requires_grad_(False)

    if cfg.base_iql_deterministic:
        iql.actor.eval()

    dummy_env_actor, dummy_env_critic = make_dummy_env_for_pd(
        obs_dim=state_dim,
        act_space=env.action_space,
        critic_input=cfg.critic_input,
    )
    pd_args = SimpleNamespace(actor_input=cfg.actor_input)

    res_actor = Actor(dummy_env_actor, pd_args).to(device)
    qf1 = SoftQNetwork(dummy_env_critic, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf2 = SoftQNetwork(dummy_env_critic, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf1_target = SoftQNetwork(dummy_env_critic, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf2_target = SoftQNetwork(dummy_env_critic, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.q_lr)
    actor_optimizer = optim.Adam(list(res_actor.parameters()), lr=cfg.policy_lr)

    if cfg.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        if cfg.alpha_init is not None:
            log_sac_alpha = torch.tensor(cfg.alpha_init, requires_grad=True, device=device)
        else:
            log_sac_alpha = torch.zeros(1, requires_grad=True, device=device)
        sac_alpha = log_sac_alpha.exp().item()
        a_optimizer = optim.Adam([log_sac_alpha], lr=cfg.q_lr)
    else:
        target_entropy = None
        log_sac_alpha = None
        a_optimizer = None
        sac_alpha = cfg.sac_alpha

    rb = DecoratorReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=cfg.buffer_size,
        device=device,
    )

    offline_rb = build_offline_replay_buffer_from_dataset(
        dataset=train_dataset,
        state_mean=state_mean,
        state_std=state_std,
        action_dim=action_dim,
        device=device,
    )

    if cfg.log_wandb:
        wandb.init(project=cfg.project, group=cfg.group, name=cfg.name, config=asdict(cfg))

    if cfg.checkpoints_path is not None:
        Path(cfg.checkpoints_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.checkpoints_path, "config.txt"), "w") as f:
            f.write(str(cfg))

    obs, _ = env.reset()
    episode_return = 0.0
    episode_len = 0
    global_step = 0
    global_update = 0
    done = False

    num_updates_per_training = max(1, int(cfg.training_freq * cfg.utd))
    eval_successes = []

    print(f"Offline buffer size: {offline_rb.size()}")
    print(f"Online buffer size: {rb.size()}")

    while global_step < cfg.online_steps:
        for _ in range(cfg.training_freq):
            if global_step >= cfg.online_steps:
                break
            global_step += 1

            base_action = iql.actor.act(obs, device=device)

            if cfg.progressive_residual:
                res_ratio = min(global_step / max(1, cfg.prog_explore), 1.0)
                enable_res = (np.random.rand() < res_ratio)
            else:
                enable_res = True

            if rb.size() < cfg.learning_starts:
                res_action = np.zeros_like(base_action, dtype=np.float32)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                base_t = torch.tensor(base_action, dtype=torch.float32, device=device).unsqueeze(0)
                if cfg.actor_input == "obs":
                    actor_input = obs_t
                else:
                    actor_input = torch.cat([obs_t, base_t], dim=1)

                res_action_t, _, _ = res_actor.get_action(actor_input)
                res_action = res_action_t.squeeze(0).detach().cpu().numpy()
                if not enable_res:
                    res_action[...] = 0.0

            final_action = base_action + cfg.res_scale * res_action
            final_action = np.clip(final_action, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(final_action)
            done = terminated | truncated
            episode_return += reward

            real_done = float(info["success"])

            if cfg.normalize_reward:
                reward = modify_reward_online(reward, cfg.env, **reward_mod_dict)

            # only store current env action
            rb.add(obs, next_obs, final_action, float(reward), real_done)

            obs = next_obs
            episode_len += 1

            if done:
                log_ep = {
                    "train/episode_return": episode_return,
                    "train/episode_len": episode_len,
                    "train/is_success": float(info["success"]),
                    "global_step": global_step,
                }

                if cfg.log_wandb:
                    wandb.log(log_ep, step=global_step)
                else:
                    print(log_ep)

                obs, _ = env.reset()
                episode_len = 0
                episode_return = 0.0
                done = False

        if rb.size() < cfg.learning_starts:
            continue

        for _ in range(num_updates_per_training):
            global_update += 1

            data = sample_mixed_batch(
                offline_rb=offline_rb,
                online_rb=rb,
                batch_size=cfg.batch_size,
                offline_mixing_ratio=cfg.offline_mixing_ratio,
                min_offline_per_batch=cfg.min_offline_per_batch,
            )

            current_actions = data.actions

            with torch.no_grad():
                base_actions = get_base_action_tensor(iql, data.observations, max_action=max_action)
                base_next_actions = get_base_action_tensor(iql, data.next_observations, max_action=max_action)

            if cfg.res_scale <= 0:
                raise ValueError("cfg.res_scale must be > 0")

            res_actions = (current_actions - base_actions) / cfg.res_scale

            # --------------------------
            # Q update
            # --------------------------
            with torch.no_grad():
                if cfg.actor_input == "obs":
                    actor_input_next = data.next_observations
                else:
                    actor_input_next = torch.cat([data.next_observations, base_next_actions], dim=1)

                next_state_res_actions, next_state_log_pi, _ = res_actor.get_action(actor_input_next)

                if cfg.critic_input == "res":
                    next_state_actions = next_state_res_actions
                elif cfg.critic_input == "sum":
                    next_state_actions = base_next_actions + cfg.res_scale * next_state_res_actions
                else:
                    next_state_actions = torch.cat([next_state_res_actions, base_next_actions], dim=1)

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - sac_alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * cfg.gamma * min_qf_next_target.view(-1)

            if cfg.critic_input == "res":
                current_q_actions = res_actions
            elif cfg.critic_input == "sum":
                current_q_actions = current_actions
            else:
                current_q_actions = torch.cat([res_actions, base_actions], dim=1)

            qf1_a_values = qf1(data.observations, current_q_actions).view(-1)
            qf2_a_values = qf2(data.observations, current_q_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            qf1_grad_norm = nn.utils.clip_grad_norm_(qf1.parameters(), cfg.max_grad_norm)
            qf2_grad_norm = nn.utils.clip_grad_norm_(qf2.parameters(), cfg.max_grad_norm)
            q_optimizer.step()

            train_log = {
                "update/qf_loss": qf_loss.item(),
                "update/qf1_loss": qf1_loss.item(),
                "update/qf2_loss": qf2_loss.item(),
                "update/qf1_mean": qf1_a_values.mean().item(),
                "update/qf2_mean": qf2_a_values.mean().item(),
                "update/target_q_mean": next_q_value.mean().item(),
                "update/qf1_grad_norm": float(qf1_grad_norm),
                "update/qf2_grad_norm": float(qf2_grad_norm),
                "update/sac_alpha": float(sac_alpha),
            }

            # --------------------------
            # Actor update
            # --------------------------
            if global_update % cfg.policy_frequency == 0:
                if cfg.actor_input == "obs":
                    actor_input_curr = data.observations
                else:
                    actor_input_curr = torch.cat([data.observations, base_actions], dim=1)

                res_pi, log_pi, _ = res_actor.get_action(actor_input_curr)

                if cfg.critic_input == "res":
                    pi = res_pi
                elif cfg.critic_input == "sum":
                    pi = base_actions + cfg.res_scale * res_pi
                else:
                    pi = torch.cat([res_pi, base_actions], dim=1)

                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((sac_alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(res_actor.parameters(), cfg.max_grad_norm)
                actor_optimizer.step()

                train_log.update({
                    "update/actor_loss": actor_loss.item(),
                    "update/actor_grad_norm": float(actor_grad_norm),
                    "update/log_pi": log_pi.mean().item(),
                    "update/res_abs_mean": res_pi.abs().mean().item(),
                })

                if cfg.autotune:
                    with torch.no_grad():
                        _, log_pi_alpha, _ = res_actor.get_action(actor_input_curr)
                    sac_alpha_loss = (-log_sac_alpha * (log_pi_alpha + target_entropy)).mean()

                    a_optimizer.zero_grad(set_to_none=True)
                    sac_alpha_loss.backward()
                    a_optimizer.step()
                    sac_alpha = log_sac_alpha.exp().item()

                    train_log["update/sac_alpha_loss"] = sac_alpha_loss.item()
                    train_log["update/sac_alpha"] = float(sac_alpha)

            # --------------------------
            # Target update
            # --------------------------
            if global_update % cfg.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

            if cfg.log_wandb:
                wandb.log(train_log, step=global_step)

        if global_step % cfg.eval_freq == 0:
            eval_scores, success_rate = evaluate_ogbench(
                env=eval_env,
                iql=iql,
                res_actor=res_actor,
                cfg=cfg,
                device=device,
                n_episodes=cfg.n_episodes,
            )

            eval_successes.append(success_rate)
            eval_log = {
                "eval/regret": float(np.mean(1 - np.array(eval_successes))),
                "eval/success_rate": float(success_rate),
                "eval/score_mean": float(np.mean(eval_scores)),
            }

            print("[Eval {}] ".format(global_step) + ", ".join(
                [f"{k}={v:.4f}" for k, v in eval_log.items()]
            ))

            if cfg.log_wandb:
                wandb.log(eval_log, step=global_step)

            if cfg.checkpoints_path is not None:
                ckpt = {
                    "res_actor": res_actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                }
                if cfg.autotune:
                    ckpt["log_sac_alpha"] = log_sac_alpha.detach().cpu()
                torch.save(ckpt, os.path.join(cfg.checkpoints_path, f"pd_step_{global_step}.pt"))

    if cfg.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()