# file: algorithms/finetune/calql_pex_d4rl.py
#
# PEX-style offline-to-online training on top of a frozen offline checkpoint.
#
# Main behavior:
#   1) load a frozen offline policy from checkpoint
#   2) learn a new online actor + double Q
#   3) during rollout:
#        - propose offline action a_beta
#        - propose online action a_theta
#        - use current online critic to score both
#        - sample one action by softmax(Q / temperature)
#   4) during evaluation:
#        - use ONLY the online policy
#
# Notes:
#   - This keeps the online actor/Q update style close to your current code:
#       SAC-style actor loss + double Q + target Q + optional alpha autotune
#   - This is PEX-style action selection, but not a literal reproduction
#     of the exact actor update from the original PEX paper.
#
# Example:
# python algorithms/finetune/calql_pex_d4rl.py \
#   --env halfcheetah-medium-v2 \
#   --iql_checkpoint path/to/offline_ckpt.pt \
#   --device cuda

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl  # noqa: F401
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb


TensorBatch = List[torch.Tensor]
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ============================================================
# Utilities / env helpers
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


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0


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

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
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


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
) -> Dict[str, float]:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= (max_ret - min_ret)
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": float(max_ret),
            "min_ret": float(min_ret),
            "max_episode_steps": float(max_episode_steps),
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= (kwargs["max_ret"] - kwargs["min_ret"])
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


# ============================================================
# Minimal CORL-style modules for loading offline checkpoint
# ============================================================

class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn=nn.ReLU,
        output_activation_fn=None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
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

    def forward(self, obs: torch.Tensor):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, state: Union[np.ndarray, torch.Tensor], device: str = "cpu"):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        elif state.ndim == 1:
            state = state.unsqueeze(0).to(device=device, dtype=torch.float32)
        else:
            state = state.to(device=device, dtype=torch.float32)

        dist = self(state)
        if self.training:
            action = dist.sample()
        else:
            action = dist.mean
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()


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
    def act(self, state: Union[np.ndarray, torch.Tensor], device: str = "cpu"):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        elif state.ndim == 1:
            state = state.unsqueeze(0).to(device=device, dtype=torch.float32)
        else:
            state = state.to(device=device, dtype=torch.float32)

        action = torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor):
        return self.v(state)


class DummyLRSchedule:
    def load_state_dict(self, state_dict):
        return


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
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        **kwargs,
    ):
        self.max_action = max_action
        self.actor = actor
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.v_optimizer = v_optimizer
        self.discount = discount
        self.tau = tau
        self.device = device
        self.total_it = 0
        self.actor_lr_schedule = DummyLRSchedule()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        if "q_optimizer" in state_dict:
            self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        if "v_optimizer" in state_dict:
            self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])

        if "actor_lr_schedule" in state_dict:
            try:
                self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
            except Exception:
                pass

        self.total_it = state_dict.get("total_it", 0)


# ============================================================
# Online actor / critic
# ============================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, critic_layer_norm: bool = False):
        super().__init__()
        in_dim = obs_dim + action_dim
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

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        return self.net(torch.cat([x, a], dim=1))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_space: gym.Space):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(256, action_dim), std=0.01)

        h, l = action_space.high, action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x: torch.Tensor):
        h = self.backbone(x)
        mean = self.fc_mean(h)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x: torch.Tensor):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


# ============================================================
# Replay buffers
# ============================================================

class ReplayBuffer:
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
        return {
            "observations": self._states[idx],
            "next_observations": self._next_states[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "dones": self._dones[idx],
        }

    def size(self):
        return self._size


class OfflineReplayBuffer:
    def __init__(self, dataset: Dict[str, np.ndarray], device: str = "cpu"):
        self.device = device
        self.observations = torch.tensor(dataset["observations"], dtype=torch.float32, device=device)
        self.actions = torch.tensor(dataset["actions"], dtype=torch.float32, device=device)
        self.rewards = torch.tensor(dataset["rewards"], dtype=torch.float32, device=device).unsqueeze(-1)
        self.next_observations = torch.tensor(dataset["next_observations"], dtype=torch.float32, device=device)
        self.dones = torch.tensor(dataset["terminals"], dtype=torch.float32, device=device).unsqueeze(-1)
        self.size_ = self.observations.shape[0]

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size_, size=batch_size)
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
        }


def concat_batches(batch_a: Dict[str, torch.Tensor], batch_b: Dict[str, torch.Tensor]):
    out = {}
    for k in batch_a.keys():
        out[k] = torch.cat([batch_a[k], batch_b[k]], dim=0)
    return out


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-v2"
    seed: int = 0
    eval_seed: int = 42
    online_steps: int = int(1e6)
    eval_freq: int = int(5e4)
    n_episodes: int = 10
    checkpoints_path: Optional[str] = None

    # load offline checkpoint
    iql_checkpoint: str = ""
    strict_iql_load: bool = True
    base_iql_deterministic: bool = True

    # offline checkpoint architecture params
    iql_deterministic_actor: bool = False
    actor_dropout: float = 0.0
    hidden_dim: int = 256
    n_hidden: int = 2

    # only for instantiating optimizer objects before loading ckpt
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7

    # normalization
    normalize: bool = True
    normalize_reward: bool = False

    # online training
    buffer_size: int = 2_000_000
    batch_size: int = 256
    learning_starts: int = 5000
    training_freq: int = 1
    utd: float = 1.0
    gamma: float = 0.99

    # online actor/Q
    q_lr: float = 3e-4
    policy_lr: float = 3e-4
    sac_alpha: float = 0.2
    autotune: bool = True
    alpha_init: Optional[float] = None
    policy_frequency: int = 2
    target_network_frequency: int = 1
    max_grad_norm: float = 50.0
    critic_layer_norm: bool = False

    # PEX
    pex_temperature: float = 1.0
    offline_mix_ratio: float = 0.5
    online_action_before_start: str = "offline"   # ["offline", "random", "mix"]

    # logging
    project: str = "CORL"
    group: str = "PEX-D4RL"
    name: str = "PEX-D4RL"
    log_wandb: bool = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# ============================================================
# Build offline agent for loading
# ============================================================

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
        discount=cfg.gamma,
        tau=cfg.tau,
        device=device,
    )
    return iql


# ============================================================
# PEX action selection
# ============================================================

@torch.no_grad()
def pex_select_action(
    obs: np.ndarray,
    offline_actor: nn.Module,
    online_actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    action_space: gym.Space,
    device: str,
    temperature: float = 1.0,
):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    offline_action = offline_actor.act(obs, device=device)
    offline_a_t = torch.tensor(offline_action, dtype=torch.float32, device=device).unsqueeze(0)

    online_a_t, _, _ = online_actor.get_action(obs_t)

    q_beta = torch.min(qf1(obs_t, offline_a_t), qf2(obs_t, offline_a_t)).item()
    q_theta = torch.min(qf1(obs_t, online_a_t), qf2(obs_t, online_a_t)).item()

    logits = torch.tensor([q_beta, q_theta], dtype=torch.float32, device=device) / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=0)

    selected_idx = int(torch.distributions.Categorical(probs=probs).sample().item())

    if selected_idx == 0:
        final_action = offline_action
    else:
        final_action = online_a_t.squeeze(0).detach().cpu().numpy()

    final_action = np.clip(final_action, action_space.low, action_space.high)

    info = {
        "offline_action": offline_action,
        "online_action": online_a_t.squeeze(0).detach().cpu().numpy(),
        "q_beta": float(q_beta),
        "q_theta": float(q_theta),
        "p_beta": float(probs[0].item()),
        "p_theta": float(probs[1].item()),
        "selected_idx": selected_idx,  # 0=offline, 1=online
    }
    return final_action, info


# ============================================================
# Evaluation: ONLINE POLICY ONLY
# ============================================================

def evaluate_d4rl(
    env: gym.Env,
    online_actor: Actor,
    env_name: str,
    device: str,
    n_episodes: int,
) -> Dict[str, float]:
    returns, scores, successes = [], [], []
    is_env_with_goal = env_name.startswith(ENVS_WITH_GOAL)

    online_actor.eval()

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        goal_achieved = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = online_actor.get_eval_action(obs_t)
            action = action.squeeze(0).detach().cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, done, info = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)

            ep_ret += reward

        returns.append(ep_ret)
        scores.append(float(env.get_normalized_score(ep_ret) * 100.0))
        if is_env_with_goal:
            successes.append(goal_achieved)

    online_actor.train()

    out = {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/score_mean": float(np.mean(scores)),
        "eval/score_std": float(np.std(scores)),
    }
    if len(successes) > 0:
        out["eval/success_mean"] = float(np.mean(successes))
    return out


# ============================================================
# Main training
# ============================================================

@pyrallis.wrap()
def train(cfg: TrainConfig):
    assert cfg.iql_checkpoint != "", "--iql_checkpoint is required"

    device = cfg.device if torch.cuda.is_available() else "cpu"

    env = gym.make(cfg.env)
    eval_env = gym.make(cfg.env)
    is_env_with_goal = cfg.env.startswith(ENVS_WITH_GOAL)
    max_steps_per_episode = env._max_episode_steps

    set_seed(cfg.seed, env)
    set_seed(cfg.eval_seed, eval_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)
    reward_mod_dict = {}
    if cfg.normalize_reward:
        reward_mod_dict = modify_reward(dataset, cfg.env, max_episode_steps=max_steps_per_episode)

    if cfg.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
        dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    else:
        state_mean, state_std = 0, 1

    env = wrap_env(env, state_mean=state_mean, state_std=state_std, reward_scale=1.0)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std, reward_scale=1.0)

    # -------- load offline checkpoint --------
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
    else:
        iql.actor.train()

    # -------- build online actor/Q --------
    online_actor = Actor(state_dim, env.action_space).to(device)
    qf1 = SoftQNetwork(state_dim, action_dim, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf2 = SoftQNetwork(state_dim, action_dim, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf1_target = SoftQNetwork(state_dim, action_dim, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf2_target = SoftQNetwork(state_dim, action_dim, critic_layer_norm=cfg.critic_layer_norm).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.q_lr)
    actor_optimizer = optim.Adam(online_actor.parameters(), lr=cfg.policy_lr)

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

    offline_rb = OfflineReplayBuffer(dataset, device=device)
    online_rb = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=cfg.buffer_size,
        device=device,
    )

    if cfg.log_wandb:
        wandb.init(project=cfg.project, group=cfg.group, name=cfg.name, config=asdict(cfg))

    if cfg.checkpoints_path is not None:
        Path(cfg.checkpoints_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.checkpoints_path, "config.txt"), "w") as f:
            f.write(str(cfg))

    obs = env.reset()
    episode_return = 0.0
    episode_len = 0
    goal_achieved = False
    global_step = 0
    global_update = 0

    num_updates_per_training = max(1, int(cfg.training_freq * cfg.utd))

    while global_step < cfg.online_steps:
        # --------------------------------------------------
        # collect environment transitions using PEX
        # --------------------------------------------------
        for _ in range(cfg.training_freq):
            if global_step >= cfg.online_steps:
                break

            global_step += 1

            if online_rb.size() < cfg.learning_starts:
                if cfg.online_action_before_start == "random":
                    action = env.action_space.sample().astype(np.float32)
                    pex_info = {
                        "q_beta": 0.0,
                        "q_theta": 0.0,
                        "p_beta": 0.0,
                        "p_theta": 0.0,
                        "selected_idx": -1,
                    }
                elif cfg.online_action_before_start == "mix":
                    if np.random.rand() < 0.5:
                        action = iql.actor.act(obs, device=device)
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                        pex_info = {
                            "q_beta": 0.0,
                            "q_theta": 0.0,
                            "p_beta": 1.0,
                            "p_theta": 0.0,
                            "selected_idx": 0,
                        }
                    else:
                        action = env.action_space.sample().astype(np.float32)
                        pex_info = {
                            "q_beta": 0.0,
                            "q_theta": 0.0,
                            "p_beta": 0.0,
                            "p_theta": 0.0,
                            "selected_idx": -1,
                        }
                else:
                    action = iql.actor.act(obs, device=device)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    pex_info = {
                        "q_beta": 0.0,
                        "q_theta": 0.0,
                        "p_beta": 1.0,
                        "p_theta": 0.0,
                        "selected_idx": 0,
                    }
            else:
                action, pex_info = pex_select_action(
                    obs=obs,
                    offline_actor=iql.actor,
                    online_actor=online_actor,
                    qf1=qf1,
                    qf2=qf2,
                    action_space=env.action_space,
                    device=device,
                    temperature=cfg.pex_temperature,
                )

            next_obs, reward, done, info = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)

            raw_reward = reward
            if cfg.normalize_reward:
                reward = modify_reward_online(reward, cfg.env, **reward_mod_dict)

            real_done = False
            if done and episode_len < max_steps_per_episode:
                real_done = True

            online_rb.add(obs, next_obs, action, float(reward), float(real_done))

            episode_return += raw_reward
            episode_len += 1
            obs = next_obs

            if done:
                ep_score = env.get_normalized_score(episode_return) * 100.0
                log_ep = {
                    "train/episode_return": episode_return,
                    "train/episode_len": episode_len,
                    "train/episode_score": ep_score,
                    "global_step": global_step,
                }
                if is_env_with_goal:
                    log_ep["train/is_success"] = goal_achieved
                if pex_info["selected_idx"] >= 0:
                    log_ep["train/pex_p_beta"] = pex_info["p_beta"]
                    log_ep["train/pex_p_theta"] = pex_info["p_theta"]
                    log_ep["train/pex_selected_offline"] = float(pex_info["selected_idx"] == 0)
                    log_ep["train/pex_selected_online"] = float(pex_info["selected_idx"] == 1)

                if cfg.log_wandb:
                    wandb.log(log_ep, step=global_step)
                else:
                    print(log_ep)

                obs = env.reset()
                episode_return = 0.0
                episode_len = 0
                goal_achieved = False

        if online_rb.size() < cfg.learning_starts:
            continue

        # --------------------------------------------------
        # updates
        # --------------------------------------------------
        for _ in range(num_updates_per_training):
            global_update += 1

            offline_bs = int(cfg.batch_size * cfg.offline_mix_ratio)
            online_bs = cfg.batch_size - offline_bs

            if offline_bs <= 0:
                batch = online_rb.sample(cfg.batch_size)
            elif online_bs <= 0:
                batch = offline_rb.sample(cfg.batch_size)
            else:
                batch_offline = offline_rb.sample(offline_bs)
                batch_online = online_rb.sample(online_bs)
                batch = concat_batches(batch_offline, batch_online)

            observations = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"].flatten()
            next_observations = batch["next_observations"]
            dones = batch["dones"].flatten()

            # -------- Q update --------
            with torch.no_grad():
                next_actions, next_log_pi, _ = online_actor.get_action(next_observations)
                qf1_next_target = qf1_target(next_observations, next_actions)
                qf2_next_target = qf2_target(next_observations, next_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - sac_alpha * next_log_pi
                next_q_value = rewards + (1 - dones) * cfg.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(observations, actions).view(-1)
            qf2_a_values = qf2(observations, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad(set_to_none=True)
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
                "update/offline_batch_size": offline_bs,
                "update/online_batch_size": online_bs,
            }

            # -------- actor update --------
            if global_update % cfg.policy_frequency == 0:
                pi, log_pi, _ = online_actor.get_action(observations)
                qf1_pi = qf1(observations, pi)
                qf2_pi = qf2(observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((sac_alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(online_actor.parameters(), cfg.max_grad_norm)
                actor_optimizer.step()

                train_log.update({
                    "update/actor_loss": actor_loss.item(),
                    "update/actor_grad_norm": float(actor_grad_norm),
                    "update/log_pi": log_pi.mean().item(),
                    "update/pi_abs_mean": pi.abs().mean().item(),
                })

                if cfg.autotune:
                    with torch.no_grad():
                        _, log_pi_alpha, _ = online_actor.get_action(observations)
                    sac_alpha_loss = (-log_sac_alpha * (log_pi_alpha + target_entropy)).mean()

                    a_optimizer.zero_grad(set_to_none=True)
                    sac_alpha_loss.backward()
                    a_optimizer.step()
                    sac_alpha = log_sac_alpha.exp().item()

                    train_log["update/sac_alpha_loss"] = sac_alpha_loss.item()
                    train_log["update/sac_alpha"] = float(sac_alpha)

            # -------- target update --------
            if global_update % cfg.target_network_frequency == 0:
                soft_update(qf1_target, qf1, cfg.tau)
                soft_update(qf2_target, qf2, cfg.tau)

            if cfg.log_wandb:
                wandb.log(train_log, step=global_step)

        # --------------------------------------------------
        # evaluation: online policy only
        # --------------------------------------------------
        if global_step % cfg.eval_freq == 0:
            eval_log = evaluate_d4rl(
                env=eval_env,
                online_actor=online_actor,
                env_name=cfg.env,
                device=device,
                n_episodes=cfg.n_episodes,
            )
            eval_log["global_step"] = global_step

            print(
                f"[Eval {global_step}] "
                + ", ".join(
                    [f"{k}={v:.4f}" for k, v in eval_log.items() if isinstance(v, float)]
                )
            )

            if cfg.log_wandb:
                wandb.log(eval_log, step=global_step)

            if cfg.checkpoints_path is not None:
                save_ckpt = {
                    "online_actor": online_actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "global_step": global_step,
                    "config": asdict(cfg),
                    "state_mean": np.array(state_mean),
                    "state_std": np.array(state_std),
                    "offline_policy_checkpoint": cfg.iql_checkpoint,
                }
                if cfg.autotune:
                    save_ckpt["log_sac_alpha"] = log_sac_alpha.detach().cpu()
                    save_ckpt["a_optimizer"] = a_optimizer.state_dict()

                torch.save(save_ckpt, os.path.join(cfg.checkpoints_path, f"pex_step_{global_step}.pt"))

    if cfg.log_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()