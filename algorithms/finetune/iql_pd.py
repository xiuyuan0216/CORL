# file: algorithms/finetune/iql_policy_decorator_d4rl.py
# D4RL online Policy Decorator on top of a frozen offline-trained IQL base policy.
#
# Key constraints satisfied:
#   1) IQL classes are kept unchanged (import directly from CORL finetune/iql.py)
#   2) Online residual actor/Q use Policy Decorator architecture (Actor, SoftQNetwork)
#   3) Goal success check uses CORL iql.py helper is_goal_reached(reward, info)
#
# Usage example:
# python algorithms/finetune/iql_policy_decorator_d4rl.py \
#   --env halfcheetah-medium-v2 \
#   --iql_checkpoint /path/to/offline_iql_ckpt.pt \
#   --base_iql_deterministic True \
#   --online_steps 1000000 \
#   --eval_freq 50000 \
#   --critic_input sum \
#   --actor_input obs_base_action

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl  # noqa: F401
import gym
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
# IMPORTANT: import IQL classes/functions DIRECTLY from CORL
# (unchanged class definitions / architectures / load_state_dict)
# ============================================================
# from algorithms.finetune.iql import (
#     GaussianPolicy,
#     DeterministicPolicy,
#     TwinQ,
#     ValueFunction,
#     ImplicitQLearning,
#     set_seed,
#     compute_mean_std,
#     wrap_env,
#     modify_reward,
#     ENVS_WITH_GOAL,
#     is_goal_reached,  # <-- use this for success check
# )

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")

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
    return reward > 0  # Assuming that reaching target is a positive reward

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
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


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
    

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
    lengths.append(ep_len)  # but still keep track of number of steps
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
            raise ValueError("MLP requires at least two dims (input and output)")

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
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

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
    # experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-v2"
    seed: int = 0
    eval_seed: int = 42
    online_steps: int = int(1e6)
    eval_freq: int = int(5e4)
    n_episodes: int = 10
    checkpoints_path: Optional[str] = None

    # load offline IQL checkpoint (required)
    iql_checkpoint: str = ""
    strict_iql_load: bool = True
    base_iql_deterministic: bool = True  # controls GaussianPolicy eval/train mode for .act behavior

    # CORL IQL architecture params (must match offline checkpoint exactly)
    iql_deterministic_actor: bool = False
    actor_dropout: float = 0.0
    hidden_dim: int = 256
    n_hidden: int = 2

    # IQL optimizer hypers only used to instantiate/load optimizer states
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7

    # env / normalization
    normalize: bool = True
    normalize_reward: bool = False

    # online data / training
    buffer_size: int = 2_000_000
    batch_size: int = 256
    learning_starts: int = 5000
    training_freq: int = 1
    utd: float = 1.0  # updates per env step
    gamma: float = 0.99

    # policy decorator settings (same semantics as pd code)
    actor_input: str = "obs"   # ['obs', 'obs_base_action']
    critic_input: str = "sum"              # ['res', 'sum', 'concat']
    res_scale: float = 0.1
    policy_frequency: int = 2
    target_network_frequency: int = 1
    q_lr: float = 3e-4
    policy_lr: float = 3e-4
    sac_alpha: float = 0.2
    autotune: bool = True
    max_grad_norm: float = 50.0
    alpha_init: float = None

    # exploration (pd-style progressive enable mask)
    progressive_residual: bool = False
    prog_explore: int = int(1e5)
    critic_layer_norm: bool = False

    # logging
    project: str = "CORL"
    group: str = "IQL-PolicyDecorator-D4RL"
    name: str = "IQL-PD-D4RL"
    log_wandb: bool = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

# ============================================================
# Replay buffer for decorator
# Stores base actions and residual actions for critic_input modes.
# ============================================================

class DecoratorReplayBuffer:
    """
    Stored action payload format:
      if critic_input == 'res' and actor_input == 'obs':
         actions = [res]                                  dim = A
      else:
         actions = [res, base, base_next]                 dim = 3A
    """
    def __init__(self, state_dim: int, action_dim_payload: int, buffer_size: int, device: str = "cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim_payload), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self._device)

    def add(self, obs, next_obs, action_payload, reward, done):
        i = self._pointer
        self._states[i] = self._to_tensor(obs)
        self._next_states[i] = self._to_tensor(next_obs)
        self._actions[i] = self._to_tensor(action_payload)
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
# Policy Decorator architectures (copied style from pi_dec_bet_maniskill2.py)
# ============================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SoftQNetwork(nn.Module):
    # same architecture style as policy_decorator online/pi_dec_bet_maniskill2.py SoftQNetwork
    # + optional LayerNorm controlled by critic_layer_norm
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
    # same architecture style as policy_decorator online/pi_dec_bet_maniskill2.py Actor
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
    """
    Build dummy env objects to instantiate Actor/SoftQNetwork unchanged.
    """
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
    # MUST match CORL offline IQL architecture / optimizer objects
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
        max_steps=max(1, cfg.online_steps),  # scheduler exists for loading
        discount=cfg.gamma,
        tau=cfg.tau,
        device=device,
    )
    return iql


def evaluate_d4rl(
    env: gym.Env,
    iql: ImplicitQLearning,
    res_actor: Actor,
    cfg: TrainConfig,
    device: str,
    n_episodes: int,
) -> Dict[str, float]:
    scores = []
    returns = []
    successes = []
    is_env_with_goal = cfg.env.startswith(ENVS_WITH_GOAL)
    goal_achieved = False

    # eval mode for residual actor
    res_actor.eval()
    iql.actor.eval()

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {}
        goal_achieved = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            base_a_t = iql.actor.act(obs_t, device=device)

            if cfg.actor_input == "obs":
                actor_in = obs_t
            else:
                actor_in = torch.cat([obs_t, base_a_t], dim=1)

            res_a_t = res_actor.get_eval_action(actor_in).squeeze(0).detach().cpu().numpy()
            final_a = base_a_t + cfg.res_scale * res_a_t
            # final_a = final_a_t.squeeze(0).cpu().numpy()
            final_a = np.clip(final_a, env.action_space.low, env.action_space.high)

            obs, reward, done, info = env.step(final_a)
            
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)
            
            ep_ret += reward

        returns.append(ep_ret)
        scores.append(float(env.get_normalized_score(ep_ret) * 100.0))

        if is_env_with_goal:
            successes.append(goal_achieved)

    res_actor.train()
    
    print(successes)

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

    # D4RL dataset used for normalization stats (and optional reward normalization)
    dataset = d4rl.qlearning_dataset(env)
    if cfg.normalize_reward:
        modify_reward(dataset, cfg.env)

    if cfg.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    env = wrap_env(env, state_mean=state_mean, state_std=state_std, reward_scale=1.0)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std, reward_scale=1.0)

    # -------- build/load IQL (UNCHANGED classes) --------
    iql = build_iql_agent_for_loading(cfg, state_dim, action_dim, max_action, device)
    ckpt = torch.load(cfg.iql_checkpoint)
    iql.load_state_dict(ckpt)

    # Freeze all IQL params
    for p in iql.qf.parameters():
        p.requires_grad_(False)
    for p in iql.vf.parameters():
        p.requires_grad_(False)
    for p in iql.actor.parameters():
        p.requires_grad_(False)

    # Base actor mode controls deterministic/stochastic behavior for GaussianPolicy.act()
    iql.actor.eval()

    # -------- build policy decorator residual actor/Q (UNCHANGED pd architecture) --------
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
    actor_optimizer = optim.Adam(list(res_actor.parameters()), lr=cfg.actor_lr)

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

    # replay payload size exactly following pd logic
    if cfg.critic_input == "res" and cfg.actor_input == "obs":
        action_payload_dim = action_dim
    else:
        action_payload_dim = action_dim * 3

    rb = DecoratorReplayBuffer(
        state_dim=state_dim,
        action_dim_payload=action_payload_dim,
        buffer_size=cfg.buffer_size,
        device=device,
    )

    # logging
    if cfg.log_wandb:
        wandb.init(project=cfg.project, group=cfg.group, name=cfg.name, config=asdict(cfg))

    if cfg.checkpoints_path is not None:
        Path(cfg.checkpoints_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.checkpoints_path, "config.txt"), "w") as f:
            f.write(str(cfg))

    # -------- online loop --------
    obs = env.reset()
    episode_return = 0.0
    goal_achieved = False
    episode_len = 0
    global_step = 0
    global_update = 0
    
    num_updates_per_training = max(1, int(cfg.training_freq * cfg.utd))
    eval_successes = []
    train_successes = []
    
    # eval_log = evaluate_d4rl(
    #             env=eval_env,
    #             iql=iql,
    #             res_actor=res_actor,
    #             cfg=cfg,
    #             device=device,
    #             n_episodes=cfg.n_episodes,
    #         )
    # eval_log["global_step"] = global_step
    # print("[Eval {}] ".format(global_step) + ", ".join(
    #     [f"{k}={v:.4f}" for k, v in eval_log.items() if isinstance(v, float)]
    # ))
    
    # assert False

    while global_step < cfg.online_steps:
        # collect transitions
        for _ in range(cfg.training_freq):
            if global_step >= cfg.online_steps:
                break
            global_step += 1

            # base IQL action (unchanged actor class API)
            base_action = iql.actor.act(obs, device=device)

            # progressive residual mask (optional, pd-style)
            if cfg.progressive_residual:
                res_ratio = min(global_step / max(1, cfg.prog_explore), 1.0)
                enable_res = (np.random.rand() < res_ratio)
            else:
                enable_res = True

            # choose residual action
            if rb.size() < cfg.learning_starts:
                res_action = env.action_space.sample().astype(np.float32)
                res_action[...] = 0.0
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

            # compose final action
            final_action = base_action + cfg.res_scale * res_action
            final_action = np.clip(final_action, env.action_space.low, env.action_space.high)

            next_obs, reward, done, info = env.step(final_action)
            
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)
            episode_return += reward

            real_done = False  # Episode can timeout which is different from done
            if done and episode_len < max_steps_per_episode:
                real_done = True
                
            if cfg.normalize_reward:
                reward = modify_reward_online(reward, cfg.env, **reward_mod_dict)

            # compute next base action for replay payload if needed
            base_next_action = iql.actor.act(next_obs, device=device)

            # payload follows pd logic
            if cfg.critic_input == "res" and cfg.actor_input == "obs":
                action_payload = res_action
            else:
                action_payload = np.concatenate([res_action, base_action, base_next_action], axis=0)

            rb.add(obs, next_obs, action_payload, float(reward), float(real_done))

            obs = next_obs
            episode_len += 1

            # episode end logging
            if done:
                ep_score = env.get_normalized_score(episode_return) * 100.0

                log_ep = {
                    "train/episode_return": episode_return,
                    "train/episode_len": episode_len,
                    "train/episode_score": ep_score,
                    "global_step": global_step,
                }

                # CORL-style goal success check (NOT info['is_success'])
                if is_env_with_goal:
                    log_ep["train/is_success"] = goal_achieved

                if cfg.log_wandb:
                    wandb.log(log_ep, step=global_step)
                else:
                    print(log_ep)

                obs, done = env.reset(), False
                episode_len = 0
                episode_return = 0
                goal_achieved = False

        # training only after warmup
        if rb.size() < cfg.learning_starts:
            continue

        for _ in range(num_updates_per_training):
            global_update += 1
            data = rb.sample(cfg.batch_size)

            # unpack replay payload exactly like pd script logic
            if cfg.critic_input != "res" or cfg.actor_input == "obs_base_action":
                res_actions = data.actions[:, :action_dim]
                base_actions = data.actions[:, action_dim: action_dim * 2]
                base_next_actions = data.actions[:, action_dim * 2: action_dim * 3]
            else:
                res_actions = data.actions
                base_actions = None
                base_next_actions = None

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
                    scaled_res_actions = cfg.res_scale * next_state_res_actions
                    next_state_actions = base_next_actions + scaled_res_actions
                else:  # concat
                    next_state_actions = torch.cat([next_state_res_actions, base_next_actions], dim=1)

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - sac_alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * cfg.gamma * min_qf_next_target.view(-1)

            if cfg.critic_input == "res":
                current_actions = res_actions
            elif cfg.critic_input == "sum":
                scaled_res_actions = cfg.res_scale * res_actions
                current_actions = base_actions + scaled_res_actions
            else:  # concat
                current_actions = torch.cat([res_actions, base_actions], dim=1)

            qf1_a_values = qf1(data.observations, current_actions).view(-1)
            qf2_a_values = qf2(data.observations, current_actions).view(-1)
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
            # actor update (delayed)
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
                    scaled_res_actions = cfg.res_scale * res_pi
                    pi = base_actions + scaled_res_actions
                else:  # concat
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
            # target update
            # --------------------------
            if global_update % cfg.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

            if cfg.log_wandb:
                wandb.log(train_log, step=global_step)

        # evaluation
        if global_step % cfg.eval_freq == 0:
            eval_log = evaluate_d4rl(
                env=eval_env,
                iql=iql,
                res_actor=res_actor,
                cfg=cfg,
                device=device,
                n_episodes=cfg.n_episodes,
            )
            eval_log["global_step"] = global_step
            print("[Eval {}] ".format(global_step) + ", ".join(
                [f"{k}={v:.4f}" for k, v in eval_log.items() if isinstance(v, float)]
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
