# source reduced from the provided Cal-QL ogbench version
# Main changes:
#   1) remove mc_returns / get_return_to_go / calibration
#   2) remove all CQL regularization and related hyperparameters
#   3) keep SAC-style actor-critic updates
#   4) keep offline pretraining + online finetuning + offline/online mixed batches

import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ogbench
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "cube-double-play-singletask-task2-v0"
    seed: int = 0
    eval_seed: int = 0
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    offline_iterations: int = int(1e6)
    online_iterations: int = int(1e6)
    checkpoints_path: Optional[str] = None
    load_model: str = ""

    # Replay / optimization
    buffer_size: int = 2_000_000
    batch_size: int = 256
    discount: float = 0.99
    alpha_multiplier: float = 1.0
    use_automatic_entropy_tuning: bool = True
    backup_entropy: bool = False
    policy_lr: float = 3e-5
    qf_lr: float = 3e-4
    soft_target_update_rate: float = 5e-3
    bc_steps: int = 0
    target_update_period: int = 1

    # Architecture / normalization
    orthogonal_init: bool = True
    normalize: bool = True
    normalize_reward: bool = False
    q_n_hidden_layers: int = 2
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    # RLPD-style mixing
    mixing_ratio: float = 0.5
    is_sparse_reward: bool = False

    # Wandb logging
    project: str = "CORL"
    group: str = "RLPD-ogbench"
    name: str = "RLPD"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


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


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_ogbench_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(1.0 - data["masks"][..., None])

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


def set_env_seed(env: Optional[gym.Env], seed: int):
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


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, env_infos = env.step(action)
            done = terminated | truncated
            episode_reward += reward
        successes.append(env_infos["success"])
        episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)


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
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
) -> Dict:
    modification_data = {}
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        modification_data = {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
    return modification_data


def modify_reward_online(
    reward: float,
    env_name: str,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    **kwargs,
) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    reward = reward * reward_scale + reward_bias
    return reward


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)
            nn.init.constant_(module.bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)
        return action_sample, log_prob


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
    ):
        super().__init__()

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]

        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])

        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)

        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class RLPD:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: float = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps: int = 0,
        target_update_period: int = 1,
        device: str = "cpu",
    ):
        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=self.policy_lr
            )
        else:
            self.log_alpha = None

    def update_target_network(self, tau: float):
        soft_update(self.target_critic_1, self.critic_1, tau)
        soft_update(self.target_critic_2, self.critic_2, tau)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> torch.Tensor:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        with torch.no_grad():
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

            if self.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            target_q_values = target_q_values.unsqueeze(-1)
            td_target = rewards + (1.0 - dones) * self.discount * target_q_values
            td_target = td_target.squeeze(-1)

        qf1_loss = F.mse_loss(q1_predicted, td_target)
        qf2_loss = F.mse_loss(q2_predicted, td_target)
        qf_loss = qf1_loss + qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )
        return qf_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        observations, actions, rewards, next_observations, dones = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)
        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        qf_loss = self._q_loss(
            observations,
            actions,
            next_observations,
            rewards,
            dones,
            alpha,
            log_dict,
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }
        if self.use_automatic_entropy_tuning:
            state["sac_log_alpha"] = self.log_alpha.state_dict()
            state["sac_log_alpha_optim"] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic_1.load_state_dict(state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict["critic2"])
        self.target_critic_1.load_state_dict(state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict["critic2_target"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])

        if self.use_automatic_entropy_tuning and "sac_log_alpha" in state_dict:
            self.log_alpha.load_state_dict(state_dict["sac_log_alpha"])
            self.alpha_optimizer.load_state_dict(state_dict["sac_log_alpha_optim"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(config.env)
    eval_env = ogbench.make_env_and_datasets(config.env, env_only=True)

    batch_size_offline = int(config.batch_size * config.mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(
            train_dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(train_dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    train_dataset["observations"] = normalize_states(
        train_dataset["observations"], state_mean, state_std
    )
    train_dataset["next_observations"] = normalize_states(
        train_dataset["next_observations"], state_mean, state_std
    )

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    offline_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    online_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    offline_buffer.load_ogbench_dataset(train_dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
    }

    print("---------------------------------------")
    print(f"Training RLPD, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    trainer = RLPD(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file, map_location=config.device))
        actor = trainer.actor

    wandb_init(asdict(config))

    state, _ = env.reset()
    done = False
    episode_return = 0
    episode_step = 0

    eval_successes = []
    train_successes = []

    print("Offline pretraining")
    total_iters = int(config.offline_iterations) + int(config.online_iterations)

    for t in range(total_iters):
        online_log = {}

        if t >= config.offline_iterations:
            if t == config.offline_iterations:
                print("Online tuning")

            episode_step += 1
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1),
                    device=config.device,
                    dtype=torch.float32,
                )
            )
            action = action.cpu().data.numpy().flatten()

            next_state, reward, terminated, truncated, env_infos = env.step(action)
            done = terminated | truncated
            episode_return += reward

            real_done = env_infos["success"]

            reward_to_store = reward
            if config.normalize_reward:
                reward_to_store = modify_reward_online(
                    reward_to_store,
                    config.env,
                    reward_scale=config.reward_scale,
                    reward_bias=config.reward_bias,
                    **reward_mod_dict,
                )

            online_buffer.add_transition(
                state, action, reward_to_store, next_state, real_done
            )
            state = next_state

            if done:
                state, _ = env.reset()
                done = False
                train_successes.append(env_infos["success"])
                online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                online_log["train/is_success"] = float(env_infos["success"])
                online_log["train/episode_return"] = episode_return
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0

        if t < config.offline_iterations:
            batch = offline_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
        else:
            offline_batch = offline_buffer.sample(batch_size_offline)
            online_batch = online_buffer.sample(batch_size_online)
            batch = [
                torch.vstack(tuple(b)).to(config.device)
                for b in zip(offline_batch, online_batch)
            ]

        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        wandb.log(log_dict, step=trainer.total_it)

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_log = {
                "eval/regret": np.mean(1 - np.array(eval_successes + [success_rate])),
                "eval/success_rate": success_rate,
                "eval/episode_return": eval_score,
            }
            eval_successes.append(success_rate)

            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f}, Ogbench success rate: {success_rate:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(eval_log, step=trainer.total_it)


if __name__ == "__main__":
    train()