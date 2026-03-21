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
from copy import deepcopy
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Normal, TanhTransform, TransformedDistribution

TensorBatch = List[torch.Tensor]

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

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)

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
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

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
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

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
        with torch.no_grad():
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
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

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
    
class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

class CalQL:
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
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
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
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

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
        mc_returns: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(
            1, cql_q1_current_actions.shape[1]
        )

        num_vals = torch.sum(lower_bounds == lower_bounds)
        bound_rate_cql_q1_current_actions = (
            torch.sum(cql_q1_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_current_actions = (
            torch.sum(cql_q2_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q1_next_actions = (
            torch.sum(cql_q1_next_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_next_actions = (
            torch.sum(cql_q2_next_actions < lower_bounds) / num_vals
        )

        """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

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

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
                bound_rate_cql_q1_current_actions=bound_rate_cql_q1_current_actions.item(),  # noqa
                bound_rate_cql_q2_current_actions=bound_rate_cql_q2_current_actions.item(),  # noqa
                bound_rate_cql_q1_next_actions=bound_rate_cql_q1_next_actions.item(),
                bound_rate_cql_q2_next_actions=bound_rate_cql_q2_next_actions.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            mc_returns,
        ) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations,
            actions,
            next_observations,
            rewards,
            dones,
            mc_returns,
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
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
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

    # load offline Cal-QL checkpoint (required)
    calql_checkpoint: str = ""
    strict_load: bool = True

    # CORL Cal-QL architecture params (must match offline checkpoint exactly)
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    bc_steps: int = int(0)  # Number of BC steps at start
    target_update_period: int = 1  # Frequency of target nets updates
    cql_alpha: float = 10.0  # CQL offline regularization parameter
    cql_alpha_online: float = 10.0  # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

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
    actor_lr: float = 3e-4
    sac_alpha: float = 0.2
    autotune: bool = True
    max_grad_norm: float = 50.0
    alpha_init: float = None
    tau: float = 0.005

    # exploration (pd-style progressive enable mask)
    progressive_residual: bool = False
    prog_explore: int = int(1e5)
    critic_layer_norm: bool = False

    # logging
    project: str = "CORL"
    group: str = "CalQL-PolicyDecorator-D4RL"
    name: str = "CalQL-PD-D4RL"
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


def build_calql_agent_for_loading(
    cfg: TrainConfig,
    state_dim: int,
    action_dim: int,
    max_action: float,
    device: str,
):
    # MUST match CORL offline Cal-QL architecture / optimizer objects

    # policy
    calql_actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        orthogonal_init=cfg.orthogonal_init,
    ).to(cfg.device)
    calql_actor_optimizer = torch.optim.Adam(calql_actor.parameters(), cfg.policy_lr)

    # critics
    calql_qf1 = FullyConnectedQFunction(
        observation_dim=state_dim,
        action_dim=action_dim,
        orthogonal_init=cfg.orthogonal_init,
        n_hidden_layers=cfg.q_n_hidden_layers,
    ).to(device)

    calql_qf2 = FullyConnectedQFunction(
        observation_dim=state_dim,
        action_dim=action_dim,
        orthogonal_init=cfg.orthogonal_init,
        n_hidden_layers=cfg.q_n_hidden_layers,
    ).to(device)

    # optimizers
    critic_1_optimizer = torch.optim.Adam(list(calql_qf1.parameters()), cfg.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(calql_qf2.parameters()), cfg.qf_lr)

    # build trainer/agent
    calql = CalQL(
        critic_1=calql_qf1,
        critic_2=calql_qf2,
        critic_1_optimizer=critic_1_optimizer,
        critic_2_optimizer=critic_2_optimizer,
        actor=calql_actor,
        actor_optimizer=calql_actor_optimizer,
        discount=cfg.gamma,
        soft_target_update_rate=cfg.soft_target_update_rate,
        target_update_period=cfg.target_update_period,
        device=device,
        use_automatic_entropy_tuning=cfg.use_automatic_entropy_tuning,
        backup_entropy=cfg.backup_entropy,
        target_entropy=-action_dim,
        alpha_multiplier=cfg.alpha_multiplier,
        bc_steps=cfg.bc_steps,
        cql_n_actions=cfg.cql_n_actions,
        cql_importance_sample=cfg.cql_importance_sample,
        cql_lagrange=cfg.cql_lagrange,
        cql_target_action_gap=cfg.cql_target_action_gap,
        cql_temp=cfg.cql_temp,
        cql_alpha=cfg.cql_alpha,
        cql_max_target_backup=cfg.cql_max_target_backup,
        cql_clip_diff_min=cfg.cql_clip_diff_min,
        cql_clip_diff_max=cfg.cql_clip_diff_max,
    )

    return calql

def evaluate_d4rl(
    env: gym.Env,
    calql: CalQL,
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
    calql.actor.eval()

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {}
        goal_achieved = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            base_a_t = calql.actor.act(obs_t, device=device)

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
    assert cfg.calql_checkpoint != "", "--calql_checkpoint is required"

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
    calql = build_calql_agent_for_loading(cfg, state_dim, action_dim, max_action, device)
    ckpt = torch.load(cfg.calql_checkpoint)
    calql.load_state_dict(ckpt)

    # Freeze all Cal-QL params
    for p in calql.critic_1.parameters():
        p.requires_grad_(False)
    for p in calql.critic_2.parameters():
        p.requires_grad_(False)
    for p in calql.target_critic_1.parameters():
        p.requires_grad_(False)
    for p in calql.target_critic_2.parameters():
        p.requires_grad_(False)
    for p in calql.actor.parameters():
        p.requires_grad_(False)

    # Base actor mode controls deterministic/stochastic behavior for GaussianPolicy.act()
    calql.actor.eval()

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
            base_action = calql.actor.act(obs, device=device)

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
            base_next_action = calql.actor.act(next_obs, device=device)

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
                calql=calql,
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