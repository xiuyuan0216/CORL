# file: algorithms/finetune/fql.py
# PyTorch / CORL-style rewrite of original FQL
#
# Faithful-to-original version:
# - no extra expl_noise
# - no extra OneStepPolicy wrapper class
# - one global optimizer
# - same actor/critic loss structure as original FQL
#
# References:
# - original FQL agent: agents/fql.py
# - original FQL training loop: main.py

import os
import copy
import uuid
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb

import d4rl  # noqa: F401

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")
TensorBatch = List[torch.Tensor]

# =========================================================
# Utils
# =========================================================

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
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
):
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def return_reward_range(dataset: Dict[str, np.ndarray], max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0

    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0

    if ep_len > 0:
        returns.append(ep_ret)
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

def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    agent: "FQL",
    device: str,
    n_episodes: int,
    seed: int,
) -> np.ndarray:
    scores = []
    returns = []
    successes = []
#    is_env_with_goal = cfg.env.startswith(ENVS_WITH_GOAL)
    agent.actor_onestep_flow.eval()

    for ep in range(n_episodes):
        state = env.reset()

        done = False
        ep_ret = 0.0
        
        info = {}
        goal_achieved = False

        while not done:
            state_t = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = agent.sample_actions(state_t).squeeze(0).cpu().numpy()

            next_state, reward, done, info = env.step(action)
            
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)

            ep_ret += reward
            state = next_state
        
        returns.append(ep_ret)
        scores.append(float(env.get_normalized_score(ep_ret) * 100.0))

        successes.append(goal_achieved)
        
    agent.actor_onestep_flow.train()
    
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


# =========================================================
# Config
# =========================================================

@dataclass
class TrainConfig:
    # experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-v2"
    seed: int = 0
    eval_seed: int = 42
    eval_freq: int = int(1e5)
    offline_iterations: int = int(1e6)
    online_iterations: int = int(1e6)
    n_episodes: int = 50
    checkpoints_path: Optional[str] = None
    load_model: str = ""

    # replay / dataset
    buffer_size: int = 2_000_000
    normalize: bool = True
    normalize_reward: bool = False
    balanced_sampling: bool = False

    # optimization
    lr: float = 3e-4
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005

    # architecture
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 512, 512)
    value_hidden_dims: Tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True
    actor_layer_norm: bool = False

    # FQL
    q_agg: str = "min"  # ["mean", "min"]
    alpha: float = 10.0
    flow_steps: int = 10
    normalize_q_loss: bool = False

    # logging
    project: str = "CORL"
    group: str = "FQL-D4RL"
    name: str = "FQL-D4RL"
    log_wandb: bool = True

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# =========================================================
# Replay Buffer
# =========================================================

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
        self._device = device

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._terminals = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._masks = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self._device)

    @property
    def size(self) -> int:
        return self._size

    def load_d4rl_dataset(self, dataset: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        n = dataset["observations"].shape[0]
        if n > self._buffer_size:
            raise ValueError("Replay buffer is smaller than dataset")

        self._states[:n] = self._to_tensor(dataset["observations"])
        self._actions[:n] = self._to_tensor(dataset["actions"])
        self._rewards[:n] = self._to_tensor(dataset["rewards"][..., None])
        self._next_states[:n] = self._to_tensor(dataset["next_observations"])
        self._terminals[:n] = self._to_tensor(dataset["terminals"][..., None])
        self._masks[:n] = 1.0 - self._terminals[:n]

        self._size = n
        self._pointer = n % self._buffer_size
        print(f"Dataset size: {n}")

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations": self._states[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_observations": self._next_states[idx],
            "terminals": self._terminals[idx],
            "masks": self._masks[idx],
        }

    def add_transition(self, transition: Dict[str, Union[np.ndarray, float]]):
        self._states[self._pointer] = self._to_tensor(transition["observations"])
        self._actions[self._pointer] = self._to_tensor(transition["actions"])
        self._rewards[self._pointer] = self._to_tensor(np.array([transition["rewards"]], dtype=np.float32))
        self._next_states[self._pointer] = self._to_tensor(transition["next_observations"])
        self._terminals[self._pointer] = self._to_tensor(np.array([transition["terminals"]], dtype=np.float32))
        self._masks[self._pointer] = self._to_tensor(np.array([transition["masks"]], dtype=np.float32))

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


def concat_batches(batch_a: Dict[str, torch.Tensor], batch_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([batch_a[k], batch_b[k]], dim=0) for k in batch_a.keys()}


# =========================================================
# Networks
# =========================================================

def build_mlp_from_dims(dims, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class Value(nn.Module):
    """
    Match original FQL critic:
    outputs 2 Q-values.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...],
        layer_norm: bool,
        num_ensembles: int = 2,
    ):
        super().__init__()
        assert num_ensembles == 2
        dims = [state_dim + action_dim, *hidden_dims, 1]
        self.q1 = build_mlp_from_dims(dims, layer_norm=layer_norm)
        self.q2 = build_mlp_from_dims(dims, layer_norm=layer_norm)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observations, actions], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return torch.cat([q1, q2], dim=1)

    def both(self, observations: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([observations, actions], dim=-1)
        return self.q1(x), self.q2(x)


class ActorVectorField(nn.Module):
    """
    Used for both:
    - actor_bc_flow(observation, x_t, t)
    - actor_onestep_flow(observation, noise)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...],
        layer_norm: bool,
        with_time: bool,
    ):
        super().__init__()
        self.with_time = with_time
        self.action_dim = action_dim
        in_dim = state_dim + action_dim + (1 if with_time else 0)
        dims = [in_dim, *hidden_dims, action_dim]
        self.net = build_mlp_from_dims(dims, layer_norm=layer_norm)

    def forward(
        self,
        observations: torch.Tensor,
        actions_or_noises: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_time:
            assert t is not None
            x = torch.cat([observations, actions_or_noises, t], dim=-1)
        else:
            x = torch.cat([observations, actions_or_noises], dim=-1)
        return self.net(x)


# =========================================================
# FQL Agent
# =========================================================

class FQL:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        cfg: TrainConfig,
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.action_dim = action_dim
        self.max_action = max_action

        self.critic = Value(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.layer_norm,
            num_ensembles=2,
        ).to(self.device)

        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.actor_bc_flow = ActorVectorField(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            layer_norm=cfg.actor_layer_norm,
            with_time=True,
        ).to(self.device)

        self.actor_onestep_flow = ActorVectorField(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            layer_norm=cfg.actor_layer_norm,
            with_time=False,
        ).to(self.device)

        # original FQL uses a single optimizer over all modules
        self.optimizer = torch.optim.Adam(
            list(self.critic.parameters())
            + list(self.actor_bc_flow.parameters())
            + list(self.actor_onestep_flow.parameters()),
            lr=cfg.lr,
        )

        self.total_it = 0

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        noises = torch.randn(observations.shape[0], self.action_dim, device=self.device)
        actions = self.actor_onestep_flow(observations, noises)
        actions = torch.clamp(actions, -1.0, 1.0)
        return actions

    @torch.no_grad()
    def compute_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        actions = noises
        for i in range(self.cfg.flow_steps):
            t = torch.full(
                (observations.shape[0], 1),
                fill_value=i / self.cfg.flow_steps,
                device=self.device,
                dtype=torch.float32,
            )
            vels = self.actor_bc_flow(observations, actions, t)
            actions = actions + vels / self.cfg.flow_steps
            actions = torch.clamp(actions, -1.0, 1.0)
        return actions

    def critic_loss(self, batch: Dict[str, torch.Tensor]):
        with torch.no_grad():
            next_actions = self.sample_actions(batch["next_observations"])
            next_qs = self.target_critic(batch["next_observations"], next_actions)
            if self.cfg.q_agg == "min":
                next_q = next_qs.min(dim=1, keepdim=True)[0]
            else:
                next_q = next_qs.mean(dim=1, keepdim=True)

            target_q = batch["rewards"] + self.cfg.discount * batch["masks"] * next_q

        q = self.critic(batch["observations"], batch["actions"])
        critic_loss = ((q - target_q) ** 2).mean()

        info = {
            "critic_loss": critic_loss.item(),
            "q_mean": q.mean().item(),
            "q_max": q.max().item(),
            "q_min": q.min().item(),
        }
        return critic_loss, info

    def actor_loss(self, batch: Dict[str, torch.Tensor]):
        batch_size, action_dim = batch["actions"].shape

        # 1) BC flow loss
        x_0 = torch.randn(batch_size, action_dim, device=self.device)
        x_1 = batch["actions"]
        t = torch.rand(batch_size, 1, device=self.device)
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.actor_bc_flow(batch["observations"], x_t, t)
        bc_flow_loss = ((pred - vel) ** 2).mean()

        # 2) Distillation loss
        noises = torch.randn(batch_size, action_dim, device=self.device)
        with torch.no_grad():
            target_flow_actions = self.compute_flow_actions(batch["observations"], noises=noises)
        actor_actions = self.actor_onestep_flow(batch["observations"], noises)
        distill_loss = ((actor_actions - target_flow_actions) ** 2).mean()

        # 3) Q loss
        actor_actions = torch.clamp(actor_actions, -1.0, 1.0)
        qs = self.critic(batch["observations"], actor_actions)
        q = qs.mean(dim=1, keepdim=True)
        q_loss = -q.mean()

        if self.cfg.normalize_q_loss:
            lam = 1.0 / (q.abs().mean().detach() + 1e-8)
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.cfg.alpha * distill_loss + q_loss

        with torch.no_grad():
            sampled_actions = self.sample_actions(batch["observations"])
            mse = ((sampled_actions - batch["actions"]) ** 2).mean()

        info = {
            "actor_loss": actor_loss.item(),
            "bc_flow_loss": bc_flow_loss.item(),
            "distill_loss": distill_loss.item(),
            "q_loss": q_loss.item(),
            "q": q.mean().item(),
            "mse": mse.item(),
        }
        return actor_loss, info

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.total_it += 1

        critic_loss, critic_info = self.critic_loss(batch)
        actor_loss, actor_info = self.actor_loss(batch)
        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        soft_update(self.target_critic, self.critic, self.cfg.tau)

        info = {}
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v
        return info

    def save(self, path: str):
        torch.save(
            {
                "config": asdict(self.cfg),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_bc_flow": self.actor_bc_flow.state_dict(),
                "actor_onestep_flow": self.actor_onestep_flow.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_bc_flow.load_state_dict(ckpt["actor_bc_flow"])
        self.actor_onestep_flow.load_state_dict(ckpt["actor_onestep_flow"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_it = ckpt.get("total_it", 0)


# =========================================================
# Training
# =========================================================

@pyrallis.wrap()
def train(cfg: TrainConfig):
    assert d4rl is not None, "Please install d4rl first."

    env = gym.make(cfg.env)
    eval_env = gym.make(cfg.env)
    
    max_steps_per_episode = env._max_episode_steps
    
    is_env_with_goal = cfg.env.startswith(ENVS_WITH_GOAL)

    set_seed(cfg.seed, env)
    set_seed(cfg.eval_seed, eval_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)

    if cfg.normalize_reward:
        modify_reward(dataset, cfg.env)

    if cfg.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0.0, 1.0

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    train_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=max(cfg.buffer_size, dataset["observations"].shape[0] + 1),
        device=cfg.device,
    )
    train_buffer.load_d4rl_dataset(dataset)

    # faithful to original main.py:
    # balanced_sampling => separate replay buffer
    # otherwise use training dataset itself as replay buffer
    if cfg.balanced_sampling:
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=cfg.buffer_size,
            device=cfg.device,
        )
    else:
        replay_buffer = train_buffer

    agent = FQL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        cfg=cfg,
    )
    
    # logging
    if cfg.log_wandb:
        wandb.init(project=cfg.project, group=cfg.group, name=cfg.name, config=asdict(cfg))

    if cfg.load_model != "":
        agent.load(cfg.load_model)
        print(f"Loaded model from {cfg.load_model}")

    if cfg.checkpoints_path is not None:
        Path(cfg.checkpoints_path).mkdir(parents=True, exist_ok=True)

    total_steps = cfg.offline_iterations + cfg.online_iterations
    
    obs = env.reset()
    episode_return = 0.0
    goal_achieved = False
    episode_len = 0
    global_step = 0

    eval_successes = []
    train_successes = []

    for i in range(1, total_steps + 1):
        
        global_step += 1
        
        if i <= cfg.offline_iterations:
            batch = train_buffer.sample(cfg.batch_size)
            update_info = agent.update(batch)
        else:

            obs_t = torch.tensor(obs, device=cfg.device, dtype=torch.float32).unsqueeze(0)
            action = agent.sample_actions(obs_t).squeeze(0).cpu().numpy()

            next_obs, reward, done, info = env.step(action.copy())
                
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, info)
            episode_return += reward
            
            real_done = False  # Episode can timeout which is different from done
            if done and episode_len < max_steps_per_episode:
                real_done = True
                
            if cfg.normalize_reward:
                reward = modify_reward_online(reward, cfg.env, **reward_mod_dict)

            replay_buffer.add_transition(
                dict(
                    observations=obs,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - float(real_done),
                    next_observations=next_obs,
                )
            )

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

            if cfg.balanced_sampling:
                dataset_batch = train_buffer.sample(cfg.batch_size // 2)
                replay_batch = replay_buffer.sample(cfg.batch_size // 2)
                batch = concat_batches(dataset_batch, replay_batch)
            else:
                batch = replay_buffer.sample(cfg.batch_size)

            update_info = agent.update(batch)
            
            train_log = {
                "train/critic_loss": update_info['critic/critic_loss'],
                "train/actor_loss": update_info['actor/actor_loss'],
                "train/bc_flow_loss": update_info['actor/bc_flow_loss'],
                "train/distill_loss": update_info['actor/distill_loss'],
                "train/q_loss": update_info['actor/q_loss'],
            }
            
            if cfg.log_wandb:
                wandb.log(train_log, step=global_step)

        if i % cfg.eval_freq == 0:
            eval_log = eval_actor(
                env=eval_env,
                agent=agent,
                device=cfg.device,
                n_episodes=cfg.n_episodes,
                seed=cfg.eval_seed,
            )
            eval_log["global_step"] = global_step
            print("[Eval {}] ".format(global_step) + ", ".join(
                [f"{k}={v:.4f}" for k, v in eval_log.items() if isinstance(v, float)]
            ))
            if cfg.log_wandb:
                wandb.log(eval_log, step=global_step)

            if cfg.checkpoints_path is not None:
                agent.save(os.path.join(cfg.checkpoints_path, f"fql_step_{i}.pt"))
    if cfg.log_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()
