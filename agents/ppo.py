from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


# -----------------------------
# Hyperparameters
# -----------------------------
# γ: discount factor
GAMMA = 0.99
# λ: GAE parameter for advantage estimation
GAE_LAMBDA = 0.95
# Learning rate (Adam)
LR = 1e-4
# Clipping parameter
CLIP_EPSILON = 0.2
# Entropy coefficient
ENTROPY_COEF = 0.01
# loss coefficient
VALUE_COEF = 0.5

PPO_EPOCHS = 15
BATCH_SIZE = 64
# collect before updating
ROLLOUT_LENGTH = 1024
# Max gradient norm for clipping
MAX_GRAD_NORM = 0.5


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # Shared feature
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(64, act_dim)
        
        # Critic head (value func)
        self.critic = nn.Linear(64, 1)
        
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return: logtis, value
        logtis (raw): [B, act_dim]
        value: [B, 1]
        """
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        """
        Sample action
        Args:
            x: state [B, obs_dim]
            deterministic: if True, select argmax action
        Returns:
            action: sampled action
            log_prob: log probability of the action
            value: value estimate V(s)
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy


class RolloutBuffer:
    """
    Batch train. 先收集数据再一起训
    Stores: (state, action, reward, value, log_prob, done)
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def push(self, state, action, reward, value, log_prob, done):
        # single transition.
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        # Return all
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
        )

    def clear(self):
        # clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


@dataclass
class PPOConfig:
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    lr: float = LR
    clip_epsilon: float = CLIP_EPSILON
    entropy_coef: float = ENTROPY_COEF
    value_coef: float = VALUE_COEF
    ppo_epochs: int = PPO_EPOCHS
    batch_size: int = BATCH_SIZE
    rollout_length: int = ROLLOUT_LENGTH
    max_grad_norm: float = MAX_GRAD_NORM

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOSolver:
    def __init__(self, observation_space: int, action_space: int, cfg: PPOConfig | None = None):
        # dim and hyperparam
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or PPOConfig()

        # device
        self.device = torch.device(self.cfg.device)

        # Build network
        self.network = ActorCriticNet(self.obs_dim, self.act_dim).to(self.device)

        # Optimizer
        self.optim = optim.Adam(self.network.parameters(), lr=self.cfg.lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Global counters
        self.steps = 0
        self.episodes = 0
        
        # 与 train.py 兼容？ (DQN has exploration_rate)
        self.exploration_rate = 0.0  # PPO doesn't use epsilon-greedy

    # -----------------------------
    # Acting & memory
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Sample action from the policy.
        - If evaluation_mode=True, act deterministically (argmax).
        - If evaluation_mode=False (training), sample from policy distribution.
        
        Inputs:
          state_np: numpy array with shape [1, obs_dim]
        """
        with torch.no_grad():
            s_np = np.asarray(state_np, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]  # (1, obs_dim)
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            
            action, log_prob, value = self.network.get_action(s, deterministic=evaluation_mode)
            
            # Store for later use in step()
            if not evaluation_mode:
                self._last_value = value.cpu().numpy().squeeze()
                self._last_log_prob = log_prob.cpu().numpy().squeeze()
            
            return int(action.cpu().numpy().squeeze())

    def remember(self, state: np.ndarray, action: int, reward: float, done: bool):
        """Store a single transition in the rollout buffer."""
        # Squeeze state if needed
        state_squeezed = state.squeeze() if state.ndim > 1 else state
        
        self.buffer.push(
            state=state_squeezed,
            action=action,
            reward=reward,
            value=self._last_value,
            log_prob=self._last_log_prob,
            done=done
        )

    # -----------------------------
    # Learning
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # in main.py
        self.remember(state, action, reward, done)
        self.steps += 1
        
        # Update when rollout buffer is full or episode ends
        if done:
            self.episodes += 1
            # Perform update at the end of episode or when buffer reaches rollout_length
            if len(self.buffer) >= self.cfg.rollout_length:
                self.update()
        elif len(self.buffer) >= self.cfg.rollout_length:
            self.update()

    def update(self):
        """
        Perform PPO update using collected trajectories.
        Steps:
          1) Get all transitions from buffer
          2) Compute advantages using GAE
          3) Normalize advantages
          4) Perform multiple epochs of minibatch updates
          5) Clear buffer
        """
        if len(self.buffer) == 0:
            return

        # Get data from buffer
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Convert to tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages for stable training
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update for multiple epochs
        dataset_size = len(states)
        
        for epoch in range(self.cfg.ppo_epochs):
            # Generate random minibatch indices
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                batch_indices = indices[start:end]
                
                # Get minibatch
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]

                # Evaluate actions under current policy
                log_probs, values, entropy = self.network.evaluate_actions(batch_states, batch_actions)
                
                # Compute probability ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus for exploration
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss
                
                # Optimization step
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

        # Clear buffer after update
        self.buffer.clear()

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE balances bias and variance in advantage estimation:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        Args:
            rewards: array of rewards [T]
            values: array of value estimates [T]
            dones: array of done flags [T]
        
        Returns:
            advantages: array of advantages [T]
            returns: array of returns (targets for value function) [T]
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Bootstrap value is 0 for terminal state
            else:
                next_value = values[t + 1]
            
            # Mask next_value if episode ended
            mask = 1.0 - dones[t]
            
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.cfg.gamma * next_value * mask - values[t]
            
            # GAE accumulation
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * last_gae
            advantages[t] = last_gae
        
        # Returns are advantages + values
        returns = advantages + values
        
        return advantages, returns

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: str):
        """
        Save network weights and config for reproducibility.
        """
        torch.save(
            {
                "network": self.network.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        """
        Load weights from disk onto the correct device.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        # Optional: restore cfg from ckpt["cfg"] if needed
