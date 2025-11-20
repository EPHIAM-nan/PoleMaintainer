"""
PyTorch Double DQN for CartPole (Gymnasium)
-------------------------------------------
- Online Q-network + Target Q-network
- Double DQN target calculation to reduce overestimation bias
- Vectorized replay updates
- Designed to be imported by train.py
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse QNet and ReplayBuffer from the original DQN implementation
# to avoid code duplication, or redefine them if we want a standalone file.
# Here we redefine them for clarity and independence.

# -----------------------------
# Default Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 5e-4  # Lowered learning rate for stability
BATCH_SIZE = 128  # Increased batch size for smoother gradients
MEMORY_SIZE = 50_000
INITIAL_EXPLORATION_STEPS = 1_000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9999  # Slower decay for per-step updates
TARGET_UPDATE_STEPS = 500


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        s = np.asarray(s)
        s2 = np.asarray(s2)
        if s.ndim == 2 and s.shape[0] == 1:
            s = s.squeeze(0)
        if s2.ndim == 2 and s2.shape[0] == 1:
            s2 = s2.squeeze(0)
        self.buf.append((s, a, r, s2, 0.0 if done else 1.0))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, m = zip(*batch)
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class DoubleDQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    memory_size: int = 50_000
    initial_exploration: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    
    target_update: int = 0    
    tau: float = 0.005        
    train_freq: int = 4
    device: str = "cpu" 


class DoubleDQNSolver:
    """
    Double DQN Agent with Soft Update and Delayed Training.
    """

    def __init__(self, observation_space: int, action_space: int, cfg: DoubleDQNConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or DoubleDQNConfig()

        self.device = torch.device(self.cfg.device)

        self.online = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.target = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict()) # Init target
        self.target.eval() # Target net need not calc grad

        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.memory = ReplayBuffer(self.cfg.memory_size)

        self.steps = 0
        self.exploration_rate = self.cfg.eps_start

    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        if not evaluation_mode and np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_dim)

        with torch.no_grad():
            s_np = np.asarray(state_np, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            q = self.online(s)
            a = int(torch.argmax(q, dim=1).item())
        return a

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Step function called by training loop.
        """
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1
        
        if self.steps % self.cfg.train_freq == 0:
            self.experience_replay()

        self._decay_eps()

    def experience_replay(self):
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            return

        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t  = torch.as_tensor(m,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # --- Compute Q targets (Double DQN) ---
        with torch.no_grad():
            # 1. Selection: Online net chooses best action for next state
            a_next = self.online(s2_t).argmax(dim=1, keepdim=True)
            # 2. Evaluation: Target net calculates Q value for that action
            q_next = self.target(s2_t).gather(1, a_next)
            
            target = r_t + m_t * self.cfg.gamma * q_next

        # --- Compute Q current ---
        q_sa = self.online(s_t).gather(1, a_t)

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optim.step()

        # [修改] 每次训练后执行 Soft Update，不再等待 500 步
        self.update_target(tau=self.cfg.tau)

    def update_target(self, tau: float = 0.005):
        # Soft update: θ_target = τ*θ_online + (1-τ)*θ_target
        with torch.no_grad():
            for p_t, p in zip(self.target.parameters(), self.online.parameters()):
                p_t.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path: str):
        torch.save(
            {"online": self.online.state_dict(), "cfg": self.cfg.__dict__},
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"]) # Sync target on load

    def _decay_eps(self):
        self.exploration_rate = max(self.cfg.eps_end, self.exploration_rate * self.cfg.eps_decay)
