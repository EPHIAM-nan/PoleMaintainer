from __future__ import annotations
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from collections import deque

@dataclass
class RainbowConfig:
    gamma: float = 0.99
    lr: float = 1e-4  # Rainbow通常需要较小的学习率
    batch_size: int = 32
    memory_size: int = 20000
    target_update: int = 200  # 硬更新频率
    
    # Noisy Nets
    sigma_init: float = 0.5
    
    # Multi-step Learning
    n_step: int = 3
    
    # Distributional RL (C51)
    v_min: float = 0.0
    v_max: float = 500.0
    atom_size: int = 51
    
    # Prioritized Experience Replay
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 20000 # beta 线性增长到 1.0 所需步数
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        # Factorized Gaussian Noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowNet(nn.Module):
    """Combines Dueling Network + Noisy Nets + Distributional RL (C51)."""
    def __init__(self, obs_dim, act_dim, atom_size, sigma_init):
        super(RainbowNet, self).__init__()
        self.act_dim = act_dim
        self.atom_size = atom_size

        # Feature Layer
        self.fc = nn.Linear(obs_dim, 128)
        
        # Dueling Architecture: Advantage Stream
        self.adv_hidden = NoisyLinear(128, 128, sigma_init)
        self.adv_out = NoisyLinear(128, act_dim * atom_size, sigma_init)
        
        # Dueling Architecture: Value Stream
        self.val_hidden = NoisyLinear(128, 128, sigma_init)
        self.val_out = NoisyLinear(128, atom_size, sigma_init)

    def forward(self, x):
        x = F.relu(self.fc(x))
        
        adv = F.relu(self.adv_hidden(x))
        adv = self.adv_out(adv)
        adv = adv.view(-1, self.act_dim, self.atom_size) # [batch, act, atom]
        
        val = F.relu(self.val_hidden(x))
        val = self.val_out(val)
        val = val.view(-1, 1, self.atom_size) # [batch, 1, atom]
        
        # Dueling Combination: Q = V + A - mean(A)
        q_logits = val + adv - adv.mean(dim=1, keepdim=True)
        
        # Return Logits or Probabilities? C51 usually works with Logits -> Softmax
        dist = F.softmax(q_logits, dim=2)
        return dist # [batch, act, atom]

    def reset_noise(self):
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()
        self.val_hidden.reset_noise()
        self.val_out.reset_noise()

class SumTree:
    """Segment Tree for efficient Prioritized Replay."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_total(self):
        return self.tree[0]

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: max_p = 1.0
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        batch_idx, batch, weights = [], [], []
        segment = self.tree.get_total() / batch_size
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.get_total()
        max_weight = (p_min * self.capacity) ** (-beta) if p_min > 0 else 1.0 # 避免除零

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            # Importance Sampling Weight
            prob = p / self.tree.get_total()
            weight = (self.capacity * prob) ** (-beta)
            
            weights.append(weight / max_weight)
            batch_idx.append(idx)
            batch.append(data)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), 
                np.array(weights), batch_idx)

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.tree.update(idx, p ** self.alpha)
    
    def __len__(self):
        return self.tree.count

class RainbowSolver(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg: RainbowConfig = None):
        super().__init__()
        self.cfg = cfg or RainbowConfig()
        self.device = torch.device(self.cfg.device)
        self.act_dim = act_dim
        
        # Networks
        self.online = RainbowNet(obs_dim, act_dim, self.cfg.atom_size, self.cfg.sigma_init).to(self.device)
        self.target = RainbowNet(obs_dim, act_dim, self.cfg.atom_size, self.cfg.sigma_init).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        
        # Memory & N-step
        self.memory = PrioritizedReplayBuffer(self.cfg.memory_size, self.cfg.alpha)
        self.use_n_step = self.cfg.n_step > 1
        self.n_step_buffer = deque(maxlen=self.cfg.n_step)
        
        # C51 Support (Z)
        self.support = torch.linspace(self.cfg.v_min, self.cfg.v_max, self.cfg.atom_size).to(self.device)
        self.delta_z = (self.cfg.v_max - self.cfg.v_min) / (self.cfg.atom_size - 1)
        
        self.beta = self.cfg.beta_start
        self.step_count = 0

    def act(self, state, evaluation_mode=False):
        # No greedy epsilon, exploration is handled by Noisy Nets
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Q(s,a) = sum_z z * p(z|s,a)
            dist = self.online(state) # [1, act, atom]
            expected_q = torch.sum(dist * self.support, dim=2) # [1, act]
            action = expected_q.argmax().item()
        return action

    def step(self, state, action, reward, next_state, done):
        # N-step buffer logic
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.cfg.n_step:
            state_0, action_0, reward_n, next_state_n, done_n = self._get_n_step_info()
            self.memory.push(state_0, action_0, reward_n, next_state_n, done_n)
        
        # 如果 episode 结束，清空剩余 buffer
        if done and len(self.n_step_buffer) > 0:
             # 为了简单起见，CartPole这种短周期的环境，我们可以选择丢弃或者清空
             # 严谨做法是把 buffer 里剩下的都存进去，这里简化处理直接清空
             self.n_step_buffer.clear()

        # Update
        if len(self.memory) > self.cfg.batch_size:
            self.update()
            
        # Target network hard update
        if self.step_count % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
            
        self.step_count += 1

    def _get_n_step_info(self):
        """计算 N-step return"""
        reward, next_state, done = 0, None, False
        for i in range(self.cfg.n_step):
            _, _, r, n_s, d = self.n_step_buffer[i]
            reward += r * (self.cfg.gamma ** i)
            if d:
                done = True
                next_state = n_s # Terminal state
                break
            next_state = n_s
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], reward, next_state, done

    def update(self):
        # Anneal beta
        self.beta = min(1.0, self.beta + (1.0 - self.cfg.beta_start) / self.cfg.beta_frames)
        
        # Sample
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.cfg.batch_size, self.beta)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # ------------------- Categorical DQN Algorithm -------------------
        # 1. Next State Distribution (Target Net)
        with torch.no_grad():
            # Double DQN: action selection using Online Net
            next_dist_online = self.online(next_states)
            next_actions = (next_dist_online * self.support).sum(2).argmax(1) # [batch]
            
            # Evaluate using Target Net
            next_dist = self.target(next_states) # [batch, act, atom]
            next_dist = next_dist[range(self.cfg.batch_size), next_actions] # [batch, atom]

            # 2. Projection (Bellman Update for Distributions)
            t_z = rewards + (1 - dones) * (self.cfg.gamma ** self.cfg.n_step) * self.support
            t_z = t_z.clamp(min=self.cfg.v_min, max=self.cfg.v_max)
            
            # Map back to support indices
            b = (t_z - self.cfg.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Fix indices out of bounds (due to floating point errors)
            l = l.clamp(min=0, max=self.cfg.atom_size - 1)
            u = u.clamp(min=0, max=self.cfg.atom_size - 1)

            # Distribute probability mass
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            offset = torch.linspace(0, (self.cfg.batch_size - 1) * self.cfg.atom_size, self.cfg.batch_size).long().unsqueeze(1).to(self.device)
            
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        # 3. Current State Distribution (Online Net)
        dist = self.online(states)
        log_p = torch.log(dist[range(self.cfg.batch_size), actions]) # [batch, atom]

        # 4. Loss (KL Divergence / Cross Entropy)
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * logP) - sum(P * logQ)
        # Minimize -sum(Target * log(Online))
        elementwise_loss = -(proj_dist * log_p).sum(1)
        
        # PER Weighted Loss
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping is often useful for stability
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0) 
        self.optimizer.step()

        # Update Priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, loss_for_prior + 1e-6)
        
        # Reset Noisy Nets Noise
        self.online.reset_noise()
        self.target.reset_noise()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))