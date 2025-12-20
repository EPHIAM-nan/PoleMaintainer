from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

@dataclass
class A2CConfig:
    gamma: float = 0.99
    actor_lr: float = 1e-2
    critic_lr: float = 1e-1
    hidden_dim: int = 128
    entropy_coef: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class VNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class A2CSolver(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg: A2CConfig = None):
        super().__init__()
        self.cfg = cfg or A2CConfig()
        self.gamma = self.cfg.gamma
        self.device = torch.device(self.cfg.device)
        
        self.actor = PolicyNet(obs_dim, self.cfg.hidden_dim, act_dim).to(self.device)
        self.critic = VNet(obs_dim, self.cfg.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        
        self.transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }

    def act(self, state, evaluation_mode=False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        if evaluation_mode:
            action = torch.argmax(probs, dim=1)
        else:
            action = action_dist.sample()
        return action.item()

    def step(self, state, action, reward, next_state, done):
        self.transition_dict['states'].append(state.flatten())
        self.transition_dict['actions'].append(action)
        self.transition_dict['next_states'].append(next_state.flatten())
        self.transition_dict['rewards'].append(reward)
        self.transition_dict['dones'].append(done)

        if done:
            self.update(self.transition_dict)
            self.transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # TD Target = r + gamma * V(s') * (1 - done)
        next_values = self.critic(next_states)
        td_target = rewards + self.gamma * next_values * (1 - dones)
        current_values = self.critic(states)
        
        critic_loss = F.mse_loss(current_values, td_target.detach())

        # 2. calculate Actor Loss
        td_error = td_target - current_values
        
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions.squeeze())
        
        # Calculate entropy: H(π(·|s))
        dist_entropy = dist.entropy()
        
        # Actor Loss = -mean(log_prob * advantage) - beta * mean(entropy)
        # Subtracting entropy term to maximize entropy (since Loss is minimized)
        actor_loss = torch.mean(-log_probs * td_error.detach().squeeze() - self.cfg.entropy_coef * dist_entropy)

        # 3. Backpropagation
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))