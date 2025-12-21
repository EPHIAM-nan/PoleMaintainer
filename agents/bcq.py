"""
BCQ (Batch Constrained Q-learning) for CartPole (Gymnasium)
--------------------------------------------------------------------
- VAE for learning action distribution from dataset
- Perturbation network for action refinement
- Q-network with target network for value estimation
- Designed to learn from pre-collected demonstrations without online interaction
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 256
VAE_BATCH_SIZE = 256
VAE_EPOCHS = 50
BCQ_EPOCHS = 200
LAMBDA = 0.75  # BCQ constraint coefficient
PHI = 0.05  # Perturbation range (for discrete actions, this controls action selection threshold)
TAU = 0.005  # Soft update coefficient for target network
VAE_LR = 1e-3
Q_LR = 1e-3
PERTURB_LR = 1e-3


# -----------------------------
# Data Collection // same as BC
# -----------------------------
def collect_bcq_dataset(
    data_dir: str = "./data",
    episode_num: int = 100,
    expert: str = "ppo",
    model_path: str = "models/cartpole_ppo.torch"
) -> None:
    """
    Collect complete transitions (s, a, r, s', done) from expert policy.
    
    Args:
        data_dir: Directory to save dataset
        episode_num: Number of episodes to collect
        expert: Expert type ("ppo")
        model_path: Path to expert model
    """
    os.makedirs(data_dir, exist_ok=True)
    
    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Load expert
    if expert.lower() == "ppo":
        from agents.ppo import PPOSolver
        agent = PPOSolver(obs_dim, act_dim)
        agent.load(model_path)
        print(f"[BCQ Data] Loaded PPO expert from {model_path}")
    else:
        raise ValueError(f"Unsupported expert type: {expert}")
    
    # Collect transitions
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []
    total_steps = 0
    
    print(f"[BCQ Data] Collecting transitions for {episode_num} episodes...")
    
    for episode in range(1, episode_num + 1):
        state, _ = env.reset(seed=episode)
        state = np.reshape(state, (1, obs_dim))
        episode_steps = 0
        
        while True:
            episode_steps += 1
            
            # Get expert action
            action = agent.act(state, evaluation_mode=True)
            
            # Step environment
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            states_list.append(state.squeeze())
            actions_list.append(action)
            rewards_list.append(reward)
            next_states_list.append(next_state_raw.squeeze())
            dones_list.append(done)
            
            # Move to next state
            state = np.reshape(next_state_raw, (1, obs_dim))
            
            if done:
                total_steps += episode_steps
                if episode % 10 == 0:
                    print(f"  Episode {episode}/{episode_num}: {episode_steps} steps")
                break
    
    env.close()
    
    # Convert to numpy arrays
    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.int64)
    rewards = np.array(rewards_list, dtype=np.float32)
    next_states = np.array(next_states_list, dtype=np.float32)
    dones = np.array(dones_list, dtype=np.float32)
    
    # Save dataset
    data_path = os.path.join(data_dir, f"{expert}_bcq_dataset.npz")
    np.savez(data_path, 
             states=states, 
             actions=actions, 
             rewards=rewards, 
             next_states=next_states, 
             dones=dones)
    
    print(f"[BCQ Data] Collected {len(states)} transitions from {episode_num} episodes")
    print(f"[BCQ Data] Average steps per episode: {total_steps / episode_num:.2f}")
    print(f"[BCQ Data] Data saved to {data_path}")
    print(f"[BCQ Data] States shape: {states.shape}, Actions shape: {actions.shape}")


# -----------------------------
# VAE for Action Distribution
# -----------------------------
class VAE(nn.Module):
    """
    Variational Autoencoder for learning action distribution from dataset.
    For discrete actions, decoder outputs action logits.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        
        # Encoder: state -> latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_log_std = nn.Linear(64, latent_dim)
        
        # Decoder: state + latent -> action logits
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def encode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state to latent distribution parameters."""
        h = self.encoder(state)
        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h)
        return mean, log_std
    
    def reparameterize(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample from latent distribution."""
        std = torch.exp(log_std.clamp(-20, 2))  # Clamp for numerical stability
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode state + latent to action logits."""
        x = torch.cat([state, latent], dim=1)
        return self.decoder(x)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, sample, decode.
        
        Returns:
            action_logits: Decoded action logits
            mean: Latent mean
            log_std: Latent log std
        """
        mean, log_std = self.encode(state)
        latent = self.reparameterize(mean, log_std)
        action_logits = self.decode(state, latent)
        return action_logits, mean, log_std
    
    def sample_action(self, state: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample actions from VAE.
        
        Args:
            state: [B, obs_dim] or [obs_dim]
            num_samples: Number of action samples per state
        
        Returns:
            action_logits: [B * num_samples, act_dim] or [num_samples, act_dim]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, obs_dim]
        
        batch_size = state.shape[0]
        state_expanded = state.repeat(num_samples, 1)  # [B * num_samples, obs_dim]
        
        mean, log_std = self.encode(state_expanded)
        latent = self.reparameterize(mean, log_std)
        action_logits = self.decode(state_expanded, latent)
        
        if num_samples == 1:
            return action_logits  # [B, act_dim]
        else:
            return action_logits  # [B * num_samples, act_dim]


# -----------------------------
# Perturbation Network
# -----------------------------
class PerturbationNetwork(nn.Module):
    """
    Perturbation network for refining actions generated by VAE.
    For discrete actions, outputs action selection probabilities.
    """
    
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """
        Perturb action.
        
        Args:
            state: [B, obs_dim]
            action_onehot: [B, act_dim]
        
        Returns:
            action_logits: [B, act_dim]
        """
        x = torch.cat([state, action_onehot], dim=1)
        return self.net(x)


# -----------------------------
# Q-Network
# -----------------------------
class QNetwork(nn.Module):
    """Q-network for value estimation."""
    
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
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for all actions."""
        return self.net(state)


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class BCQConfig:
    gamma: float = GAMMA
    lr: float = LR
    batch_size: int = BATCH_SIZE
    vae_batch_size: int = VAE_BATCH_SIZE
    vae_epochs: int = VAE_EPOCHS
    bcq_epochs: int = BCQ_EPOCHS
    lambda_: float = LAMBDA
    phi: float = PHI
    tau: float = TAU
    vae_lr: float = VAE_LR
    q_lr: float = Q_LR
    perturb_lr: float = PERTURB_LR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# BCQ Solver
# -----------------------------
class BCQSolver:
    """
    BCQ (Batch Constrained Q-learning) agent for offline RL.
    """
    
    def __init__(self, observation_space: int, action_space: int, cfg: BCQConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or BCQConfig()
        
        self.device = torch.device(self.cfg.device)
        
        # Networks
        self.vae = VAE(self.obs_dim, self.act_dim).to(self.device)
        self.perturbation = PerturbationNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.q_network = QNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.q_target = QNetwork(self.obs_dim, self.act_dim).to(self.device)
        
        # Initialize target network
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()
        
        # Optimizers
        self.vae_optim = optim.Adam(self.vae.parameters(), lr=self.cfg.vae_lr)
        self.perturb_optim = optim.Adam(self.perturbation.parameters(), lr=self.cfg.perturb_lr)
        self.q_optim = optim.Adam(self.q_network.parameters(), lr=self.cfg.q_lr)
        
        # Counters
        self.steps = 0
        self.exploration_rate = 0.0  # BCQ doesn't use epsilon-greedy
    
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Select action using BCQ policy.
        
        Args:
            state_np: numpy array with shape [1, obs_dim] or [obs_dim]
            evaluation_mode: If True, use deterministic action selection
        
        Returns:
            action: Selected action index
        """
        with torch.no_grad():
            s_np = np.asarray(state_np, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            
            # Generate candidate actions from VAE
            action_logits_vae = self.vae.sample_action(s, num_samples=1)  # [1, act_dim]
            action_probs_vae = F.softmax(action_logits_vae, dim=1)
            
            # Apply perturbation
            action_onehot = F.one_hot(torch.argmax(action_probs_vae, dim=1), self.act_dim).float()
            perturb_logits = self.perturbation(s, action_onehot)
            perturb_probs = F.softmax(perturb_logits, dim=1)
            
            # Combine VAE and perturbation (weighted combination)
            combined_probs = (1 - self.cfg.phi) * action_probs_vae + self.cfg.phi * perturb_probs
            
            # Select action with highest Q-value among candidates
            q_values = self.q_network(s)  # [1, act_dim]
            
            # For BCQ, we select action that maximizes Q among actions with high probability
            # Threshold: only consider actions with prob > lambda
            threshold = self.cfg.lambda_
            mask = (combined_probs > threshold).float()
            
            if mask.sum() > 0:
                # Select action with highest Q among valid actions
                masked_q = q_values * mask + (mask - 1) * 1e8  # Mask invalid actions
                action = torch.argmax(masked_q, dim=1).item()
            else:
                # Fallback: select action with highest probability
                action = torch.argmax(combined_probs, dim=1).item()
            
            return action
    
    def eval(self):
        """Set all networks to evaluation mode."""
        self.vae.eval()
        self.perturbation.eval()
        self.q_network.eval()
        self.q_target.eval()
    
    def train(self):
        """Set all networks to training mode."""
        self.vae.train()
        self.perturbation.train()
        self.q_network.train()
        self.q_target.eval()  # Target network always in eval mode
    
    def save(self, path: str):
        """Save all network weights and config."""
        torch.save(
            {
                "vae": self.vae.state_dict(),
                "perturbation": self.perturbation.state_dict(),
                "q_network": self.q_network.state_dict(),
                "q_target": self.q_target.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )
    
    def load(self, path: str):
        """Load network weights from disk."""
        ckpt = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ckpt["vae"])
        self.perturbation.load_state_dict(ckpt["perturbation"])
        self.q_network.load_state_dict(ckpt["q_network"])
        self.q_target.load_state_dict(ckpt["q_target"])


# -----------------------------
# Training Functions
# -----------------------------
def train_vae(vae: VAE, optimizer: optim.Optimizer, states: np.ndarray, actions: np.ndarray, 
              epochs: int, batch_size: int, device: torch.device) -> list:
    """
    Pre-train VAE on dataset.
    
    Args:
        vae: VAE model to train
        optimizer: Optimizer for VAE
        states: State array
        actions: Action array
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use
    
    Returns:
        loss_history: List of training losses
    """
    vae.train()
    loss_history = []
    
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    
    num_samples = len(states)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(num_samples, device=device)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_states = states_t[batch_indices]
            batch_actions = actions_t[batch_indices]
            
            # Forward pass
            action_logits, mean, log_std = vae(batch_states, batch_actions)
            
            # Reconstruction loss (cross-entropy for discrete actions)
            recon_loss = criterion(action_logits, batch_actions)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_std - mean.pow(2) - log_std.exp(), dim=1).mean()
            
            # Total loss
            loss = recon_loss + 0.5 * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  VAE Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return loss_history


def train_bcq(
    dataset_path: str = "./data/ppo_bcq_dataset.npz",
    model_path: str = "./models/cartpole_bcq.torch",
    cfg: BCQConfig | None = None,
    eval_freq: int = 20,
    eval_episodes: int = 10
) -> BCQSolver:
    """
    Train BCQ agent on offline dataset.
    
    Steps:
    1. Load dataset
    2. Pre-train VAE
    3. Train Q-network and perturbation network
    """
    if cfg is None:
        cfg = BCQConfig()
    
    device = torch.device(cfg.device)
    print(f"[BCQ Training] Using device: {device}")
    
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Please run collect_bcq_dataset() first."
        )
    
    data = np.load(dataset_path)
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_states = data["next_states"]
    dones = data["dones"]
    
    print(f"[BCQ Training] Loaded dataset: {len(states)} transitions")
    
    obs_dim = states.shape[1]
    act_dim = int(actions.max()) + 1
    
    # Create agent
    agent = BCQSolver(obs_dim, act_dim, cfg)
    
    # Step 1: Pre-train VAE
    print("[BCQ Training] Pre-training VAE...")
    vae_losses = train_vae(agent.vae, agent.vae_optim, states, actions, 
                           cfg.vae_epochs, cfg.vae_batch_size, device)
    
    # Step 2: Train Q-network and perturbation network
    print("[BCQ Training] Training Q-network and perturbation network...")
    
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
    
    num_samples = len(states)
    q_losses = []
    perturb_losses = []
    training_scores = []  # 记录训练过程中的评估分数
    training_epochs = []  # 记录评估时的epoch
    
    # 创建环境用于训练过程中的评估
    ENV_NAME = "CartPole-v1"
    eval_env = gym.make(ENV_NAME)
    
    for epoch in range(cfg.bcq_epochs):
        # Shuffle data
        indices = torch.randperm(num_samples, device=device)
        
        epoch_q_loss = 0.0
        epoch_perturb_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, cfg.batch_size):
            batch_indices = indices[i:i + cfg.batch_size]
            batch_states = states_t[batch_indices]
            batch_actions = actions_t[batch_indices]
            batch_rewards = rewards_t[batch_indices]
            batch_next_states = next_states_t[batch_indices]
            batch_dones = dones_t[batch_indices]
            
            # Train Q-network
            with torch.no_grad():
                # Generate candidate actions for next states using VAE
                next_action_logits = agent.vae.sample_action(batch_next_states, num_samples=1)
                next_action_probs = F.softmax(next_action_logits, dim=1)
                
                # Apply perturbation
                next_action_onehot = F.one_hot(torch.argmax(next_action_probs, dim=1), act_dim).float()
                next_perturb_logits = agent.perturbation(batch_next_states, next_action_onehot)
                next_perturb_probs = F.softmax(next_perturb_logits, dim=1)
                
                # Combine
                next_combined_probs = (1 - cfg.phi) * next_action_probs + cfg.phi * next_perturb_probs
                
                # Compute target Q-values
                next_q_values = agent.q_target(batch_next_states)  # [B, act_dim]
                
                # Select actions with high probability (BCQ constraint)
                threshold = cfg.lambda_
                mask = (next_combined_probs > threshold).float()
                
                # Compute max Q among valid actions
                masked_q = next_q_values * mask + (mask - 1) * 1e8
                next_q_max = masked_q.max(dim=1, keepdim=True)[0]
                
                # TD target
                targets = batch_rewards.unsqueeze(1) + (1 - batch_dones.unsqueeze(1)) * cfg.gamma * next_q_max
            
            # Current Q-values
            current_q = agent.q_network(batch_states)
            q_sa = current_q.gather(1, batch_actions.unsqueeze(1))
            
            # Q-loss
            q_loss = F.mse_loss(q_sa, targets)
            
            agent.q_optim.zero_grad()
            q_loss.backward()
            agent.q_optim.step()
            
            epoch_q_loss += q_loss.item()
            
            # Train perturbation network
            # Generate actions from VAE
            vae_action_logits = agent.vae.sample_action(batch_states, num_samples=1)
            vae_action_probs = F.softmax(vae_action_logits, dim=1)
            vae_action_onehot = F.one_hot(torch.argmax(vae_action_probs, dim=1), act_dim).float()
            
            # Perturb
            perturb_logits = agent.perturbation(batch_states, vae_action_onehot)
            perturb_probs = F.softmax(perturb_logits, dim=1)
            
            # Combined
            combined_probs = (1 - cfg.phi) * vae_action_probs + cfg.phi * perturb_probs
            
            # Q-values for current states
            q_vals = agent.q_network(batch_states)
            
            # Perturbation loss: maximize Q-value of selected actions
            # Select actions with high probability
            threshold = cfg.lambda_
            mask = (combined_probs > threshold).float()
            
            if mask.sum() > 0:
                masked_q = q_vals * mask + (mask - 1) * 1e8
                selected_q = masked_q.max(dim=1)[0]
                perturb_loss = -selected_q.mean()  # Maximize Q
            else:
                # Fallback: use action with highest probability
                selected_actions = torch.argmax(combined_probs, dim=1)
                selected_q = q_vals.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
                perturb_loss = -selected_q.mean()
            
            agent.perturb_optim.zero_grad()
            perturb_loss.backward()
            agent.perturb_optim.step()
            
            epoch_perturb_loss += perturb_loss.item()
            num_batches += 1
            
            # Soft update target network
            with torch.no_grad():
                for param, target_param in zip(agent.q_network.parameters(), agent.q_target.parameters()):
                    target_param.data.mul_(1 - cfg.tau).add_(cfg.tau * param.data)
        
        avg_q_loss = epoch_q_loss / num_batches if num_batches > 0 else 0.0
        avg_perturb_loss = epoch_perturb_loss / num_batches if num_batches > 0 else 0.0
        q_losses.append(avg_q_loss)
        perturb_losses.append(avg_perturb_loss)
        
        # evaluate performance
        if (epoch + 1) % eval_freq == 0 or epoch == 0:
            agent.eval()
            eval_scores = []
            for _ in range(eval_episodes):
                state, _ = eval_env.reset()
                state = np.reshape(state, (1, obs_dim))
                done = False
                steps = 0
                while not done:
                    action = agent.act(state, evaluation_mode=True)
                    next_state, _, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    state = np.reshape(next_state, (1, obs_dim))
                    steps += 1
                eval_scores.append(steps)
            agent.train()
            
            avg_eval_score = np.mean(eval_scores)
            training_scores.append(avg_eval_score)
            training_epochs.append(epoch + 1)
            print(f"  BCQ Epoch {epoch + 1}/{cfg.bcq_epochs}, "
                  f"Q Loss: {avg_q_loss:.4f}, Perturb Loss: {avg_perturb_loss:.4f}, "
                  f"Eval Score: {avg_eval_score:.1f}")
        else:
            if (epoch + 1) % 20 == 0:
                print(f"  BCQ Epoch {epoch + 1}/{cfg.bcq_epochs}, "
                      f"Q Loss: {avg_q_loss:.4f}, Perturb Loss: {avg_perturb_loss:.4f}")
    
    eval_env.close()
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    print(f"[BCQ Training] Model saved to {model_path}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # VAE Loss
    axes[0].plot(vae_losses, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('VAE Training Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if vae_losses:
        axes[0].axhline(y=np.mean(vae_losses[-10:]), color='r', linestyle='--', 
                       alpha=0.5, label=f'Final Avg: {np.mean(vae_losses[-10:]):.4f}')
        axes[0].legend(fontsize=9)
    
    # Q-Network Loss
    axes[1].plot(q_losses, 'g-', linewidth=1.5)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Q-Network Training Loss', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    if q_losses:
        # Plot moving average for smoother visualization
        window = max(1, len(q_losses) // 20)
        if window > 1:
            q_smooth = [np.mean(q_losses[max(0, i-window):i+1]) for i in range(len(q_losses))]
            axes[1].plot(q_smooth, 'g--', alpha=0.7, linewidth=1, label='Moving Avg')
        axes[1].axhline(y=np.mean(q_losses[-10:]), color='r', linestyle='--', 
                       alpha=0.5, label=f'Final Avg: {np.mean(q_losses[-10:]):.4f}')
        axes[1].legend(fontsize=9)
    
    # Perturbation Network Loss
    axes[2].plot(perturb_losses, 'm-', linewidth=1.5)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].set_title('Perturbation Network Loss', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    if perturb_losses:
        window = max(1, len(perturb_losses) // 20)
        if window > 1:
            p_smooth = [np.mean(perturb_losses[max(0, i-window):i+1]) for i in range(len(perturb_losses))]
            axes[2].plot(p_smooth, 'm--', alpha=0.7, linewidth=1, label='Moving Avg')
        axes[2].axhline(y=np.mean(perturb_losses[-10:]), color='r', linestyle='--', 
                       alpha=0.5, label=f'Final Avg: {np.mean(perturb_losses[-10:]):.4f}')
        axes[2].legend(fontsize=9)
    
    plt.tight_layout()
    os.makedirs("./models", exist_ok=True)
    plt.savefig('./models/bcq_training_losses.png', dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[BCQ Training] Training curves saved to ./models/bcq_training_losses.png")
    
    # Performance variation plot
    if training_scores:
        plot_training_progress(training_epochs, training_scores, 
                              save_path="./scores/bcq_training_progress.png")
    
    return agent


# same in score_logger.py
def plot_training_progress(epochs: list, scores: list, save_path: str = "./scores/bcq_training_progress.png"):
    """
    Plot the performance variation during training (similar to online learning training progress plot)
    
    Args:
        epochs: List of epochs when evaluation was performed
        scores: Corresponding evaluation scores list
        save_path: Save path
    """
    if not scores or len(scores) == 0:
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # score
    plt.plot(epochs, scores, 'b-o', linewidth=2, markersize=6, label="Evaluation Score")
    
    # average
    if len(scores) > 1:
        window = min(3, len(scores) // 2)
        if window > 1:
            moving_avg = []
            for i in range(len(scores)):
                start = max(0, i - window + 1)
                moving_avg.append(np.mean(scores[start:i+1]))
            plt.plot(epochs, moving_avg, 'r--', linewidth=2, alpha=0.7, 
                    label=f"Moving Average (window={window})")
    
    # target
    goal = 475
    plt.axhline(y=goal, color='orange', linestyle=':', linewidth=2, 
                label=f"Goal ({goal} Avg)")
    
    # final
    if len(scores) > 0:
        final_avg = np.mean(scores[-min(5, len(scores)):])
        plt.axhline(y=final_avg, color='g', linestyle='--', linewidth=2, 
                   label=f"Final Average: {final_avg:.1f}")
    
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("Evaluation Score", fontsize=12)
    plt.title("BCQ - Training Progress (Offline Learning)", fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"[BCQ Training] Training progress plot saved to {save_path}")


def evaluate(
    model_path: str = "./models/cartpole_bcq.torch",
    episodes: int = 100,
    render: bool = False,
    fps: int = 60,
    save_plots: bool = True
) -> list:
    """
    Evaluate trained BCQ agent.
    
    Args:
        model_path: Path to saved model
        episodes: Number of evaluation episodes
        render: Whether to render environment
        fps: Frame rate for rendering
        save_plots: Whether to save evaluation plots
    
    Returns:
        scores: List of episode scores
    """
    import time
    
    ENV_NAME = "CartPole-v1"
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Load agent
    agent = BCQSolver(obs_dim, act_dim)
    agent.load(model_path)
    print(f"[BCQ Eval] Loaded model from {model_path}")
    
    scores = []
    dt = (1.0 / fps) if render and fps else 0.0
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0
        
        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1
            
            if dt > 0:
                time.sleep(dt)
        
        scores.append(steps)
        if ep % 10 == 0:
            print(f"[BCQ Eval] Episode {ep}/{episodes}: steps={steps}")
    
    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    std = float(np.std(scores)) if scores else 0.0
    print(f"[BCQ Eval] Average over {episodes} episodes: {avg:.2f} ± {std:.2f}")
    
    # Plot evaluation results
    if save_plots:
        os.makedirs("./scores", exist_ok=True)
        plot_evaluation_results(scores, save_path="./scores/bcq_evaluation.png")
    
    return scores


def plot_evaluation_results(scores: list, save_path: str = "./scores/bcq_evaluation.png"):
    """
    Plot evaluation results similar to ScoreLogger.
    
    Args:
        scores: List of episode scores
        save_path: Path to save the plot
    """
    if not scores:
        return
    
    episodes = list(range(1, len(scores) + 1))
    avg_score = np.mean(scores)
    
    plt.figure(figsize=(10, 6))
    
    # Plot individual scores
    plt.plot(episodes, scores, 'b-', alpha=0.3, linewidth=0.5, label="Score per Episode")
    
    # Plot moving average
    window = min(20, len(scores) // 5)
    if window > 1:
        moving_avg = []
        for i in range(len(scores)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(scores[start:i+1]))
        plt.plot(episodes, moving_avg, 'r-', linewidth=2, label=f"Moving Average (window={window})")
    
    # Plot overall average
    plt.axhline(y=avg_score, color='g', linestyle='--', linewidth=2, 
                label=f"Overall Average: {avg_score:.2f}")
    
    # Plot goal line (CartPole solved threshold)
    goal = 475
    plt.axhline(y=goal, color='orange', linestyle=':', linewidth=2, 
                label=f"Goal ({goal} Avg)")
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("BCQ - Evaluation Results", fontsize=14, fontweight='bold')
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"[BCQ Eval] Evaluation plot saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    # Step 1: Collect dataset (if not exists)
    dataset_path = "./data/ppo_bcq_dataset.npz"
    if not os.path.exists(dataset_path):
        print("[BCQ] Collecting dataset...")
        collect_bcq_dataset(
            data_dir="./data",
            episode_num=100,
            expert="ppo",
            model_path="models/cartpole_ppo.torch"
        )
    
    # Step 2: Train BCQ
    print("[BCQ] Training BCQ agent...")
    agent = train_bcq(
        dataset_path=dataset_path,
        model_path="./models/cartpole_bcq.torch"
    )
    
    # Step 3: Evaluate
    print("[BCQ] Evaluating BCQ agent...")
    evaluate(
        model_path="./models/cartpole_bcq.torch",
        episodes=100,
        render=False
    )

