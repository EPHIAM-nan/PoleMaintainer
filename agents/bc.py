# Behavior Cloning -> PPO

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# =========

MLP_LATENT_DIM = 8
TRAIN_EPOCHS = 100
BATCH_SIZE = 64

# =========

def generate_data(
    data_dir: str = "./data",
    episode_num: int = 100, # one episode 500 steps, dataset = episode_num * 500 
    expert: str = "ppo", # 装个样子☝️🤓, 实际只支持 ppo
    model_path: str = "models/cartpole_ppo.torch"
) -> None:

    os.makedirs(data_dir, exist_ok=True)
    
    # env
    ENV_NAME = "CartPole-v1"
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # expert
    if expert.lower() == "ppo":
        from ppo import PPOSolver
        agent = PPOSolver(obs_dim, act_dim)
        agent.load(model_path)
        print(f"[BC Data] Loaded PPO expert from {model_path}")
    else:
        raise ValueError(f"Unsupported expert type: {expert}")
    
    # Collect demonstration data
    states_list = []
    actions_list = []
    total_steps = 0
    
    print(f"[BC Data] Collecting demonstrations for {episode_num} episodes...")
    
    for episode in range(1, episode_num + 1):
        state, _ = env.reset(seed=episode)
        state = np.reshape(state, (1, obs_dim))
        episode_steps = 0
        
        while True:
            episode_steps += 1
            
            # Get expert action (evaluation mode = deterministic/greedy)
            action = agent.act(state, evaluation_mode=True)
            
            # Store state-action pair
            # Squeeze state to [obs_dim] for storage
            states_list.append(state.squeeze())
            actions_list.append(action)
            
            # Step environment
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Reshape for next iteration
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
    
    # Save data
    data_path = os.path.join(data_dir, f"{expert}_demonstrations.npz")
    np.savez(data_path, states=states, actions=actions)
    
    print(f"[BC Data] Collected {len(states)} transitions from {episode_num} episodes")
    print(f"[BC Data] Average steps per episode: {total_steps / episode_num:.2f}")
    print(f"[BC Data] Data saved to {data_path}")
    print(f"[BC Data] States shape: {states.shape}, Actions shape: {actions.shape}")
    
class MLP(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, act_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class BCSolver:
    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(obs_dim=obs_dim, act_dim=act_dim, latent_dim=MLP_LATENT_DIM).to(self.device)

    def act(self, state_np: np.ndarray, evaluation_mode: bool = True) -> int:
        with torch.no_grad():
            s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
            logits = self.model(s)
            action = int(torch.argmax(logits, dim=1).item())
        return action

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt)

def train_bc():
    loss = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(4, 2, latent_dim=MLP_LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    data = np.load("./data/ppo_demonstrations.npz")
    states = data["states"]
    actions = data["actions"]

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)

    batch_size = 64
    num_epochs = TRAIN_EPOCHS
    num_samples = states.shape[0]
    loss_list = []

    for epoch in range(1, num_epochs + 1):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_states = states_tensor[indices]
            batch_actions = actions_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_states)
            batch_loss = loss(outputs, batch_actions)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * batch_states.size(0)

        epoch_loss /= num_samples
        loss_list.append(epoch_loss)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step()
    
    model_path = "./models/bc_cartpole.torch"
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[BC Training] Model saved to {model_path}")

    # Plot and save loss
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, num_epochs + 1), loss_list)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('BC Training Loss', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.savefig('./models/bc_loss.png')
    print(f"[BC Training] Loss plot saved to ./models/bc_loss.png")



def evaluate(episodes: int = 100,
             render: bool = False,
             fps: int = 60):
    
    import time

    model_path = "./models/bc_cartpole.torch"
    ENV_NAME = "CartPole-v1"

    # Create env for evaluation; 'human' enables pygame-based rendering
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = BCSolver(obs_dim, act_dim)
    agent.load(model_path)
    
    scores = []
    # Sleep interval to approximate fps; set 0 for fastest evaluation
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            # Greedy action (no exploration) by calling act() in evaluation mode
            action = agent.act(state, evaluation_mode=True)

            # Step env forward
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            # Slow down rendering to be watchable
            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores



if __name__ == "__main__":
    train_bc()
    evaluate(episodes=100, render=False, fps=60)