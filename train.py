# train.py
"""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- Trains a DQN agent and logs scores via ScoreLogger (PNG + CSV)
- Saves model to ./models/cartpole_dqn.torch
- Evaluates from a saved model (render optional)

Student reading map:
  1) train(): env loop → agent.act() → env.step() → agent.remember() → agent.experience_replay()
  2) evaluate(): loads saved model and runs greedy policy (no exploration)
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch

from agents.cartpole_dqn import DQNSolver, DQNConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cartpole_dqn.torch")


def train(num_episodes: int = 200, terminal_penalty: bool = True) -> DQNSolver:
    """
    Main training loop:
      - Creates the environment and agent
      - For each episode:
          * Reset env → get initial state
          * Loop: select action, step environment, store transition, learn
          * Log episode score with ScoreLogger
      - Saves the trained model to disk
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create CartPole environment (no render during training for speed)
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    # Infer observation/action dimensions from the env spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Construct agent with default config (students can swap configs here)
    agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    print(f"[Info] Using device: {agent.device}")

    # Episode loop
    for run in range(1, num_episodes + 1):
        # Gymnasium reset returns (obs, info). Seed for repeatability.
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1

            # ε-greedy action from the agent
            action = agent.act(state)

            # Gymnasium step returns: obs', reward, terminated, truncated, info
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Optional small terminal penalty (encourage agent to avoid failure)
            if terminal_penalty and done:
                reward = -1.0

            # Store transition and trigger one learning step
            next_state = np.reshape(next_state, (1, obs_dim))
            agent.remember(state.squeeze(0), action, reward, next_state.squeeze(0), done)
            agent.experience_replay()

            # Move to next state
            state = next_state

            # Episode end: log and break
            if done:
                print(f"Run: {run}, Epsilon: {agent.exploration_rate:.3f}, Score: {steps}")
                logger.add_score(steps, run)  # writes CSV + updates score PNG
                break

    env.close()
    # Persist the trained model
    agent.save(MODEL_PATH)
    print(f"[Train] Model saved to {MODEL_PATH}")
    return agent


def evaluate(model_path: str | None = None,
             algorithm: str = "dqn",
             episodes: int = 5,
             render: bool = True,
             fps: int = 60):
    """
    Evaluate a trained agent in the environment using greedy policy (no ε).
    - Loads weights from disk
    - Optionally renders (pygame window)
    - Reports per-episode steps and average

    Args:
        model_path: If None, auto-pick the first .torch file under ./models
        algorithm: Reserved hook if you later support PPO/A2C agents
        episodes: Number of evaluation episodes
        render: Whether to show a window; set False for headless CI
        fps: Target frame-rate during render (sleep-based pacing)
    """
    # Resolve model path
    model_dir = MODEL_DIR
    if model_path is None:
        candidates = [f for f in os.listdir(model_dir) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError(f"No saved model found in '{model_dir}/'. Please train first.")
        model_path = os.path.join(model_dir, candidates[0])
        print(f"[Eval] Using detected model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    # Create env for evaluation; 'human' enables pygame-based rendering
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # (If you add PPO/A2C later, pick their agent classes by 'algorithm' here.)
    if algorithm.lower() == "dqn":
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    else:
        raise ValueError(...)

    # Load trained weights
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

    scores = []
    # Sleep interval to approximate fps; set 0 for fastest evaluation
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            # Greedy action (no exploration in evaluation)
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device)  # [1, obs_dim]
                q = agent.online(s_t)[0].cpu().numpy()                                  # [act_dim]
                action = int(np.argmax(q))

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
    # Example: quick training then a short evaluation
    agent = train(num_episodes=10, terminal_penalty=True)
    evaluate(model_path="models/cartpole_dqn.torch", algorithm="dqn", episodes=5, render=True, fps=60)