""""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- Trains a DQN agent and logs scores via ScoreLogger (PNG + CSV)
- Saves model to ./models/cartpole_dqn.torch
- Evaluates from a saved model (render optional)

Student reading map:
  1) train(): env loop → agent.act() → env.step() → agent.step() [Encapsulated]
  2) evaluate(): loads saved model and runs agent.act(evaluation_mode=True)
"""

from __future__ import annotations
import os
import time
import argparse
import numpy as np
import gymnasium as gym
import torch

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.bcq import train_bcq as train_bcq_offline, evaluate as evaluate_bcq, collect_bcq_dataset
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
          * Loop: select action, step environment, call agent.step()
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

            # 1. ε-greedy action from the agent (training mode)
            #    state shape is [1, obs_dim]
            action = agent.act(state)

            # 2. Gymnasium step returns: obs', reward, terminated, truncated, info
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Optional small terminal penalty (encourage agent to avoid failure)
            #    Only penalize if it actually failed (terminated), not if it reached the step limit (truncated)
            if terminal_penalty and terminated and not truncated:
                reward = -1.0
            
            # 4. Reshape next_state for agent and next loop iteration
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 5. Give (s, a, r, s', done) to the agent, which handles
            #    remembering and learning internally.
            agent.step(state, action, reward, next_state, done)

            # 6. Move to next state
            state = next_state

            # 7. Episode end: log and break
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
    parser = argparse.ArgumentParser(description="CartPole training / evaluation entrypoint")
    parser.add_argument(
        "-a",
        "--agent",
        "--algorithm",
        dest="agent",
        choices=["dqn", "bcq"],
        default="dqn",
        help="选择使用的算法/agent（dqn 或 bcq）",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "train_eval"],
        default="train_eval",
        help="运行模式：仅训练(train)、仅评估(eval) 或 先训练再评估(train_eval)",
    )
    parser.add_argument(
        "-k",
        "--episodes",
        type=int,
        default=500,
        help="在线训练 DQN 时的训练回合数（对 BCQ 无效）",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="评估时运行的回合数",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="评估时是否渲染环境",
    )
    args = parser.parse_args()

    if args.agent == "dqn":
        # 在线 DQN 训练 + 评估（保持原有行为，只是通过命令行参数控制）
        if args.mode in ("train", "train_eval"):
            train(num_episodes=args.episodes, terminal_penalty=True)
        if args.mode in ("eval", "train_eval"):
            evaluate(
                model_path=MODEL_PATH,
                algorithm="dqn",
                episodes=args.eval_episodes,
                render=args.render,
                fps=60,
            )

    elif args.agent == "bcq":
        # BCQ 是离线算法：先用专家策略收集数据，再离线训练，然后评估
        dataset_path = "./bc_data/ppo_bcq_dataset.npz"
        model_path = "./models/cartpole_bcq.torch"

        if args.mode in ("train", "train_eval"):
            # 如果离线数据集还不存在，先自动调用数据收集
            if not os.path.exists(dataset_path):
                print("[Main] BCQ 数据集未找到，开始用 PPO 专家收集离线数据...")
                collect_bcq_dataset(
                    data_dir="./bc_data",
                    episode_num=100,
                    expert="ppo",
                    model_path="models/cartpole_ppo.torch",
                )

            print("[Main] 开始离线训练 BCQ agent...")
            train_bcq_offline(
                dataset_path=dataset_path,
                model_path=model_path,
            )

        if args.mode in ("eval", "train_eval"):
            print("[Main] 使用训练好的 BCQ 模型进行评估...")
            evaluate_bcq(
                model_path=model_path,
                episodes=args.eval_episodes,
                render=args.render,
            )