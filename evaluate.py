"""
Unified Evaluation Script for CartPole Agents
---------------------------------------------
Supports: DQN, PPO, A2C, Rainbow, BC, BCQ, and World Model (WM).
Usage: python evaluate.py --agent dqn --render
"""

import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym

# Import agents from the agents/ directory
from agents.cartpole_dqn import DQNSolver
from agents.ppo import PPOSolver
from agents.cartpole_a2c import A2CSolver
from agents.cartpole_rainbow import RainbowSolver
from agents.bc import BCSolver
from agents.bcq import BCQSolver
from agents.worldmodel import Controller

# Mapping of agent types to their respective classes and default model paths
AGENT_MAP = {
    "dqn": {"class": DQNSolver, "path": "models/cartpole_dqn.torch"},
    "ppo": {"class": PPOSolver, "path": "models/cartpole_ppo.torch"},
    "a2c": {"class": A2CSolver, "path": "models/cartpole_a2c.torch"},
    "rainbow": {"class": RainbowSolver, "path": "models/cartpole_rainbow.torch"},
    "bc": {"class": BCSolver, "path": "models/bc_cartpole.torch"},
    "bcq": {"class": BCQSolver, "path": "models/cartpole_bcq.torch"},
    "wm": {"class": Controller, "path": "models/wm_controller.torch"},
}

def get_agent(agent_type, obs_dim, act_dim, model_path):
    """Initialize and load the agent based on type."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_info = AGENT_MAP[agent_type]
    
    if agent_type == "wm":
        # World Model Controller uses standard state_dict loading
        agent = agent_info["class"](obs_dim, act_dim).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
    else:
        # Other agents have a custom .load() method
        agent = agent_info["class"](obs_dim, act_dim)
        agent.load(model_path)
        if hasattr(agent, "eval"):
            agent.eval()
            
    return agent

def run_evaluation(agent, agent_type, episodes=10, render=False, fps=60):
    """Run the evaluation loop."""
    env_name = "CartPole-v1"
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    
    scores = []
    dt = (1.0 / fps) if render and fps else 0.0
    
    print(f"\n[Eval] Starting evaluation for {agent_type}...")
    print(f"[Eval] Render: {render}, Episodes: {episodes}")

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10000 + ep)
        done = False
        steps = 0
        
        while not done:
            # Handle different input requirements for act()
            if agent_type == "wm":
                # WM Controller expects a torch tensor
                s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                # Move to same device as agent
                s_tensor = s_tensor.to(next(agent.parameters()).device)
                with torch.no_grad():
                    action = agent.act(s_tensor)
            else:
                # Most agents expect a numpy array with shape [1, obs_dim]
                state_input = np.reshape(state, (1, obs_dim))
                action = agent.act(state_input, evaluation_mode=True)
            
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if dt > 0:
                time.sleep(dt)
        
        scores.append(steps)
        print(f"  Episode {ep:3d}: Score = {steps:3d}")
    
    env.close()
    avg_score = np.mean(scores)
    print(f"\n[Eval] Finished! Average Score: {avg_score:.2f}")
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CartPole Agents")
    parser.add_argument("-a", "--agent", type=str, default="dqn", 
                        choices=list(AGENT_MAP.keys()) + ["all"],
                        help="Agent type to evaluate (default: dqn, or 'all' to run all)")
    parser.add_argument("-m", "--model", type=str, default=None, 
                        help="Path to the model file (overrides default, ignored if agent is 'all')")
    parser.add_argument("-e", "--episodes", type=int, default=100, 
                        help="Number of episodes to evaluate (default: 100)")
    parser.add_argument("-r", "--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("-f", "--fps", type=int, default=60, 
                        help="Target FPS for rendering (default: 60)")
    
    args = parser.parse_args()
    
    # Get environment dimensions
    temp_env = gym.make("CartPole-v1")
    obs_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.n
    temp_env.close()

    agents_to_eval = list(AGENT_MAP.keys()) if args.agent == "all" else [args.agent]
    results = {}
    
    for agent_type in agents_to_eval:
        # Determine model path
        model_path = args.model if (args.model and args.agent != "all") else AGENT_MAP[agent_type]["path"]
        
        if not os.path.exists(model_path):
            print(f"\n[Skip] Model file not found for {agent_type} at {model_path}")
            if args.agent != "all":
                print(f"Please ensure the model is trained and saved in the models/ directory.")
                exit(1)
            results[agent_type] = "N/A (Model not found)"
            continue
            
        try:
            # Initialize agent
            agent = get_agent(agent_type, obs_dim, act_dim, model_path)
            
            # Run evaluation
            avg_score = run_evaluation(agent, agent_type, episodes=args.episodes, render=args.render, fps=args.fps)
            results[agent_type] = f"{avg_score:.2f}"
            
        except Exception as e:
            print(f"An error occurred during evaluation of {agent_type}: {e}")
            results[agent_type] = "Error"
            if args.agent != "all":
                import traceback
                traceback.print_exc()

    # Print summary table
    print("\n" + "="*30)
    print(f"{'Agent Type':<15} | {'Average Score':<15}")
    print("-" * 30)
    for agent_type, score in results.items():
        print(f"{agent_type:<15} | {score:<15}")
    print("="*30 + "\n")
