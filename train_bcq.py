"""
Script to train BCQ agent on offline dataset.
Run this script to train the BCQ agent after collecting the dataset.
"""

import os
import sys
import csv
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

from agents.bcq import train_bcq, evaluate, BCQConfig
from scores.score_logger import ScoreLogger

if __name__ == "__main__":
    print("=" * 60)
    print("Train BCQ")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = "./data/ppo_bcq_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} not found")
        sys.exit(1)
    
    # Configuration (Hyperparameters)
    cfg = BCQConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=256,
        vae_batch_size=256,
        vae_epochs=50,
        bcq_epochs=200,
        lambda_=0.75,  # BCQ约束系数
        phi=0.05,      # 扰动范围
        tau=0.005,      # 目标网络软更新系数
        vae_lr=1e-3,
        q_lr=1e-3,
        perturb_lr=1e-3,
        device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    print(f"\nUsing: {dataset_path}")
    print(f"Device: {cfg.device}")
    print(f"VAE epochs: {cfg.vae_epochs}")
    print(f"BCQ epochs: {cfg.bcq_epochs}")
    print(f"batch size: {cfg.batch_size}")
    
    # Train BCQ (with periodic evaluation during training)
    model_path = "./models/cartpole_bcq.torch"
    agent = train_bcq(
        dataset_path=dataset_path,
        model_path=model_path,
        cfg=cfg,
        eval_freq=20,  # 每20个epoch评估一次
        eval_episodes=10  # 每次评估运行10个episodes
    )
    
    print("\n" + "=" * 60)
    print(f"model saved to: {model_path}")
    print("=" * 60)
    
    # Evaluate
    print("\nbegin evaluation BCQ agent...\n")
    scores = evaluate(
        model_path=model_path,
        episodes=100,
        render=False,
        save_plots=True
    )
    
    # Save scores to CSV (compatible with ScoreLogger format)
    scores_dir = "./scores"
    os.makedirs(scores_dir, exist_ok=True)
    bcq_scores_csv = os.path.join(scores_dir, "bcq_scores.csv")
    with open(bcq_scores_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for score in scores:
            writer.writerow([score])
    print(f"[BCQ] Scores saved to {bcq_scores_csv}")
    
    # Create comprehensive evaluation plot using ScoreLogger style
    try:
        logger = ScoreLogger("CartPole-v1")
        for i, score in enumerate(scores, 1):
            logger.add_score(score, i)
        print(f"[BCQ] Evaluation plots saved to {scores_dir}/")
    except Exception as e:
        print(f"[BCQ] Warning: Could not create ScoreLogger plots: {e}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print(f"episodes: {len(scores)}")
    print(f"average: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print("=" * 60)

