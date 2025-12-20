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
    print("BCQ训练脚本")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = "./data/ppo_bcq_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集文件 {dataset_path}")
        print("请先运行 collect_bcq_data.py 来收集数据。")
        sys.exit(1)
    
    # Configuration (可以调整这些超参数)
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
    
    print(f"\n使用数据集: {dataset_path}")
    print(f"设备: {cfg.device}")
    print(f"VAE训练轮数: {cfg.vae_epochs}")
    print(f"BCQ训练轮数: {cfg.bcq_epochs}")
    print(f"批次大小: {cfg.batch_size}")
    print("\n开始训练BCQ agent...\n")
    
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
    print("训练完成！")
    print(f"模型已保存到: {model_path}")
    print("=" * 60)
    
    # Evaluate
    print("\n开始评估训练好的BCQ agent...\n")
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
    print("评估完成！")
    print(f"总episodes: {len(scores)}")
    print(f"平均分数: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"最高分数: {max(scores)}")
    print(f"最低分数: {min(scores)}")
    print(f"中位数: {np.median(scores):.2f}")
    print(f"达到500步的episodes: {sum(1 for s in scores if s >= 500)} ({sum(1 for s in scores if s >= 500)/len(scores)*100:.1f}%)")
    print("=" * 60)

