"""
Script to collect BCQ dataset from PPO expert.
Run this script to generate the offline dataset needed for BCQ training.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, '.')

from agents.bcq import collect_bcq_dataset

if __name__ == "__main__":
    print("=" * 60)
    print("BCQ数据集收集脚本")
    print("=" * 60)
    
    # Check if PPO model exists
    ppo_model_path = "models/cartpole_ppo.torch"
    if not os.path.exists(ppo_model_path):
        print(f"错误: 找不到PPO模型文件 {ppo_model_path}")
        print("请先训练PPO模型或确保模型文件存在。")
        sys.exit(1)
    
    # Collect dataset
    print(f"\n使用PPO专家模型: {ppo_model_path}")
    print("开始收集BCQ数据集（包含完整的transitions: states, actions, rewards, next_states, dones）...\n")
    
    collect_bcq_dataset(
        data_dir="./data",
        episode_num=100,  # 收集100个episode的数据
        expert="ppo",
        model_path=ppo_model_path
    )
    
    print("\n" + "=" * 60)
    print("数据收集完成！")
    print("数据集已保存到: ./data/ppo_bcq_dataset.npz")
    print("现在可以运行 train_bcq() 来训练BCQ agent了。")
    print("=" * 60)

