# AIB Project : CartPole
汪士楠 郑仁杰 杨一鸣

# 实现算法

## Online

| Method | code | model |
| ---    | ---  | ---   |
| DQN | `agents/cartpole_dqn.py` | `models/cartpole_dqn.torch` |
| PPO | `agents/ppo.py` | `models/cartpole_ppo.torch` |
| A2C | `agents/cartpole_a2c.py` | `models/cartpole_a2c.torch` |
| Rainbow | `agents/cartpole_rainbow.py` | `models/cartpole_rainbow.torch` |

## Offline

| Method | code | model | dataset |
| --- | --- | --- | --- |
| BC | `agents/bc.py` | `models/bc_cartpole.torch` | `data/ppo_demonstrations.npz` |
| BCQ | `agents/bcq.py` | `models/` | `data/ppo_bcq_dataset.npz` | 
| World Model | `agents/worldmodel.py` | `models/wm_controller.torch` & `models/wm_dynamics.torch` | `data/world_model_dataset.npz` |

`ppo_demonstrations.npz` 和 `ppo_bcq_dataset.npz` 是使用 `models/cartpole_ppo.py` 收集的，`world_model_dataset.npz` 由随机 action 收集得到。

## 训练脚本

`train.py` 主要专注于 DQN 的训练，且如果要使其支持所有模型的训练需要修改的难度较大。于是我们组采用了“直接修改`train.py`+不 commit“的方式，因此部分模型的训练脚本丢失(标记为lost)。已有的训练程序：

| Method | train script | 
| DQN | `train.py` |
| PPO | `trainppo.py` |
| A2C | lost |
| Rainbow | lost |
| BC | `agents/bc.py` |
| BCQ | `train_bcq.py` |
| World Model | `agents/worldmodel.py` |

# Evaluate 一键评测

`evaluate.py` 可以在 cartpole 环境中测试 100 episodes 内的平均分。

**使用方法**

```bash
python evaluate --agent all
```

将测试所有实现的模型的得分并汇总为表格。

也可以通过 `--agent` 选择 agent，每个模型对应的编号为：

| method | `--agent` |
| --- | --- |
| DQN | dqn |
| PPO | ppo |
| A2C | a2c | 
| Rainbow | rainbow |
| BC | bc |
| BCQ | bcq |
| World Model | wm |

