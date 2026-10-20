# 代码导航

[返回首页](../README.md)

为了保持历史入口和 Python import 稳定，本次整理不移动算法与训练源文件。阅读顺序如下：

| 层次 | 目录或文件 | 职责 |
|---|---|---|
| 核心算法 | [algorithms/](../spa_next/algorithms/) | 回报、动作优势、PPO/KL、权重 |
| Agent交互 | [verl_adapter.py](../spa_next/training/verl_adapter.py)、[agents/](../spa_next/agents/) | 生成调用、token边界、轨迹 |
| 环境接口 | [envs/](../spa_next/envs/) | WebShop动作解析和执行 |
| PRM | [prm/](../spa_next/prm/) | 数据筛选、模型、训练、部署 |
| 训练接入 | [verl_v1.py](../spa_next/training/verl_v1.py) | 优势注入、评分、分卡补位 |
| 策略入口 | [verl_v1_launch.py](../spa_next/training/verl_v1_launch.py) | 参数检查、展开配置、启动训练 |
| 配置 | [configs/](../spa_next/configs/) | V0/V1配置与AgentLoop注册 |
| 验证 | [tests/](../spa_next/tests/) | 数学、数据、接口及三卡一致性 |
| 结果 | [results/](../results/) | 脱敏后的已完成实验汇总 |

## 历史辅助脚本

training 中的 run_*、resume_*、c_first_handoff、finish_prm_only 等用于历史实验监督与恢复，不是每次训练都要执行。patch_* 用于固定版本适配或排查；prm/diagnose_* 用于精度诊断。它们作为工程记录保留，首次阅读从上表核心入口开始即可。

## 仓库不包含什么

模型权重、checkpoint、商品数据、完整交互轨迹、服务器日志与凭据均不发布。results 是汇总证据，不是上述材料的替代品。复现前置条件见[训练说明](training.md)。
