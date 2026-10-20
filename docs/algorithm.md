# 核心算法与实现边界

[返回首页](../README.md) · [训练入口](training.md) · [结果](results.md)

## 定义

同一任务、相同初始状态采样 N=4 条轨迹。i 为轨迹，k 为动作序号，K_i 为完整动作数量。一次完整 Assistant 生成调用（Thought、Action、EOS）算一个动作；不按 Thought 字符串计数。

R_i 是环境终局奖励，p_ik 是 PRM 预测分数。两者不能混为真值。

## 计算流程

1. **PRM 监督。**在动作末 token 的词表 logits 上接线性评分头，预测 p_ik；只读取历史和当前动作。独立训练损失为 `mean((sum_k p_ik - R_i)^2)`。基座冻结，训练独立 LoRA 与评分头；策略训练期间 PRM 冻结。
2. **末步校准。**`delta_i = R_i - sum_k p_ik`；前面动作保持原分数，仅最后一步加 delta_i，保证校准分数之和为 R_i，不裁剪校准分数。
3. **剩余回报。**`G_ik = sum_{l>=k} gamma^(l-k) * calibrated_p_il`。实际 gamma=1，因此 `G_ik = R_i - sum_{l<k} p_il`。
4. **组内标准化。**终局优势 `A_out_i=(R_i-mean(R))/(sample_std(R)+epsilon)`；同深度优势 `A_spa_ik=(G_ik-mean_k(G))/(sample_std_k(G)+epsilon)`。只统计同任务中实际存在第 k 个动作的轨迹，标准差分母为样本数减一。
5. **混合。**正常时 `A_ik=0.8*A_out_i+0.2*A_spa_ik`。各 token 共享所在动作的优势，但有各自的概率比和损失。
6. **策略损失。**`rho_t=exp(log_pi_t-log_pi_old_t)`；`ell_t=-min(rho_t*A_ik, clip(rho_t,0.8,1.2)*A_ik)`，不使用 dual-clip。
7. **KL。**`d_t=clip(log_pi_ref_t-log_pi_t,-20,20)`；`kappa_t=clip(exp(d_t)-d_t-1,0,10)`。这是采样 token 上的惩罚，不是完整词表精确 KL。
8. **聚合。**每个有效 token 权重为 `1/(K_i*|T_ik|)`；对真实轨迹取平均，最终 loss 为动作均衡的 `ell + 0.001*kappa`。固定每任务 N=4 时任务组也等权。

例：R=1，原评分 [0.2,0.3,0.1]，校准后 [0.2,0.3,0.5]，回报 [1,0.8,0.5]。

## 保护规则

| 条件 | 处理 |
|---|---|
| 同深度不足2条或标准差<1e-4 | 该深度使用完整 outcome 优势 |
| 同任务终局标准差<1e-4 | outcome 优势置零，整组禁用 SPA |
| 缺失/非有限评分、数量不符、绝对残差>1 | 整组回退 outcome |
| 不完整动作或上下文不足结束 | 整组回退 outcome |
| PRM 不可用或版本错误 | 报错停止更新 |

epsilon=1e-6。优势和评分不参与策略反向传播。Human、System、Observation、padding、不完整片段不计 loss；完整但格式错误的回复仍作为一次决策。

更新前策略重算的 old logprob 固定；原始生成端 logprob 另存做一致性检查。reference 固定为 SFT 基座，不随更新改变。

## 方法能说明什么、不能说明什么

- 总分 MSE 约束不能唯一确定逐步贡献。
- 同深度动作可能位于不同状态；这里没有做同状态候选动作的分支采样。
- gamma=1 时第一动作回报等于终局奖励；当前位置回报只由终局奖励与此前评分决定。末步原始评分在校准中抵消，但仍影响残差回退判断。
- 动作均衡是优化目标选择，不是 GRPO 的必要条件。
- A（outcome/token平均）、B（outcome/动作平均）、C（混合/动作平均）是已实现的实验分支；已有 V0 对照不等同于完成统一预算的 A/B/C 消融。

## 代码对应

- [action_grpo.py](../spa_next/algorithms/action_grpo.py)：动作边界检查、校准、标准化、回退、权重。
- [losses.py](../spa_next/algorithms/losses.py)：标准 clip、KL、全局分母和跨卡缩放。
- [verl_v1.py](../spa_next/training/verl_v1.py)：收齐任务组后评分，向 actor 传递动作字段。
- [test_v1_algorithms.py](../spa_next/tests/test_v1_algorithms.py)、[test_three_gpu.py](../spa_next/tests/test_three_gpu.py)：定向测试。
