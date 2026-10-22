# 模型架构与 LoRA / PRM 接入

本项目采用 **Qwen3-4B-Instruct-2507**。本文说明实际模型结构，以及 SFT、策略模型和 PRM 如何使用这一主干。模型结构与 Agent 的环境交互循环是两个层次。

## 1. 从输入到输出

```text
任务 + 历史动作 + 环境观察
→ Tokenizer：token ID
→ Embedding：[B,L,2560]
→ 36 层 Decoder
→ 最终 RMSNorm
→ LM Head：词表 logits
→ 贪心或采样得到下一个 token
```

每个 token 不一定对应一个汉字或单词。推理逐 token 生成；训练时答案或采样轨迹已知，可以并行计算多个位置，因果 mask 保证每个位置看不到未来。

| 配置 | 值 |
|---|---:|
| 类型 | 稠密 Decoder-only Transformer，文本模型 |
| Decoder 层数 | 36 |
| 隐藏维度 | 2560 |
| FFN 中间维度 | 9728 |
| Q / KV 头数 | 32 / 8 |
| 每头维度 | 128 |
| 词表大小 | 151936 |
| 输入与输出嵌入 | 权重共享 |
| 官方最大上下文 | 262144 token |
| 本项目上下文预算 | 4096 token |

该型号为官方 non-thinking 版本。项目中的 `Thought:` 是提示和 SFT 示范形成的文本格式，不代表启用原生 thinking 模式。模型也没有视觉编码器，WebShop 观察以文本提供。

## 2. 单层 Decoder

用列向量简写：

```text
u = x + Attention(RMSNorm(x))
y = u + MLP(RMSNorm(u))
```

RMSNorm 按均方根归一化，控制特征尺度，不先减去均值；残差连接保留信息并帮助梯度传播。Pre-Norm 表示归一化位于子层之前。

注意力在不同位置间交换信息，MLP 对每个位置进行非线性特征变换；同层各位置共享 MLP 参数。

## 3. 注意力维度、GQA 和位置编码

| 投影 | 输入维度 → 输出维度 |
|---|---|
| q_proj | 2560 → 4096，即 32×128 |
| k_proj / v_proj | 2560 → 1024，即 8×128 |
| o_proj | 4096 → 2560 |

**不能用 2560÷32 推断每头维度为 80。** 本配置显式设置 head_dim=128。

Q 表示查询，K 用于匹配，V 是被读取的信息。对 Q/K 做每头 RMSNorm，再通过 RoPE 注入位置信息；注意力计算为：

```text
softmax(QKᵀ / sqrt(128) + causal_mask) V
```

因果 mask 屏蔽未来位置。RoPE 通过位置相关旋转作用于 Q、K，使注意力利用相对位置信息。

GQA 中每 4 个 Q 头共享一组 K/V。与相同头维度、32 个 KV 头的多头注意力相比，KV 缓存元素数为四分之一；不能据此说总显存或全部计算降低四分之三。

## 4. SwiGLU 前馈网络

```text
MLP(x) = W_down [SiLU(W_gate x) ⊙ (W_up x)]
gate_proj / up_proj：2560 → 9728
down_proj：9728 → 2560
```

gate 分支经过 SiLU，与 up 分支逐元素相乘，再投影回主干维度。门控是连续数值运算，不是手写条件判断。

## 5. LoRA 的实际位置和参数量

全部 36 层的七个投影均加 LoRA：

- 注意力：q_proj、k_proj、v_proj、o_proj。
- MLP：gate_proj、up_proj、down_proj。

```text
y = Wx + (alpha/r) BAx
r = 32，alpha = 64
A：[32,d_in]；B：[d_out,32]
```

主干 W 冻结，训练 A/B。每个投影新增 r(d_in+d_out) 个参数；上述配置每层新增 1,835,008 个，共 **66,060,288 个（约 6606 万）**，约占基础模型参数的 1.64%。

注意力适配调整信息检索与融合，MLP 适配调整特征变换。冻结主干并不意味着省略主干前向，也不意味着不需要为 LoRA 梯度传播计算相关激活。

## 6. 策略、参考模型和 PRM

| 角色 | 起点与用途 |
|---|---|
| 策略模型 | 合并后的 SFT 基座 + 新建策略 LoRA；生成动作并更新 |
| 参考模型 | 固定 SFT 起点；提供 KL 约束的参考 logprob |
| PRM | 同一 SFT 基座 + 独立 PRM LoRA + 评分头；策略训练时冻结 |

本项目 PRM 在完整动作末 token 处取 **151936 维词表 logits**，而不是直接在 2560 维隐藏状态上接分类头：

```text
截至当前动作的上下文 → PRM 主干 → 动作末 logits z_k
p_k = wᵀ z_k + b
```

评分头有 151937 个参数，输出允许正负，不加 sigmoid。终局奖励用于监督标签，不作为 PRM 前向输入；只计算所需动作末位置的 logits。实现见 [PRM 模型](../spa_next/prm/model.py)，优势构造见 [算法说明](algorithm.md)。

## 7. 模型与 Agent 的边界

```text
外部程序构造上下文
→ 模型逐 token 生成 Thought / Action
→ 解析器提取 search 或 click
→ WebShop 执行动作
→ 观察加入上下文
→ 下一次调用模型
```

36 层是模型深度；生成 token 数是回复长度；循环次数是环境动作步数。一次完整 Assistant 回复作为一个动作，不通过统计 Thought 字符串次数划分。

KV cache 用于推理时复用历史 K/V。训练的 loss mask 与因果 attention mask 也不同：Human/Observation 不直接计入策略损失，不代表模型不能读取这些上下文。

## 8. 面试回答

> 我使用 Qwen3-4B-Instruct-2507，是一个 36 层的稠密 Decoder-only Transformer，采用 RMSNorm、RoPE、GQA 和 SwiGLU。项目在注意力与前馈网络的七个投影上加入 rank 32 LoRA。策略模型负责生成动作，独立 PRM 在动作末位置评分，再用于构造动作级优势。模型负责文本生成，外部 Agent 程序负责解析动作、执行环境操作和反馈观察。

## 核对来源

- [官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [官方配置](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/config.json)
- [Transformers 4.56.2 Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py)
- [项目训练配置与材料状态](training.md)

官方最大上下文和本项目运行预算应分开描述；模型配置里的 transformers_version 字段也不等于实际训练运行时版本。本文仅补充文档，不更改训练行为或历史实验结论。
