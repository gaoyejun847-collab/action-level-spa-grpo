# 训练入口与复现条件

[返回首页](../README.md) · [算法](algorithm.md) · [代码导航](code-map.md)

## 前置条件

本仓库是上游 SPA 项目的新增实现。先准备 [SPA-RL-Agent](https://github.com/WangHanLinHenry/SPA-RL-Agent) 的环境源码、提示、任务索引、WebShop 商品数据与搜索索引，再把本仓库的 spa_next 复制到上游工作目录。

训练与 WebShop 环境分开安装，依赖见 [requirements-train.txt](../spa_next/requirements-train.txt) 和 [requirements-env.txt](../spa_next/requirements-env.txt)。固定核心版本为 PyTorch 2.8.0、Transformers 4.56.2、PEFT 0.17.1、verl 0.7.0、vLLM 0.10.2、Ray 2.48.0。

还需要自行准备模型、冻结划分清单、SFT 示范及 PRM 轨迹。已删除的原实验 checkpoint 未发布，本文不是从空环境一键复现的承诺。

## 入口表

| 阶段 | 入口 | 说明 |
|---|---|---|
| SFT | [sft.py](../spa_next/training/sft.py) | 数据准备与 Assistant-only 监督训练 |
| 合并 SFT | [sft_snapshot.py](../spa_next/training/sft_snapshot.py) | 生成、校验冻结基座及版本 |
| PRM | [prm/__main__.py](../spa_next/prm/__main__.py) | prepare / train / export / verify / serve |
| V1 单次训练/评估 | [verl_v1_launch.py](../spa_next/training/verl_v1_launch.py) | 推荐首先阅读的策略入口 |
| V0 对照 | [launch.py](../spa_next/training/launch.py) | 保留原 outcome GRPO 入口 |
| WebShop | [service.py](../spa_next/envs/service.py) | 环境服务；使用环境专用依赖 |
| 大规模历史运行 | [run_c_full.py](../spa_next/training/run_c_full.py) | 依赖历史目录、划分和恢复产物 |
| 最终历史测试 | [run_final_test.py](../spa_next/training/run_final_test.py) | 依赖历史模型与运行记录 |

在对应环境中查看接口：

```bash
python -m spa_next.training.sft --help
python -m spa_next.training.sft_snapshot --help
python -m spa_next.prm --help
python -m spa_next.training.verl_v1_launch --help
```

## V1 配置预览示例

以下命令供**已安装固定训练依赖**、已准备相应路径的环境展开配置；dry-run 不验证完整数据/模型，也不启动训练。

```bash
python -m spa_next.training.verl_v1_launch \
  --experiment C \
  --model /path/to/frozen-sft-base \
  --train /path/to/train.parquet \
  --val /path/to/val.parquet \
  --output /path/to/new-run/checkpoints \
  --policy-gpus 3 \
  --prm-url http://127.0.0.1:8775 \
  --prm-version YOUR_QUALIFIED_PRM_VERSION \
  --dry-run
```

正式运行前还必须启动 WebShop 与通过校验的 PRM 服务，完成 SFT snapshot 校验以及固定 verl 源码补丁接入。C 的 PRM 版本不能随便填写，必须匹配服务健康检查。GPU PRM 使用 `serve --backend original-gpu`；默认 INT8 后端不是最终实验配置。

补丁入口包括 [patch_verl_v1.py](../spa_next/training/patch_verl_v1.py)、[patch_verl_three_gpu.py](../spa_next/training/patch_verl_three_gpu.py) 及相关显存补丁。它们依赖源码锚点和已有补丁状态；先检查源文件、备份和各入口参数，不能把目录内所有 patch 脚本盲目依次执行。GPU 原精度后端还依赖其 manifest 和资格报告，单有合并权重不足以启动。

## 已完成实验的主要设置

| 项目 | 实际设置 |
|---|---|
| 基座 | Qwen3-4B-Instruct-2507 → SFT776 合并基座 |
| 策略 LoRA | rank32，alpha64，7类线性投影；约6606万可训练参数 |
| 策略 optimizer | AdamW，lr=1e-5，weight_decay=0.01，constant，无 warmup |
| rollout | 每批4任务，每任务4条轨迹；每批1个PPO epoch |
| 更新 | microbatch per GPU=1，梯度裁剪1 |
| 算法 | gamma=1，lambda=0.2，clip=0.2，KL=0.001 |
| 生成 | temperature=1，top-p=1；测试贪心 |
| 预算 | prompt1024，后续3072；每动作256 token，最多10步 |
| SFT | 四卡DDP，776条，1 epoch，lr=1e-4，全局batch16 |
| PRM | 最终三卡DDP，32475条训练/3585条验证，3 epochs，全局batch16 |
| PRM optimizer | AdamW；LoRA lr=1e-5，head lr=1e-4，weight_decay=0，cosine，3% warmup |
| 策略资源 | 三卡FSDP＋独立一卡冻结PRM；4×RTX4080，16GB/卡 |

16条轨迹补成18行供三卡分发；补位行权重和mask为零，不参与分母。同任务优势在分卡前计算；microbatch损失使用共享全局分母，保持梯度累积与分布式平均的一致性。

使用 BF16、梯度检查点、fused torch 概率计算与优化器状态 CPU offload。CPU INT8 未通过资格检查，最终采用 GPU 原精度评分。历史运行曾因系统RAM压力中断并恢复，不能只规划显存。

## 检查与验证

```bash
python -m unittest discover -s spa_next/tests -v
```

定向测试不等于新机器上的完整训练复现。需要在实际 GPU、环境服务、数据及已校验 PRM 齐备后，另做端到端 smoke 验证。本文整理不重新启动训练。
