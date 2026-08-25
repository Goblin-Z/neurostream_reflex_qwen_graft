# NeuroStream-Reflex Qwen3.8-27B（嫁接版）

**0.77B 中文神经形态模型的 Reflex 双循环架构 × Qwen3.8-27B 原始权重**

本项目将 [NeuroStream-Reflex](https://github.com/) 的双循环架构（外循环对话 + 内循环意识流）嫁接到 **Qwen3.8-27B**（27B Dense，64 层混合注意力：48 层 Gated DeltaNet + 16 层全注意力）上：
权重**直接复用 Qwen3.8-27B 原始权重**（零训练、确定性映射），Reflex 附加件（SelfModel 世界模型 / Critic / 多层记忆 / AttnRes / sigma 主动求证 / Hebbian）叠加其上。

## 📦 模型权重（ModelScope / HuggingFace）

| 平台 | 仓库 | 内容 |
|---|---|---|
| **ModelScope（推荐）** | [Marshauv/Neurostream_reflex_Qwen_graft](https://modelscope.cn/models/Marshauv/Neurostream_reflex_Qwen_graft) | `reflex_qwen3_27b_graft.pt`（54GB bf16）+ 最小可运行包 |
| HuggingFace | *（可选）* | 同权重镜像 |

**本仓库（GitHub）只放代码与文档**——权重请从 ModelScope 下载。

## 核心设计（双循环自指）

```
外循环（对话）: 输入 + 记忆 + 状态 → 生成回答 → 在线学习 → 写回记忆
内循环（意识流）: 状态 → 噪声 → SelfModel 想象 → 前向思考 → Hebbian/Critic
                → 辩证缓冲 → 记忆固化 → sigma 高时主动提问
两个尺度闭合：状态→思考→新状态；对话→学习→记忆→下一次对话
```

- **主动求证**：专家 uncertainty_head 携带 sigma 信号，困惑时主动提问，回答成为学习信号（涌现冷却）
- **多层记忆**：L1 对话注入 / L2-L3 语义槽 / L4 逐词 KV 记忆 + 自发巩固
- **部署即学习**：Hebbian 局部更新 + SelfModel 演化 + 记忆固化（可配置）

## 快速开始

```bash
# ① 下载权重（54GB，需 ModelScope 账号）
pip install modelscope
modelscope download --model Marshauv/Neurostream_reflex_Qwen_graft \
    --local_dir /data/reflex_qwen3_27b

# ② 安装依赖
pip install torch transformers safetensors tqdm

# ③ 运行（tokenizer 需 Qwen3.8-27B 的，词表 248320，约 5MB）
python chat_sft.py --checkpoint /data/reflex_qwen3_27b/reflex_qwen3_27b_graft.pt \
    --tokenizer <Qwen3.8-27B tokenizer 目录> \
    --device cuda --dtype bfloat16 --hide-think --prompt "中国的首都是"

# ④ 完整部署（双循环全开）
python run_mini.py --checkpoint /data/reflex_qwen3_27b/reflex_qwen3_27b_graft.pt \
    --tokenizer <Qwen3.8-27B tokenizer 目录> \
    --device cuda --dtype bfloat16 --no-online-ce --hide-think --sigma-cal
```

> 硬件要求：≥96GB 显存（H20 / RTX PRO 6000），CPU 内存 ≥90GB。
> 交互：`>>> 提问` / `stats` 查看内循环 / `clear` 清空记忆 / `quit` 退出。

## 从零复现嫁接（可选）

完整工具脚本（`scripts/load_qwen3_graft.py` 加载器、`verify_graft.py` 数值验证）不在此精简仓库——它们与权重一起发布在 ModelScope，或参考设计文档的映射表自行实现：

- 下载 Qwen3.8-27B（55GB）→ 确定性权重映射（RMSNorm 1+w、kv 拼接、层类型路由）→ 生成嫁接 checkpoint；
- 数值验证：与 HF 官方实现对比 logits（max|Δ| < 0.1 且 top-1 > 95% 为通过）。

## 文档

完整嫁接设计文档（哲学映射、架构对比、权重映射表、显存预算、AutoDL 手册）
随权重发布在 ModelScope 的完整代码包中。

## 目录结构

```
config/   模型配置（Qwen3GraftConfig）
core/     模型主干（Qwen3GraftLayer / Gated DeltaNet / AttnRes / 记忆）
loop/     内循环（意识流 A-K 阶段）
interaction/ 外循环（对话 / 求证 / 反馈）
learn/    在线学习（Hebbian / Critic / 巩固）
improve/  架构自修改（嫁接模式禁用）
scripts/  加载器 / 验证 / 冒烟测试 / 发布工具
run_mini.py  完整部署入口
chat_sft.py  快速测试入口
```

## License

- **代码**：MIT License（Copyright 2026 Goblin-Z）
- **模型权重**：Apache License 2.0（衍生自 Qwen/Qwen3.8-27B，© Alibaba Group）——从 ModelScope 获取，详见模型卡

## 相关链接

- 模型权重：https://modelscope.cn/models/Marshauv/Neurostream_reflex_Qwen_graft
- 上游模型：https://huggingface.co/Qwen/Qwen3.8-27B
