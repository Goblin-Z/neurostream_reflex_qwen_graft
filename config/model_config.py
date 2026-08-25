from dataclasses import dataclass, field
from typing import Tuple


class ModelConfig:
    """Alias for SEQ compatibility during checkpoint loading."""
    pass


@dataclass
class ReflexConfig:
    # ── Architecture ──
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4  # GQA: fewer KV heads (n_heads // n_kv_heads = repeat factor)
    d_ff: int = 2048
    n_stable: int = 5
    n_plastic: int = 3
    top_k: int = 2
    vocab_size: int = 151936
    max_seq_len: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1  # SDPA dropout during training (0 to always use Flash)

    # ── Backbone（嫁接扩展）──
    # 'reflex' = 原生 MoE 主干；'qwen3_dense' = Qwen3.x 稠密主干（Qwen3.8-27B 嫁接）
    backbone: str = 'reflex'

    # ── RoPE ──
    rope_theta: float = 10000.0
    # partial rotary（Qwen3.x 风格）：只旋转前 rotary_dim = head_dim * factor 维
    partial_rotary_factor: float = 1.0

    # ── Attention 扩展（Qwen3.x 兼容）──
    # 显式 head_dim（Qwen3.x: head_dim 可不等于 d_model // n_heads，如 24x256 != 5120）；
    # 0 = 自动（d_model // n_heads）
    head_dim: int = 0
    # Qwen3.x attn_output_gate: q_proj 输出双倍宽度，一半作 query、一半作 sigmoid 门控
    attn_gate: bool = False

    # ── Attention Residuals (Kimi AttnRes) ──
    attnres_enabled: bool = True
    attnres_block_size: int = 2  # layers per block (n_layers=4 -> 2 blocks -> 1 boundary)
    attnres_rank: int = 128     # low-rank key dimension for depth-wise attention
    # post_norm 初始尺度：原 1e-3 使 AttnRes（含记忆 source）几乎无声；
    # 放大到 0.1 让跨块 delta 与记忆检索真实影响输出（记忆微调前提）
    attnres_postnorm_init: float = 0.1

    # ── Weight tying ──
    tie_word_embeddings: bool = True

    # ── Expert LR spectrum ──
    # idx 0-4 = stable, 5-7 = plastic
    expert_baseline_lrs: Tuple[float, ...] = (
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7,  # stable x5
        1e-5, 1e-4, 1e-4,               # plastic x3
    )

    # ── Internal loop ──
    cont_lr: float = 1e-4
    ema_alpha: float = 0.8
    lyapunov_lambda: float = 0.01
    internal_noise_scale: float = 0.01
    internal_noise_growth: float = 2.0
    internal_entropy_threshold: float = 0.5

    noise_annealing_steps: int = 50000
    noise_min: float = 0.0001
    noise_max: float = 0.05

    # ── Verification ──
    verify_threshold: float = 0.5
    verify_min_interval: int = 10
    verify_query_max_tokens: int = 30
    max_cognitive_depth: int = 5
    # 部署初期保守冷却（审计 v2 P1-5）：前 verify_warmup_steps 步内，
    # 距上次提问不足 verify_warmup_cooldown 步则不提问，防止 sigma 未校准时的随机提问
    verify_warmup_steps: int = 500
    verify_warmup_cooldown: int = 50

    # ── SelfModel (v2 Contemplator upgrade) ──
    self_model_enabled: bool = True
    self_model_z_dim: int = 64
    self_model_hidden_dim: int = 512
    self_model_n_prior_experts: int = 3
    self_model_n_post_experts: int = 3

    # ── Curiosity drive (v2) ──
    curiosity_beta: float = 0.1
    imagination_lambda: float = 1.0
    stability_lambda: float = 0.01
    # P1-4: 内循环梯度接地——想象输出与最近对话 embedding 的 MSE 权重
    # （让意识流扎根于真实对话，而非纯自指）
    grounded_weight: float = 0.1

    # ── Consolidation ──
    sleep_lr: float = 1e-5
    distill_temperature: float = 2.0
    ewc_lambda: float = 40.0
    sleep_batch_size: int = 128
    sleep_keep_ratio: float = 0.5
    plastic_soft_reset_keep: float = 0.5
    plastic_reg_strength: float = 1e-6

    # ── Architecture self-modification ──
    # 默认关闭（无对照实验前不改写已训练权重）；部署可用 --arch-self-mod 显式开启
    arch_self_mod_enabled: bool = False
    arch_replace_reward_threshold: float = 0.2
    arch_split_load_ratio: float = 3.0
    arch_add_layer_improvement_threshold: float = 0.01

    # ── Critic ──
    critic_enabled: bool = True
    critic_hidden_dim: int = 256
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    gamma: float = 0.99
    expert_lr_bias_range: float = 2.0
    query_lr: float = 1e-4
    verify_threshold_init: float = 0.5

    # ── Feedback (Reflex key innovation) ──
    feedback_lr: float = 1e-5
    feedback_max_grad_norm: float = 1.0
    feedback_gamma_modulation_strength: float = 0.3
    feedback_alignment_weight: float = 0.5  # deprecated (unused in feedback.py)
    feedback_align_loss_weight: float = 0.2  # alignment loss weight (CE complement)
    sigma_calibration_weight: float = 0.2  # sigma calibration loss weight (0.05→0.2，强化校准)
    feedback_alignment_threshold: float = 0.2  # 反馈对齐最低余弦相似度（原硬编码 0.3 过高，P2-4）

    # ── Sampling (generation) ──
    sampling_temperature: float = 0.8
    sampling_top_k: int = 40
    sampling_top_p: float = 0.9
    sampling_repetition_penalty: float = 1.5

    # ── Gradient management (Reflex key innovation) ──
    gradient_per_layer: bool = True

    # ── Others ──
    internal_steps_per_cycle: int = 1000
    # 内循环每步后的休眠毫秒数（防忙循环烧算力；0=不限制）。
    # 默认 5ms ≈ 200 steps/s 上限，保持"持续思考"同时释放 CPU/GPU
    internal_step_delay_ms: float = 5.0
    internal_loss_clip: float = 10.0
    endosphere_capacity: int = 1024
    replay_capacity: int = 10000
    supervised_enabled: bool = False
    supervised_batch_size: int = 4
    supervised_lr: float = 2e-5
    max_new_tokens: int = 256

    # ── Memory system v4 (L1-L4) ──
    dialog_memory_alpha: float = 0.3       # L1: 对话→h_t 混合率
    memory_enabled: bool = True            # 总开关
    memory_bank_capacity: int = 128        # L2/L3: 语义槽数量
    memory_context_top_k: int = 8          # L2: AttnRes 记忆候选数
    memory_write_lr: float = 0.05          # L3: 写入门强度
    kv_cache_rounds: int = 4               # L4: KV 内容记忆保留轮数
    memory_distill_enabled: bool = True    # 记忆→权重压缩固化开关
    memory_distill_batch: int = 8          # 每次蒸馏采样语义槽数
    # ── 自发固化（salience 驱动，非程序性）──
    memory_salience_enabled: bool = True   # salience 累积与即时固化开关
    memory_salience_threshold: float = 3.0  # 成熟阈值（≈高注意力聚焦 10 次）
    memory_salience_decay: float = 0.99    # 新鲜度衰减
    memory_consolidate_cooldown: int = 50  # 单条固化后冷却步数
    memory_consolidate_batch: int = 4      # 每步最多固化的记忆条数
    memory_sigma_strength: float = 0.2     # sigma 调制固化强度系数


@dataclass
class ReflexMediumConfig(ReflexConfig):
    """
    Scaled-up config for H800/H20 training.

    ~1.45B parameters, 16 layers, 8 experts/layer, d_model=1024.
    Modern architecture: GQA + RoPE + SwiGLU + RMSNorm + AttnRes.

    Depth-width balance: 16 layers (33% deeper than v1's 12) x 8 experts
    (33% narrower) addresses the "too shallow too wide" MoE problem.
    plastic(3) > top_k(2) ensures dialectical diversity -- not all
    plastic experts are activated simultaneously, allowing distinct
    thesis/antithesis modes to emerge.
    d_ff=2688 (21x128) is GPU tensor-core aligned.
    """
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4   # GQA 4:1 ratio
    d_ff: int = 2688      # 21x128, GPU tensor-core aligned
    n_stable: int = 5
    n_plastic: int = 3
    top_k: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.1

    # RoPE
    rope_theta: float = 10000.0

    # AttnRes: 16 layers / 4 per block = 4 blocks -> 3 boundaries
    attnres_enabled: bool = True
    attnres_block_size: int = 4
    attnres_rank: int = 256  # d_model/4

    # Weight tying
    tie_word_embeddings: bool = True

    expert_baseline_lrs: Tuple[float, ...] = (
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7,  # stable x5
        1e-5, 1e-4, 1e-4,               # plastic x3
    )

    self_model_z_dim: int = 128
    self_model_hidden_dim: int = 1024
    critic_hidden_dim: int = 512
    max_new_tokens: int = 192


@dataclass
class ReflexMiniConfig(ReflexConfig):
    """
    Mini config for 32GB GPU training.

    ~0.77B parameters, 24 layers, 6 experts/layer, d_model=640.
    Same core architecture: GQA + RoPE + SwiGLU + RMSNorm + AttnRes +
    SelfModel + DialecticalBuffer + Hebbian + EWC.

    Design choices:
    - 24 layers (same depth as Qwen2.5-0.5B) for hierarchical reasoning
    - 6 experts (4S+2P) with top_k=2: 27.5% activation (212M active/token)
    - d_model=640 (5x128, GPU aligned) -- moderate backbone
    - d_ff=2048 (16x128, 3.2x SwiGLU expansion, GPU tensor-core aligned)
    - Per expert: 4.4M params -- not too wide (3.2x), not too shallow
    - AttnRes block_size=6: 24/6=4 blocks -> 3 boundaries
    - All-expert mode (top_k=6): 636M active (82.6%)
    - 15B pretrain tokens -> 19.5x params = 97% Chinchilla (near optimal)
    """
    d_model: int = 640
    n_layers: int = 24
    n_heads: int = 10
    n_kv_heads: int = 2   # GQA 5:1 ratio
    d_ff: int = 2048      # 16x128, GPU tensor-core aligned, 3.2x expansion
    n_stable: int = 4
    n_plastic: int = 2
    top_k: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.1

    # RoPE
    rope_theta: float = 10000.0

    # AttnRes: 24 layers / 6 per block = 4 blocks -> 3 boundaries
    attnres_enabled: bool = True
    attnres_block_size: int = 6
    attnres_rank: int = 160  # d_model/4

    # Weight tying
    tie_word_embeddings: bool = True

    expert_baseline_lrs: Tuple[float, ...] = (
        1e-7, 1e-7, 1e-7, 1e-7,  # stable x4
        1e-5, 1e-4,               # plastic x2
    )

    self_model_z_dim: int = 128
    self_model_hidden_dim: int = 640
    critic_hidden_dim: int = 320
    max_new_tokens: int = 192


@dataclass
class Qwen3GraftConfig(ReflexConfig):
    """
    Qwen3.8-27B 嫁接配置 —— 将 Qwen3.8-27B 原始权重作为 Reflex 的 LLM 主干。

    专家数量对齐：Qwen3.8-27B 是稠密（DENSE）模型，每层恰好 1 个 SwiGLU FFN
    → n_stable=1, n_plastic=0, top_k=1，MoE 退化为单专家等价（router 恒选专家 0，
    权重恒为 1.0）。这是"直接用原权重 + 最小改动"的必然形态；N 专家克隆升级见设计文档。

    架构（Qwen3.5/3.6/3.8 共享实现，2026-08-14 发布，Apache-2.0）：
      - 64 层混合注意力：48 层 Gated DeltaNet 线性注意力 + 16 层全注意力（3:1 交替）
      - d_model=5120, 全注意力层 24 头 x head_dim=256（≠ d_model//n_heads），GQA 4 KV 头
      - SwiGLU FFN intermediate=17408（与项目 Expert 的 w_gate/w_up/w_down 同构）
      - 词表 248320（含视觉 token），不绑定权重（embed 与 lm_head 独立）
      - RoPE: theta=1e7, partial_rotary_factor=0.25（仅旋转 64/256 维）
      - attn_output_gate: q_proj 双倍输出，一半 query 一半 sigmoid 门控
      - QK-Norm（RMSNorm per head，1+w 参数化）；RMSNorm eps=1e-6
      - 原生 262144 上下文（本项目部署上限 max_seq_len 另行设定）
    权重加载见 scripts/load_qwen3_graft.py（RMSNorm 1+w 变换、kv_proj 拼接、视觉塔/MTP 跳过）。
    """
    backbone: str = 'qwen3_dense'

    # ── 主动求证阈值（用户要求固定 0.5：sigma 校准已生效，sigma 能真实
    # 反映不确定度，0.5 为标准触发线；--verify-threshold 可临时覆盖）──
    verify_threshold: float = 0.5
    # 会话提问上限（用户要求不限制：0 = 无限；--max-asks 仍可临时设限）
    max_questions_per_session: int = 0

    # ── 主干几何（Qwen3.8-27B text_config 实测）──
    d_model: int = 5120
    n_layers: int = 64
    n_heads: int = 24
    n_kv_heads: int = 4
    head_dim: int = 256
    d_ff: int = 17408
    vocab_size: int = 248320
    tie_word_embeddings: bool = False
    max_seq_len: int = 8192        # 部署截断上限（原生 262144，按显存调）
    dropout: float = 0.0           # Qwen3.x 无 hidden dropout，保持权重行为一致
    attention_dropout: float = 0.0

    # ── RoPE（partial rotary）──
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25

    # ── attn_output_gate ──
    attn_gate: bool = True

    # ── 层类型（3:1 混合；加载器会从 config.json 覆盖）──
    layer_types: Tuple[str, ...] = ()

    # ── Gated DeltaNet 线性注意力 ──
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # ── 专家数量与 Qwen3.8-27B 对齐：1 专家/层（= 稠密 FFN）──
    n_stable: int = 1
    n_plastic: int = 0
    top_k: int = 1
    # 学习强度（用户决定"默认关闭"）：0.0 = Hebbian 关闭（机制保留、无更新）。
    # 嫁接版 Hebbian 的价值论证不充分（无监督梯度 + 不定位困惑 + 无专家分化
    # 哲学对象 + 风险真实），默认关闭最安全；对照实验用 --hebbian-lr 3e-6 开启
    expert_baseline_lrs: Tuple[float, ...] = (0.0,)

    # ── AttnRes（Reflex 附加件，随机初始化；post_norm 1e-3 起步近零）──
    attnres_enabled: bool = True
    attnres_block_size: int = 8    # 64/8 = 8 块 → 7 个边界
    attnres_rank: int = 1280       # d_model/4
    attnres_postnorm_init: float = 0.001

    # ── Reflex 附加件尺寸（d_model=5120 放大）──
    self_model_z_dim: int = 128
    self_model_hidden_dim: int = 1024
    critic_hidden_dim: int = 512
    endosphere_capacity: int = 1024
    max_new_tokens: int = 9000     # 对话回答生成上限（用户要求；仅为上限，
                                   # 模型输出终止符即自动停止，不会强制跑满）
    memory_bank_capacity: int = 128
    memory_context_top_k: int = 8
    # 记忆写入更稳（用户要求）：0.05 → 0.01（L3 语义槽写入强度降 5 倍，
    # 缓解 global_drift 因状态向量大范数写入而快速累积）
    memory_write_lr: float = 0.01

    # ── 嫁接运行模式 ──
    graft_use_past: bool = True            # generate 增量解码（past KV / 线性状态）
    graft_lite: bool = True                # 轻量在线学习（冻结主干 + Hebbian 尾层）
    graft_hebbian_layers: int = 12         # Hebbian 梯度覆盖最后 12 层（用户决定
                                           # "先降强度"：20→12，学习面缩小 40%）
    graft_freeze_backbone: bool = True     # 全局优化器排除主干权重（27B 无法装 AdamW 状态）
    graft_disable_consolidation: bool = True  # 关 mini/major 巩固（27B 蒸馏代价高）
    graft_decode_attnres: bool = False     # 增量解码跳过 AttnRes（块级 delta 语义在解码模式失真）
    graft_online_ce: bool = True           # 每轮对话的全量 CE 在线训练（27B 上最重路径，
                                           # 首轮试验建议 --no-online-ce 关闭，只留 Hebbian）
    graft_verify_max_tokens: int = 512     # 主动求证问题的生成上限（用户要求翻倍：
                                           # 256→512，给 base 模型更多思考/提问空间；
                                           # 512 ≈ 2-5 分钟，仍远低于 9000 的卡死风险）
    graft_gen_debug: bool = False          # 生成停止原因诊断（run_mini --gen-debug 开启）
    graft_think_eos_grace: int = 2         # think 未闭合时终止符宽容次数（默认 2）：
                                           # base 模型会在思考起步时误输出 eos 类 token
                                           # （如 Qwen3.8 的 248046），导致空回复；
                                           # 宽容 N 次后仍输出则尊重模型停止；
                                           # think 闭合后的终止符始终立即停止。
                                           # 设 0 = 完全立即停（旧行为）
    graft_sigma_cal: bool = False          # sigma 在线校准（--sigma-cal 开启）：每
                                           # graft_sigma_cal_interval 步用 tanh(loss_int)
                                           # 校准尾层 uncertainty_head（修复审计 P0-1：
                                           # sigma 头随机初始化且无训练路径 → 无法触发
                                           # 主动求证；注意校准会更新不确定性头参数）
    graft_sigma_cal_interval: int = 20


def config_from_checkpoint(ck: dict):
    """从 checkpoint 的 config 字段重建配置对象（嫁接/原生通用）。

    run_mini.py / chat_sft.py 用它替代硬编码 ReflexMiniConfig()。
    """
    cfg_dict = ck.get('config') if isinstance(ck, dict) else None
    if not cfg_dict:
        return ReflexMiniConfig()
    if cfg_dict.get('backbone') == 'qwen3_dense':
        fields = Qwen3GraftConfig.__dataclass_fields__
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return Qwen3GraftConfig(**kwargs)
    fields = ReflexMiniConfig.__dataclass_fields__
    kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
    return ReflexMiniConfig(**kwargs)
