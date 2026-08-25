"""
core/gated_deltanet.py — Qwen3.5/3.6/3.8 线性注意力层（Gated DeltaNet）移植。

来源：transformers `modeling_qwen3_5.py` 的 Qwen3_5GatedDeltaNet 纯 PyTorch 参考实现
（Apache-2.0）。权重名与 Qwen 官方 safetensors 完全一致，可直接加载原始权重：
  in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / conv1d / dt_bias / A_log /
  norm (RMSNormGated) / out_proj

两条计算路径（与 HF 行为一致）：
  - chunked（torch_chunk_gated_delta_rule，chunk_size=64）: 一般训练/前向
  - recurrent（torch_recurrent_gated_delta_rule）: 单 token 解码（seq_len==1），
    配合增量 past（conv 状态 + recurrent 状态），是嫁接后 generate 提速的关键。

注意：本项目移植版仅做文本前向（无 cache_params 对象），past 以 dict 传递：
  past = {'conv': [B, conv_dim, kernel-1], 'recurrent': [B, v_heads, k_dim, v_dim]}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """Qwen3_5RMSNormGated: out = RMSNorm(x) * SiLU(gate)。权重为 1+w 参数化。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def causal_conv1d_fn(hidden_states, weight, activation='silu'):
    """因果卷积（前向路径）。weight: [conv_dim, 1, kernel]。无 bias（Qwen 配置 bias=False）。"""
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=None,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def causal_conv1d_update(hidden_states, conv_state, weight, activation='silu'):
    """因果卷积（单 token 解码路径）：用缓存状态 + 新 token 计算，并更新状态。"""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), None, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def l2norm(x, dim=-1, eps=1e-6):
    """与 FLA 库一致的 L2 归一化（Qwen 参考实现原样移植）。"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                                 initial_state=None, output_final_state=False,
                                 use_qk_l2norm_in_kernel=False):
    """
    Gated Delta Rule chunked 实现（transformers `torch_chunk_gated_delta_rule` 原样移植）。

    query/key/value: [B, T, H, D]；g/beta: [B, T, H]（g 为负对数衰减，beta 为写入门）。
    返回 (core_attn_out [B, T, H, D_v], last_recurrent_state [B, H, D_k, D_v])。
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state,
                                     output_final_state=True,
                                     use_qk_l2norm_in_kernel=False):
    """
    Gated Delta Rule recurrent 实现（单步扫描；单 token 解码 / 数值验证用）。
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3GatedDeltaNet(nn.Module):
    """
    Qwen3.8-27B 线性注意力层（Gated DeltaNet）——权重结构与 Qwen 官方实现一致。

    past: dict {'conv': [B, conv_dim, kernel-1], 'recurrent': [B, v_heads, k_dim, v_dim]}
          或 None（全量前向）。返回 (output, new_past)。
    """

    def __init__(self, d_model: int, num_k_heads: int = 16, num_v_heads: int = 48,
                 head_k_dim: int = 128, head_v_dim: int = 128,
                 conv_kernel_size: int = 4, rms_eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size

        # QKV（含卷积）—— 名字与 Qwen 权重一一对应
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(self.head_v_dim, eps=rms_eps)
        self.out_proj = nn.Linear(self.value_dim, self.d_model, bias=False)
        self.in_proj_qkv = nn.Linear(self.d_model, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.d_model, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.d_model, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.d_model, self.num_v_heads, bias=False)

    def forward(self, x, attention_mask=None, past=None):
        """
        x: [B, T, d_model]
        attention_mask: [B, T] (1=valid, 0=padding) —— 线性注意力对 padding 置零
        past: 增量解码状态（见类注释）
        返回: (output [B, T, d_model], new_past)
        """
        if attention_mask is not None:
            x = x * attention_mask[:, :, None]

        batch_size, seq_len, _ = x.shape
        decode = past is not None and seq_len == 1

        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)  # [B, conv_dim, T]
        z = self.in_proj_z(x).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if decode:
            conv_state = past['conv']
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), 'silu')
        else:
            # 保存卷积状态（conv 前的最后 kernel-1 个位置），供后续增量解码续接；
            # T < kernel-1 时左侧补零（等价于因果卷积的零填充前缀）
            k1 = self.conv_kernel_size - 1
            if seq_len >= k1:
                conv_state = mixed_qkv[:, :, -k1:].detach().clone()
            else:
                pad = torch.zeros(
                    batch_size, self.conv_dim, k1 - seq_len,
                    dtype=mixed_qkv.dtype, device=mixed_qkv.device)
                conv_state = torch.cat([pad, mixed_qkv], dim=-1).detach().clone()
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv, self.conv1d.weight.squeeze(1), 'silu')

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # fp16 下不加 .float() 会使 A 变为 -inf（Qwen 原注释）
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if decode:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=past['recurrent'],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None,
                output_final_state=True,   # prefill 也要输出末态，供解码续接
                use_qk_l2norm_in_kernel=True,
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)

        # 始终输出续接状态（训练时调用方忽略；dtype 统一为输入 dtype）
        if last_recurrent_state is not None:
            new_past = {
                'conv': conv_state.detach(),
                'recurrent': last_recurrent_state.detach().to(x.dtype),
            }
        else:
            new_past = None
        return output, new_past
