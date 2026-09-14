"""
================================================================================
model.py —— NovaMind 模型骨架（项目第 3 个脚本，最核心）
================================================================================

【这个文件是什么】
    一个 Decoder-Only 的 Transformer 语言模型（和 GPT、Llama、Qwen 同一族）。
    从「词 id」进，预测「下一个词的概率」出。

【整体数据流（一句话看懂）】
    input_ids [B, T]                                # 一串 token id
      → embed_tokens 查表得到向量 [B, T, D]
      → 叠 N 个 NovaMindBlock（每块做 注意力 + 前馈）
      → 最后 RMSNorm
      → lm_head 线性层映射回词表 [B, T, vocab]
      → 和真实下一个 token 算交叉熵 loss

【本文件按「从底层到顶层」分 8 块，建议按顺序读】
    1. RMSNorm                —— 归一化
    2. RoPE                   —— 旋转位置编码（让模型知道 token 的位置）
    3. Attention              —— 注意力（GQA + KV Cache + Flash）
    4. FeedForward            —— 前馈网络（SwiGLU）
    5. MoEGate / MOEFeedForward —— 混合专家（可选，开 use_moe 才用）
    6. NovaMindBlock          —— 把 3+4 拼成一个 Transformer 块
    7. NovaMindModel          —— 把 6 叠 N 层
    8. NovaMindForCausalLM    —— 顶层：加输出层 + 算 loss

【和 minimind 的关系】
    结构同 minimind 的 model_minimind.py，但：词表 32000、宽度 1024、24 层、
    用 fp16（V100 无 bf16）、命名改为 NovaMind、不依赖 transformers 的抽象。
================================================================================
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


# ============================================================================ #
#  8 块的公共输出结构
# ============================================================================ #
@dataclass
class ModelOutput:
    """forward 的返回值，训练脚本里用 res.loss / res.logits 取。

    例子：
        res = model(input_ids, labels=labels)
        print(res.loss)      # 语言模型 loss
        print(res.aux_loss)  # MoE 负载均衡辅助 loss（非 MoE 时为 0）
        print(res.logits.shape)  # [B, T, vocab_size]
    """
    logits: torch.Tensor = None
    loss: torch.Tensor = None
    aux_loss: torch.Tensor = None
    hidden_states: torch.Tensor = None
    past_key_values: List = None


# ============================================================================ #
#  1. RMSNorm —— 归一化层
# ============================================================================ #
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization，Llama/Qwen 用的归一化（比 LayerNorm 简单、快）。

    作用：把每个 token 的向量「缩放」到单位尺度，让深层训练更稳定。

    公式：  y = x / sqrt(mean(x^2) + eps) * weight

    例子：
        x = [2.0, 2.0]  (D=2)
        mean(x^2) = (4+4)/2 = 4, sqrt = 2
        y = x / 2 * weight = [1.0, 1.0] * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放系数，初始全 1（归一化后每个维度再各自缩放）
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x.pow(2).mean(-1, keepdim=True) 是对最后一维（D）求均方；rsqrt 是 1/sqrt
        # 归一化计算用 fp32（避免 fp16 精度太低），最后转回输入 dtype
        # 关键：.to(x.dtype) 必须在最外层（乘完 weight 之后），
        #       否则 fp16 autocast 下 weight(fp32)*x(fp16) 会被 promote 成 fp32，导致后续 dtype 混乱
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(x.dtype)


# ============================================================================ #
#  2. RoPE —— 旋转位置编码
# ============================================================================ #
def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6):
    """
    预计算 RoPE 需要的 cos/sin 表（训练前算一次，存起来反复用）。

    RoPE 思路：给每个「位置」一个旋转角度，把 query/key 向量在二维平面旋转，
    这样注意力就能感知 token 的相对位置（第 1 个词在第 2 个词前面）。

    参数：
        dim       : 每个头的维度（head_dim），如 64
        end       : 最大位置数（max_position_embeddings），如 2048
        rope_base : 频率基数，1e6 是 Llama/Qwen 的取值

    返回：(cos, sin) 两个表，形状都是 [end, dim]

    例子（dim=4 的简化版）：
        freqs = [1/base^(0/4), 1/base^(2/4)]  # 只算 dim/2=2 个频率
        位置 t 的角度 = t * freqs
        最后把 cos/sin 各复制一份拼成 [end, 4]
    """
    # 只算 dim/2 个频率（RoPE 的「旋转」是把向量分成前后两半，各自旋转）
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)          # 位置 0..end-1
    freqs = torch.outer(t, freqs)                        # [end, dim/2] 每个位置×每个频率 = 角度
    # cos 复制一份拼成 [end, dim]：前半和后半用同样的角度（只是 rotate_half 时符号不同）
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin


def rotate_half(x):
    """把向量后半段取负后挪到前面 —— RoPE 的核心旋转操作。

    例子：x = [a, b, c, d]  →  rotate_half(x) = [-c, -d, a, b]
    """
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    把 RoPE 旋转作用到 query 和 key 上。

    公式：q' = q*cos + rotate_half(q)*sin
         k' = k*cos + rotate_half(k)*sin

    参数：q, k 形状 [B, T, H, D]；cos, sin 形状 [T, D]
    返回：旋转后的 q, k，形状不变

    例子（单个 token 的 q=[a,b,c,d]，角度 θ）：
        q*cos      = [a·cosθ, b·cosθ, c·cosθ, d·cosθ]
        rotate_half(q)*sin = [-c·sinθ, -d·sinθ, a·sinθ, b·sinθ]
        相加即完成「在 (a,c) 平面和 (b,d) 平面各旋转 θ 度」
    """
    cos = cos.to(q.dtype)   # 保证 cos/sin 和 q 同 dtype（fp16 训练时重要）
    sin = sin.to(q.dtype)
    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA（分组查询注意力）的 KV 扩展：让少量 KV 头「复制」成和 Q 头一样多。

    例子：x = [B, T, 4, D]（4 个 KV 头），n_rep=4
         → [B, T, 16, D]（16 个 Q 头，每 4 个 Q 头共享 1 组 KV）
    """
    b, t, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]                                            # [B,T,kv,1,D]
        .expand(b, t, num_kv_heads, n_rep, head_dim)                   # [B,T,kv,n_rep,D]
        .reshape(b, t, num_kv_heads * n_rep, head_dim)                 # [B,T,kv*n_rep,D]
    )


# ============================================================================ #
#  3. Attention —— 注意力层（GQA + KV Cache + Flash）
# ============================================================================ #
class Attention(nn.Module):
    """
    多头注意力。输入一个序列，让每个 token「看到」它前面的 token，算加权平均。

    核心公式：Attention(Q,K,V) = softmax(Q·K^T / sqrt(d)) · V

    【三个重要设计】
      - GQA：KV 头比 Q 头少（省显存、提速，效果几乎不掉）
      - KV Cache：生成时缓存历史 K/V，避免每步重算（加速自回归）
      - Flash：用 F.scaled_dot_product_attention（内存友好，V100 可用）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads          # Q 头数，如 16
        self.n_kv_heads = config.num_key_value_heads        # KV 头数，如 4
        self.n_rep = self.n_heads // self.n_kv_heads        # 每个 KV 头被几个 Q 头共享，如 4
        self.head_dim = config.hidden_size // config.num_attention_heads  # 64

        # 四个线性层。注意 K/V 的输出维度 = n_kv_heads * head_dim（比 Q 小，这就是 GQA 省参数的地方）
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(self, x, cos, sin, past_key_value=None, use_cache=False, attention_mask=None):
        b, seq_len, _ = x.shape

        # 1) 投影并拆成多头：[B, T, heads, head_dim]
        xq = self.q_proj(x).view(b, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(b, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(b, seq_len, self.n_kv_heads, self.head_dim)

        # 2) 加 RoPE 位置编码（只对 Q/K 加，V 不加）
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 3) KV Cache：把历史 K/V 拼在当前前面（生成时复用，避免重算）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 4) 转成 [B, heads, T, head_dim]，并把 KV 头复制到和 Q 头一样多（GQA）
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 5) 算注意力。两条路径：
        #    - Flash 路径（训练、无 cache、无 padding mask 时）：直接调用官方融合算子
        #    - 手写路径（生成时带 cache，或需要 padding mask 时）：手动 softmax
        if (past_key_value is None) and (attention_mask is None):
            # 训练时的标准情形，走 Flash Attention（内存友好）
            output = F.scaled_dot_product_attention(
                xq, xk, xv, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # 手动实现：scores = Q·K^T / sqrt(d)，再 softmax，再乘 V
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码：只允许看到当前位置及之前（上三角填 -inf）
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1
            )[-seq_len:, -seq_len:]
            # padding 掩码：被 padding 的位置也屏蔽
            if attention_mask is not None:
                mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).to(xq.dtype)
            output = scores @ xv

        # 6) 拼回头，过输出层
        output = output.transpose(1, 2).reshape(b, seq_len, -1)
        return self.o_proj(output), past_kv


# ============================================================================ #
#  4. FeedForward —— 前馈网络（SwiGLU）
# ============================================================================ #
class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络，Transformer 里除了注意力外的另一半计算。

    公式：FFN(x) = down( silu(gate(x)) * up(x) )
    即：一路做「门控」，一路做「内容」，两者相乘后降维。

    例子：x 是 1024 维 → gate/up 升到 2816 维 → 相乘 → down 降回 1024 维
    """

    def __init__(self, config: ModelConfig, intermediate_size=None):
        super().__init__()
        hidden = config.hidden_size
        # intermediate_size 可被覆盖（MoE 专家用它来用更小的中间维度）
        intermediate = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # silu = x * sigmoid(x)，就是 SwiGLU 里的激活函数
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ============================================================================ #
#  5. MoE —— 混合专家（可选，use_moe=True 才启用）
# ============================================================================ #
class MoEGate(nn.Module):
    """
    门控路由：决定每个 token 交给哪几个「专家」处理。

    DeepSeek 的思路：把 FFN 变成多个专家，每个 token 只激活 top-k 个，
    这样参数量变大但计算量不变，还能「分工」——不同专家擅长不同内容。

    额外输出 aux_loss（负载均衡损失），让专家别「忙闲不均」。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok          # 每个 token 激活几个专家，如 2
        self.n_experts = config.n_routed_experts         # 专家总数，如 4
        self.alpha = config.aux_loss_alpha               # 辅助损失系数
        # 门控权重：每个专家一个向量，和输入点积得到「这个专家有多匹配」
        self.weight = nn.Parameter(torch.empty(self.n_experts, config.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        b, seq_len, h = hidden_states.shape
        logits = F.linear(hidden_states.view(-1, h), self.weight)   # [B*T, n_experts]
        scores = logits.softmax(dim=-1)                              # 归一化成概率
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)  # 选 top-k 专家

        # 把 top-k 的权重重新归一化（加起来 = 1）
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 负载均衡辅助损失：鼓励每个专家被均匀使用
        if self.training and self.alpha > 0.0:
            # 统计每个专家被选中的次数（one-hot 求和）
            ce = torch.zeros(b, self.n_experts, device=hidden_states.device)
            ce.scatter_add_(1, topk_idx.view(b, -1),
                            torch.ones(b, seq_len * self.top_k, device=hidden_states.device))
            ce = ce / (seq_len * self.top_k / self.n_experts)        # 归一化
            aux_loss = (ce * scores.view(b, seq_len, -1).mean(dim=1)).sum(dim=1).mean() * self.alpha
        else:
            aux_loss = torch.zeros((), device=hidden_states.device)
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """由多个 FeedForward 专家 + 一个门控组成的 MoE 前馈。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 专家用 expert_intermediate_size（若没设就是 None，FeedForward 会用 config.intermediate_size）
        expert_size = config.expert_intermediate_size
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=expert_size) for _ in range(config.n_routed_experts)]
        )
        self.gate = MoEGate(config)
        self.shared_experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=expert_size) for _ in range(config.n_shared_experts)]
        ) if config.n_shared_experts > 0 else None
        self.aux_loss = None

    def forward(self, x):
        identity = x                                # 留给共享专家用
        orig_shape = x.shape
        b, seq_len, _ = x.shape

        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])                 # 展平成 [B*T, D]
        flat_idx = topk_idx.view(-1)                # [B*T*top_k]

        if self.training:
            # 训练：每个 token 复制 top_k 份，交给对应的专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                mask = flat_idx == i
                if mask.any():
                    y[mask] = expert(x[mask]).to(y.dtype)   # 强制 dtype 一致
            # 加权求和：把 top_k 个专家的输出按权重加起来
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理：更省显存的分组计算（只算被选中的专家）
            y = self._moe_infer(x, flat_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 共享专家：所有 token 都过一遍（DeepSeek 的设计）
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                y = y + expert(identity).to(y.dtype)

        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def _moe_infer(self, x, flat_idx, flat_weights):
        """推理时的 MoE：按专家分组，只把 token 喂给它被选中的专家。"""
        cache = torch.zeros_like(x)
        order = flat_idx.argsort()                              # 按专家 id 排序
        counts = flat_idx.bincount(minlength=self.config.n_routed_experts).cumsum(0)
        for i, end in enumerate(counts):
            start = 0 if i == 0 else counts[i - 1]
            if start == end:
                continue
            token_pos = order[start:end] // self.config.num_experts_per_tok
            out = self.experts[i](x[token_pos]) * flat_weights[order[start:end]].unsqueeze(-1)
            cache.scatter_add_(0, token_pos.unsqueeze(1).repeat(1, x.shape[-1]), out)
        return cache


# ============================================================================ #
#  6. NovaMindBlock —— 一个 Transformer 块
# ============================================================================ #
class NovaMindBlock(nn.Module):
    """
    一个 Transformer 块 = 注意力子层 + 前馈子层，都是「pre-norm + 残差」。

    数据流：
        x → RMSNorm → Attention → +x（残差）→ RMSNorm → FFN → +x（残差）

    pre-norm 的意思是「先归一化再计算」，比 post-norm 训练更稳定。
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # use_moe=True 时用 MoE 前馈，否则用普通 FFN
        self.mlp = MOEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, hidden_states, cos, sin, past_key_value=None, use_cache=False, attention_mask=None):
        # 注意力子层 + 残差
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states), cos, sin,
            past_key_value, use_cache, attention_mask,
        )
        hidden_states = residual + hidden_states

        # 前馈子层 + 残差
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_kv


# ============================================================================ #
#  7. NovaMindModel —— 把 Block 叠 N 层
# ============================================================================ #
class NovaMindModel(nn.Module):
    """模型的「主干」：词嵌入 + N 层 Block + 最终归一化。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 词嵌入：把 token id 变成向量。[vocab, hidden]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([NovaMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 表，存为 buffer（不参与训练，但会随模型移动设备）
        cos, sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                past_key_values=None, use_cache=False):
        b, seq_len = (input_ids.shape if input_ids is not None else inputs_embeds.shape[:2])
        past_key_values = past_key_values or [None] * len(self.layers)
        # start_pos：KV cache 已缓存到第几个位置（生成时用）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 支持两种输入：token id 或直接给 embedding（VLM 拼接图像特征时用后者）
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        hidden_states = self.dropout(hidden_states)

        # 取当前位置区间的 RoPE 表
        cos = self.cos[start_pos: start_pos + seq_len]
        sin = self.sin[start_pos: start_pos + seq_len]

        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, cos, sin,
                past_key_value=past_kv, use_cache=use_cache, attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        # 汇总所有 MoE 层的辅助损失
        aux_loss = sum(
            (l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)),
            torch.zeros((), device=hidden_states.device),
        )
        return hidden_states, presents, aux_loss


# ============================================================================ #
#  8. NovaMindForCausalLM —— 顶层模型（加输出层 + 算 loss）
# ============================================================================ #
class NovaMindForCausalLM(nn.Module):
    """
    完整的语言模型：主干 + 输出层（lm_head）。

    训练时传 labels，自动算「预测下一个 token」的交叉熵 loss。
    推理时只传 input_ids，拿到 logits 后采样出下一个 token。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = NovaMindModel(config)
        # lm_head：把最后一层向量映射回词表大小，得到每个词的「分数」
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 权重绑定：输入词嵌入和输出层共用同一份权重（省参数量，效果更好）
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                inputs_embeds=None, past_key_values=None, use_cache=False, logits_to_keep=0):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
        )
        # logits_to_keep>0 时只保留最后几个位置的 logits（生成加速用）
        logits = self.lm_head(hidden_states[:, -logits_to_keep:] if logits_to_keep else hidden_states)

        loss = None
        if labels is not None:
            # 语言模型任务：用「前面的 token」预测「下一个 token」
            # 所以 logits 去掉最后一个，labels 去掉第一个，一一对应
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,      # -100 的位置不算 loss（padding / 非监督部分）
            )

        return ModelOutput(
            logits=logits, loss=loss, aux_loss=aux_loss,
            hidden_states=hidden_states, past_key_values=past_key_values,
        )


# ============================================================================ #
#  运行演示：`python model.py` 会走到这里
# ============================================================================ #
if __name__ == "__main__":
    from .config import PRESETS

    print(__doc__)
    print("\n########## 示例：实例化 0.3B 模型，跑一个前向 + 反向 ##########\n")

    cfg = ModelConfig(**PRESETS["novamind-0.3b"])
    model = NovaMindForCausalLM(cfg)

    # 统计参数量
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total:.2f} M")

    # 构造假输入：batch=2, 长度=16 的随机 token id
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = input_ids.clone()

    # 前向
    res = model(input_ids, labels=labels)
    print(f"loss = {res.loss.item():.4f}")
    print(f"logits 形状 = {tuple(res.logits.shape)}  (应为 [2, 16, 32000])")

    # 反向（验证梯度能正常回传）
    res.loss.backward()
    print("反向传播 OK，梯度已回传")
