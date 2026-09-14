"""
================================================================================
config.py —— NovaMind-VL 的模型配置（项目第 1 个脚本）
================================================================================

【为什么先写它】
    后面所有脚本（模型、数据、训练、推理）都要读这份配置。
    它是整个项目的"地基"，决定了模型多大、多少层、多少注意力头、词表多大。
    改模型大小 = 改这一份配置，其它代码都不用动。

【本文件分 3 部分】
    A. ModelConfig  —— 模型结构超参数（核心）
    B. PRESETS      —— 预设规格表（一键切换模型大小）
    C. num_params()  —— 估算参数量，用来验证配置是否合理

【怎么读这个文件】
    先读 A 的每个字段注释，再读 C 的参数估算逻辑（理解"参数量从哪来"），
    最后运行 `python config.py` 看输出示例。
================================================================================
"""


class ModelConfig:
    """
    模型结构超参数。

    一个 Decoder-Only 的 Transformer 语言模型（和 GPT、Llama、Qwen 同一族），
    它的"形状"完全由下面这些数字决定。每个字段后面都写了它是干嘛的。
    """

    # ------------------------------------------------------------------ #
    # 一、最核心的三个数字：层数、宽度、词表
    #     它们几乎决定了 90% 的参数量，也是最常调的地方。
    # ------------------------------------------------------------------ #

    # 词表大小：模型"认识"多少个不同的 token（类似字典的页数）。
    #   - minimind 用 6400（太小，导致中文被拆得很碎、理解能力弱）
    #   - NovaMind 用 32000，中英文压缩率和表达能力都更强
    #   - 代价：词表越大，嵌入层参数量越大（见 num_params 的说明）
    vocab_size: int = 32000

    # 隐藏层宽度 d_model：每个 token 被表示成多长的向量。
    #   - 越大，单层"容量"越大，但计算量和显存也越大
    #   - 1024 是一个"能力更强但单卡还能训"的折中值
    hidden_size: int = 1024

    # Transformer 块的数量（深度）：模型有多少层堆叠。
    #   - 论文 MobileLLM 指出：小模型"深而窄"往往比"宽而浅"效果好
    #   - 24 层配 1024 宽，得到一个约 0.3B 的模型
    num_hidden_layers: int = 24

    # ------------------------------------------------------------------ #
    # 二、注意力（Attention）相关
    # ------------------------------------------------------------------ #

    # 查询（Query）头的数量
    num_attention_heads: int = 16

    # 键值（Key/Value）头的数量，用于 GQA（分组查询注意力）。
    #   - GQA = 多个 Q 头共享同一组 K/V 头，能省显存、提速，效果几乎不掉
    #   - 必须能被 num_attention_heads 整除（16 / 4 = 4 个 Q 头共享 1 组 KV）
    num_key_value_heads: int = 4

    # 每个注意力头的维度 = hidden_size / num_attention_heads
    # 这里 1024 / 16 = 64，是 Llama/Qwen 的常见取值
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    # ------------------------------------------------------------------ #
    # 三、前馈网络（FFN / SwiGLU）相关
    # ------------------------------------------------------------------ #

    # 前馈网络的中间维度。
    #   - SwiGLU 结构里是 gate/up/down 三路，中间维度通常取 hidden 的 2.75~3 倍
    #   - 这里 2816 ≈ 1024 * 2.75
    intermediate_size: int = 2816

    # MoE 专家独立的中间维度（None = 和 intermediate_size 一样）。
    #   - MoE 里每个专家都是一整套 FFN，专家若和 Dense FFN 一样大会让参数量爆炸
    #   - 所以专家通常做得更小（DeepSeek 也这样），1024 是常见取值
    expert_intermediate_size: int = None

    # 激活函数名：silu 就是 SwiGLU 用的激活函数
    hidden_act: str = "silu"

    # ------------------------------------------------------------------ #
    # 四、归一化 & 位置编码
    # ------------------------------------------------------------------ #

    # RMSNorm 的 epsilon（防止除以 0 的一个极小值）
    rms_norm_eps: float = 1e-6

    # RoPE 旋转位置编码的基频。
    #   - 1e6 是 Llama3 / Qwen2 的常见取值
    rope_theta: float = 1e6

    # 训练时的最大序列长度（token 数）。
    #   - 中文大约 1.5 字符 = 1 token，2048 token ≈ 3000 字
    #   - 先用 2048 起步，后续用 YaRN 外推到更长
    max_position_embeddings: int = 2048

    # 是否启用 YaRN 长文本外推（推理时可开到 4 倍长度）
    #   这里先关掉，保持训练简单，后面讲位置编码时再打开
    use_yarn: bool = False

    # ------------------------------------------------------------------ #
    # 五、Dropout & 精度
    # ------------------------------------------------------------------ #

    # Dropout 概率。小模型一般不缺正则，默认 0（关掉）。
    dropout: float = 0.0

    # 混合精度类型。
    #   ⚠️ 服务器是 V100（Volta 架构），没有原生 bf16 张量核，
    #      必须用 "fp16"，否则训练会慢一个量级。
    dtype: str = "fp16"

    # ------------------------------------------------------------------ #
    # 六、特殊 token id（和分词器约定一致）
    # ------------------------------------------------------------------ #

    bos_token_id: int = 1      # 句首标记 <|im_start|>
    eos_token_id: int = 2      # 句尾标记 <|im_end|>
    pad_token_id: int = 0      # 填充标记 <|endoftext|>

    # ------------------------------------------------------------------ #
    # 七、MoE（混合专家）—— 默认关闭，后续作为可选项
    #     开启后 FFN 变成"多个专家 + 门控路由"，DeepSeek 那套。
    # ------------------------------------------------------------------ #

    use_moe: bool = False
    n_routed_experts: int = 4       # 路由专家数量
    num_experts_per_tok: int = 2    # 每个 token 激活几个专家
    n_shared_experts: int = 1       # 共享专家数量
    aux_loss_alpha: float = 0.01    # 负载均衡辅助损失系数

    # ------------------------------------------------------------------ #

    def __init__(self, **kwargs):
        """
        允许用关键字参数覆盖任何默认值。

        例子：
            # 想快速测试，用一个更小的配置
            cfg = ModelConfig(hidden_size=768, num_hidden_layers=12)
            # 想开 MoE
            cfg = ModelConfig(use_moe=True, n_routed_experts=4)
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"未知的配置项: {key}")
            setattr(self, key, value)
        # 启动前自检：head 数必须能整除（GQA 的前提）
        self._validate()

    def _validate(self):
        """自检配置是否合法，不合法直接报错，避免训练到一半才发现。"""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) 必须能被 "
                f"num_attention_heads({self.num_attention_heads}) 整除"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads({self.num_attention_heads}) 必须能被 "
                f"num_key_value_heads({self.num_key_value_heads}) 整除（GQA 要求）"
            )

    def to_dict(self):
        """把配置转成字典（用于保存/打印/日志）。"""
        d = {}
        for key in dir(self):
            if key.startswith("_"):
                continue
            value = getattr(self, key)
            if isinstance(value, property) or callable(value):
                continue
            d[key] = value
        return d

    def __repr__(self):
        """让 print(cfg) 的输出更直观。"""
        return f"ModelConfig(vocab={self.vocab_size}, "
        f"hidden={self.hidden_size}, layers={self.num_hidden_layers}, "
        f"heads={self.num_attention_heads}, kv_heads={self.num_key_value_heads}, "
        f"moe={self.use_moe})"


# ============================================================================ #
#  B. 预设规格表
# ============================================================================ #

# 这里集中管理不同大小的模型规格，训练脚本里直接 `ModelConfig(**PRESETS["..."])` 即可。
PRESETS = {
    # 主力模型：约 0.3B，能力比 minimind 明显更强，单卡 V100 可训
    "novamind-0.3b": {
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2816,
        "max_position_embeddings": 2048,
    },
    # 快速冒烟测试用：约 0.1B，几十分钟就能跑完一个小实验
    "novamind-0.1b": {
        "vocab_size": 32000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "intermediate_size": 2048,
        "max_position_embeddings": 1024,
    },
    # 对照：minimind 的 104M 规格，用来体会 NovaMind 比它强在哪
    "minimind-104m": {
        "vocab_size": 6400,
        "hidden_size": 768,
        "num_hidden_layers": 16,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 2048,
        "max_position_embeddings": 32768,
    },
    # MoE 版本：4 路由专家 + 1 共享专家，专家用较小的中间维度（1024）
    # 总参数量约 0.5B，但每个 token 只激活 2 个专家（稀疏计算，计算量接近 Dense 0.3B）
    "novamind-moe-0.5b": {
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2816,
        "expert_intermediate_size": 1024,
        "max_position_embeddings": 2048,
        "use_moe": True,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "aux_loss_alpha": 0.01,
    },
}


# ============================================================================ #
#  C. 参数量估算
# ============================================================================ #

def num_params(config: ModelConfig, verbose: bool = True) -> dict:
    """
    不建模型也能估算参数量，用来验证配置是否合理（是否太大/太小）。

    【估算逻辑】一个 Decoder-Only 模型的参数只来自这几处：

    1. 词嵌入层 embed_tokens:
        参数量 = vocab_size * hidden_size
        例子(0.3b): 32000 * 1024 = 3276.8 万

    2. 每一层 Transformer 块包含：
       (a) 注意力 Q/K/V/O 四个线性层：
             Q: hidden_size * hidden_size
             K: hidden_size * (num_kv_heads * head_dim)
             V: hidden_size * (num_kv_heads * head_dim)
             O: hidden_size * hidden_size
           其中 num_kv_heads * head_dim = key/value 的总维度（比 hidden 小，这就是 GQA 省参数的地方）
           例子: Q=1024*1024, K=1024*256, V=1024*256, O=1024*1024
                 = 1048576 + 262144 + 262144 + 1048576 = 262.1 万
       (b) FFN 的 gate/up/down 三路:
             gate: hidden_size * intermediate_size
             up:   hidden_size * intermediate_size
             down: intermediate_size * hidden_size
           例子: 3 * 1024 * 2816 = 864.9 万
       (c) RMSNorm 两个（参数极少，可忽略）
       每层合计 ≈ 262.1 + 864.9 = 1127 万

    3. 总参数 ≈ embed_tokens + 每层合计 * num_hidden_layers
       例子(0.3b): 3276.8万 + 1127万 * 24 = 3276.8 + 27048 = 30324.8万 ≈ 0.3B ✓

    【注意】这里最后还有一个 lm_head（输出层），但因为我们采用"权重绑定"，
    lm_head 和 embed_tokens 共用同一份权重，所以不重复计算。
    """
    hidden = config.hidden_size
    n_kv = config.num_key_value_heads
    head_dim = config.head_dim  # hidden / num_heads

    # 1. 词嵌入
    embed = config.vocab_size * hidden

    # 2. 单个注意力块的参数量
    q = hidden * hidden
    k = hidden * (n_kv * head_dim)
    v = hidden * (n_kv * head_dim)
    o = hidden * hidden
    attn_per_layer = q + k + v + o

    # 3. FFN 参数量（SwiGLU 三路；MoE 时是「多个专家 + 共享专家」之和）
    if config.use_moe:
        exp_ffn = 3 * hidden * (config.expert_intermediate_size or config.intermediate_size)
        ffn_per_layer = exp_ffn * (config.n_routed_experts + config.n_shared_experts)
    else:
        ffn_per_layer = 3 * hidden * config.intermediate_size

    # 4. 每层合计
    per_layer = attn_per_layer + ffn_per_layer

    # 5. 总参数（权重绑定，不额外算 lm_head）
    total = embed + per_layer * config.num_hidden_layers

    result = {
        "embed": embed,
        "attn_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "per_layer": per_layer,
        "total": total,
        "total_M": total / 1e6,
    }

    if verbose:
        print(f"===== 参数量估算: {config!r} =====")
        print(f"  词嵌入层:      {embed/1e6:8.2f} M")
        print(f"  注意力/层:     {attn_per_layer/1e6:8.2f} M")
        print(f"  FFN/层:        {ffn_per_layer/1e6:8.2f} M")
        print(f"  每层合计:      {per_layer/1e6:8.2f} M")
        print(f"  ──────────────────────────────")
        print(f"  总参数量:      {total/1e6:8.2f} M  ({total/1e9:.2f} B)")

    return result


# ============================================================================ #
#  运行演示：`python config.py` 会走到这里
# ============================================================================ #

if __name__ == "__main__":
    print(__doc__)
    print("\n\n########## 示例：查看不同规格的参数量 ##########\n")

    for name, preset in PRESETS.items():
        cfg = ModelConfig(**preset)
        print(f"\n---------- {name} ----------")
        num_params(cfg, verbose=True)
