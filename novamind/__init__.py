"""NovaMind-VL 核心包。

对外暴露最常用的类和函数，方便训练/推理脚本 import：
    from novamind import ModelConfig, PRESETS, num_params
    from novamind import NovaMindForCausalLM
"""

from .config import ModelConfig, PRESETS, num_params
from .model import (
    RMSNorm,
    Attention,
    FeedForward,
    MoEGate,
    MOEFeedForward,
    NovaMindBlock,
    NovaMindModel,
    NovaMindForCausalLM,
    ModelOutput,
)

__all__ = [
    "ModelConfig",
    "PRESETS",
    "num_params",
    "RMSNorm",
    "Attention",
    "FeedForward",
    "MoEGate",
    "MOEFeedForward",
    "NovaMindBlock",
    "NovaMindModel",
    "NovaMindForCausalLM",
    "ModelOutput",
]
