"""
================================================================================
lora.py —— LoRA 低秩适配（身份 / 医疗 / 考试等小数据注入专用）
================================================================================

【为什么需要 LoRA】
    全参数微调小数据集（如身份问答）会把模型"练坏"：
    模型背得太死，无论问什么都回答身份介绍（灾难性遗忘）。

    LoRA 的思路：主干权重全部冻结，只训练两个很小的低秩矩阵 A/B，
    输出 = 主干输出 + (α/r)·B·A·x。
    主干一个字不动，所以通用能力永远不会被破坏；
    训练量也小（4M vs 303M 参数），几分钟搞定。

【原理公式】
    W' = W + (α/r)·B·A     （W 冻结，A∈R^(r×in)，B∈R^(out×r)，r<<min(in,out)）
    r=8, α=16 时 scale = 2

【用法】
    model = NovaMindForCausalLM(config)
    model.load_state_dict(base)          # 先加载主干权重（如 sft_0.3b）
    apply_lora(model, r=8, alpha=16)     # 注入 LoRA，冻结主干
    ... 训练（只有 A/B 有梯度）...
    save_lora(model, "lora_identity.pt") # 存适配器（只有几百 KB~几 MB）
    merge_lora(model)                    # 推理时融回主干，得到完整权重
================================================================================
"""

import torch
from torch import nn

# 默认注入的目标层：注意力 4 个投影 + FFN 3 个投影（lm_head/embedding 不动）
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """
    包装一个冻结的 Linear，加上低秩增量 A/B。

    输出 = base(x) + scale · B(A(x))
        A: (r, in)   低秩压缩
        B: (out, r)  低秩还原
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        assert not base.bias, "LoRA 只支持 bias=False 的层（本项目都是）"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False          # 主干冻结

        self.r = r
        self.scale = alpha / r               # α/r，通常取 2
        out_f, in_f = base.weight.shape
        # A 高斯小值初始化，B 全零初始化 → 训练开始时增量 = 0，不影响主干输出
        # 注意：要跟主干同一个设备（cuda），dtype 保持 fp32（优化器更新更稳，autocast 会转换）
        self.lora_A = nn.Parameter(
            torch.randn(r, in_f, device=base.weight.device) * 0.02)
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, r, device=base.weight.device))

    def forward(self, x):
        # x @ A^T: [B,T,in]→[B,T,r]；@ B^T: [B,T,r]→[B,T,out]
        delta = (x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return self.base(x) + delta * self.scale


def apply_lora(model, r=8, alpha=16, targets=LORA_TARGETS):
    """
    把 targets 里的 Linear 替换成 LoRALinear，并冻结主干、
    只留 A/B 可训练。返回可训练参数量（M）。
    """
    n_replaced = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha))
                n_replaced += 1

    # 冻结一切，只解冻 LoRA 的 A/B
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[LoRA] 替换 {n_replaced} 个层 (r={r}, alpha={alpha}), "
          f"可训练参数 {n_train:.2f}M")
    return model


def save_lora(model, path):
    """只保存 LoRA 的 A/B 参数（几 MB），不含主干权重。"""
    state = {name: p.data.cpu() for name, p in model.named_parameters()
             if "lora_A" in name or "lora_B" in name}
    torch.save(state, path)
    print(f"[LoRA] 适配器已保存至 {path} ({len(state)} 个张量)")


def load_lora(model, path):
    """加载适配器到已 apply_lora 的模型上。"""
    state = torch.load(path, map_location="cpu")
    matched = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in state:
                p.data.copy_(state[name].to(p.device))
                matched += 1
    print(f"[LoRA] 已加载 {path} ({matched} 个张量)")


@torch.no_grad()
def merge_lora(model):
    """
    把 A/B 融回主干（W += scale·B·A），并把 LoRALinear 换回普通 Linear。
    返回完整 state_dict（fp16，可直接 torch.save 成 .pth 给推理脚本用）。
    """
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                delta = (child.lora_B @ child.lora_A) * child.scale
                child.base.weight.data += delta
                plain = nn.Linear(child.base.in_features, child.base.out_features,
                                  bias=False)
                plain.weight.data = child.base.weight.data
                setattr(parent, child_name, plain)
    state = {k: v.half().cpu() for k, v in model.state_dict().items()}
    return state
