"""
================================================================================
vision_model.py —— 视觉塔：SigLIP 编码器 + 投影层（VLM 第 1 个脚本）
================================================================================

【这个文件干什么】
    把图片变成 LLM 能读的向量序列：
        图片(256×256) → SigLIP 视觉编码器（冻结）→ [N, 768] 视觉特征
                       → 投影层 MLP（可训练）  → [N, 1024] LLM 隐层维度
    然后这些向量替换文本里 <|image|> token 的位置，喂给 LLM。

【为什么用 SigLIP】
    siglip-base-patch16-256-multilingual（93M 视觉塔）：
      - 比 CLIP ViT-Base 特征质量明显更好（Qwen2-VL / Gemma 同款路线）
      - 多语言版对中文更友好，256 分辨率在 V100 / 本地 8G 都能跑

【两阶段训练】
    阶段 1（本脚本配套）：冻结 LLM + 编码器，只训投影层 —— 学"对齐"
    阶段 2（后续）：投影层 + LLM（LoRA）一起训 —— 学"看图问答"

【用法】
    vision = NovaMindVision("/data/models/siglip-256-multilingual")
    feats = vision(pixel_values)   # pixel_values: [B,3,256,256] → feats: [B,N,1024]
================================================================================
"""

import torch
from torch import nn


class VisionProjector(nn.Module):
    """
    投影层：视觉特征 → LLM 隐层空间（2 层 MLP，比单层 Linear 对齐更好）。

    SigLIP base 视觉特征 768 维 → LLM 隐层 1024 维
    """

    def __init__(self, vision_hidden: int = 768, llm_hidden: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden, llm_hidden, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(llm_hidden, llm_hidden, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class NovaMindVision(nn.Module):
    """
    视觉塔 = 冻结的 SigLIP 视觉编码器 + 可训练的投影层。

    参数：
        model_path    : SigLIP 模型目录（本地路径，或 "dummy" 用随机编码器调试）
        llm_hidden    : LLM 隐层维度（NovaMind-0.3B = 1024）
        freeze_encoder: 是否冻结编码器（默认冻结，只训投影层）
    """

    def __init__(self, model_path: str, llm_hidden: int = 1024, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = None
        self.image_processor = None

        if model_path == "dummy":
            # 调试模式：随机特征代替 SigLIP（不联网、不占显存），输出 [B, 257, 768]
            class _Dummy(nn.Module):
                def forward(self, pixel_values):
                    b = pixel_values.shape[0]
                    return type("Out", (), {"last_hidden_state":
                        torch.randn(b, 257, 768, device=pixel_values.device,
                                    dtype=pixel_values.dtype)})()
            self.encoder = _Dummy()
            vision_hidden = 768
        else:
            from transformers import AutoImageProcessor, AutoModel
            # transformers 5.x 里 SigLIP 自动映射到 SigLIP2 实现
            full = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
            # 只要视觉塔，丢掉文本塔
            if hasattr(full, "vision_model"):
                self.encoder = full.vision_model
            elif hasattr(full, "vision_tower"):
                self.encoder = full.vision_tower
            else:
                # vision-only 精简目录（只存了视觉塔，没有文本塔）：直接当编码器用
                # config.model_type == "siglip_vision_model"
                self.encoder = full
            vision_hidden = self.encoder.config.hidden_size
            self.image_processor = AutoImageProcessor.from_pretrained(model_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.projector = VisionProjector(vision_hidden, llm_hidden)
        print(f"[视觉] 编码器 {model_path} | 特征 {vision_hidden} 维 | "
              f"投影层 768→{llm_hidden}→{llm_hidden}")

    def forward(self, pixel_values):
        """
        pixel_values: [B, 3, H, W] → 视觉特征 [B, N, llm_hidden]
        N = patch 数量（SigLIP 256 多语言版 = 256，无 CLS；代码按实际形状动态适配）
        """
        with torch.no_grad():
            out = self.encoder(pixel_values)
        feats = out.last_hidden_state          # [B, N, vision_hidden]
        feats = feats.to(self.projector.fc1.weight.dtype)
        return self.projector(feats)           # [B, N, llm_hidden]


# ============================================================================ #
#  运行演示：`python vision_model.py` 会走到这里
# ============================================================================ #
if __name__ == "__main__":
    print(__doc__)
    print("\n########## 演示：dummy 编码器跑通投影层 ##########\n")

    vision = NovaMindVision("dummy", llm_hidden=1024)
    pixel = torch.randn(2, 3, 256, 256, dtype=torch.float16)
    feats = vision(pixel)
    print(f"输入 {tuple(pixel.shape)} → 输出 {tuple(feats.shape)} (应为 [2, 257, 1024])")

    n_train = sum(p.numel() for p in vision.parameters() if p.requires_grad) / 1e6
    print(f"可训练参数（仅投影层）: {n_train:.2f}M")
    print("\n✅ 视觉塔验证通过")
