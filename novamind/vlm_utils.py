"""
================================================================================
vlm_utils.py —— VLM 训练共用工具（分词器扩词表 / 图像特征拼接）
================================================================================

三个函数被阶段 1（train_vlm_pretrain.py）和阶段 2（train_vlm_sft.py）共用：
  1. prepare_vlm_tokenizer : 给分词器加 <|image|> token，存到新目录
  2. resize_llm_vocab      : LLM 词表 32000 → 32001，新行均值初始化
  3. splice_image_features : 把图像特征插进 embedding 的 <|image|> 位置
================================================================================
"""

import os

import torch
from torch import nn
from transformers import AutoTokenizer

IMAGE_TOKEN = "<|image|>"


# ============================================================================ #
#  1. VLM 分词器：现有分词器 + <|image|> 特殊 token
# ============================================================================ #
def prepare_vlm_tokenizer(tokenizer_dir, vlm_tokenizer_dir):
    """在现有分词器上加 <|image|> 特殊 token，存到新目录。"""
    if os.path.exists(os.path.join(vlm_tokenizer_dir, "tokenizer.json")):
        return AutoTokenizer.from_pretrained(vlm_tokenizer_dir)

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    os.makedirs(vlm_tokenizer_dir, exist_ok=True)
    tok.save_pretrained(vlm_tokenizer_dir)
    # 手动复制聊天模板 jinja（save_pretrained 不一定会带）
    src = os.path.join(tokenizer_dir, "chat_template.jinja")
    dst = os.path.join(vlm_tokenizer_dir, "chat_template.jinja")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, dst)
    print(f"[tokenizer] 已生成 {vlm_tokenizer_dir}（新增 {IMAGE_TOKEN}，词表 {len(tok)}）")
    return AutoTokenizer.from_pretrained(vlm_tokenizer_dir)


# ============================================================================ #
#  2. LLM 词表扩一格（给 <|image|>）
# ============================================================================ #
def resize_llm_vocab(model, old_vocab, new_vocab):
    """
    把 LLM 的词表从 old_vocab 扩到 new_vocab（+1 个 <|image|>）。
    新行用旧词表均值初始化（最稳妥，不影响已有能力）。
    注意：调用前先 load_state_dict（按原词表完整加载）。
    """
    embed = model.model.embed_tokens
    assert embed.weight.shape[0] == old_vocab, \
        f"词表大小不符: 模型 {embed.weight.shape[0]} vs 分词器 {old_vocab}"
    hidden = embed.weight.shape[1]

    old_weight = embed.weight.data
    device = old_weight.device
    new_embed = nn.Embedding(new_vocab, hidden).to(device)
    new_embed.weight.data[:old_vocab] = old_weight
    new_embed.weight.data[old_vocab:] = old_weight.mean(dim=0, keepdim=True)

    new_lm_head = nn.Linear(hidden, new_vocab, bias=False).to(device)
    new_lm_head.weight.data[:old_vocab] = old_weight
    new_lm_head.weight.data[old_vocab:] = old_weight.mean(dim=0, keepdim=True)

    model.model.embed_tokens = new_embed
    model.lm_head = new_lm_head
    model.lm_head.weight = model.model.embed_tokens.weight   # 重新绑定
    model.config.vocab_size = new_vocab
    print(f"[LLM] 词表 {old_vocab} → {new_vocab}（新行用均值初始化）")


# ============================================================================ #
#  3. 图像特征拼接：1 个 <|image|> token → N 个特征向量
# ============================================================================ #
def splice_image_features(embed_tokens, input_ids, labels, attention_mask,
                          img_feats, img_pos):
    """
    把图像特征插到 <|image|> 位置：1 个 token → N 个特征向量。
    同时把 labels/attention_mask 对应展开（特征位置的 label = -100，不监督）。

    img_pos 为 -1 的样本（没有图像 token）：原样返回，不做拼接。
    返回 inputs_embeds, labels, attention_mask（长度 = 原长 + N - 1，右侧 pad 对齐）
    """
    b = input_ids.shape[0]
    embeds = embed_tokens(input_ids)          # [B, L, D]

    out_embeds, out_labels, out_masks = [], [], []
    max_new_len = 0
    for i in range(b):
        pos = img_pos[i].item()
        if pos < 0:
            # 无图像样本：直接用文本 embedding
            e, l, m = embeds[i], labels[i], attention_mask[i]
        else:
            n = img_feats[i].shape[0]
            e = torch.cat([embeds[i][:pos], img_feats[i], embeds[i][pos + 1:]], dim=0)
            l = torch.cat([labels[i][:pos],
                           torch.full((n,), -100, dtype=labels.dtype, device=labels.device),
                           labels[i][pos + 1:]], dim=0)
            m = torch.cat([attention_mask[i][:pos],
                           torch.ones(n, dtype=attention_mask.dtype, device=attention_mask.device),
                           attention_mask[i][pos + 1:]], dim=0)
        out_embeds.append(e); out_labels.append(l); out_masks.append(m)
        max_new_len = max(max_new_len, e.shape[0])

    # 右侧 pad 对齐
    D = out_embeds[0].shape[1]
    for i in range(b):
        pad = max_new_len - out_embeds[i].shape[0]
        if pad > 0:
            out_embeds[i] = torch.cat([out_embeds[i],
                torch.zeros(pad, D, dtype=out_embeds[i].dtype, device=out_embeds[i].device)], dim=0)
            out_labels[i] = torch.cat([out_labels[i],
                torch.full((pad,), -100, dtype=out_labels[i].dtype, device=out_labels[i].device)], dim=0)
            out_masks[i] = torch.cat([out_masks[i],
                torch.zeros(pad, dtype=out_masks[i].dtype, device=out_masks[i].device)], dim=0)
    return (torch.stack(out_embeds), torch.stack(out_labels), torch.stack(out_masks))
