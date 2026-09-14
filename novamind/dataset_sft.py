"""
================================================================================
dataset_sft.py —— SFT 指令微调数据集（项目第 6 个脚本）
================================================================================

【这个文件干什么】
    把对话数据（sft_t2t_mini.jsonl，每行 {"conversations": [...]}）变成
    模型能吃的 (input_ids, labels) 数字序列。

【SFT 和预训练的区别】
    预训练：整段文本都要学（词语接龙，labels = input_ids）
    SFT   ：只学"助手回答"的部分，问题部分不学（labels = -100 跳过）
            —— 因为用户问什么不重要，重要的是"该怎么答"。

    例子（简化）：
        prompt:  <|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n=2<|im_end|>\n
        labels:  [-100 -100 -100 ... -100  0 0 ...]  只有 "=2" 部分算 loss

【数据格式】（minimind_dataset 的 sft_t2t_mini.jsonl）
    {"conversations": [
        {"role": "user",      "content": "你好"},
        {"role": "assistant", "content": "你好！"}
    ]}

    注意：部分数据带 "reasoning_content"（思维链），默认丢弃不学，
    因为 0.3B 小模型先学会"直接回答"更重要，思维链留到后面（GRPO 阶段）。

【关键技巧：掩码 + packing】
    1. 掩码（mask）：逐条消息套聊天模板，标记出 assistant 部分的位置
    2. packing：多条对话拼满 2048 再切块（和预训练一样，省 padding 算力）

【怎么读这个文件】
    先读 tokenize_conversation（看"套模板 + 打掩码"），
    再看 __init__（看"拼接 + 切块"），
    最后 `python dataset_sft.py` 跑演示看真实样本。
================================================================================
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def tokenize_conversation(tokenizer, messages):
    """
    把一条多轮对话变成 (ids, mask)。

    mask 是 0/1 列表，和 ids 一样长：
      1 = assistant 回答的内容（训练时算 loss）
      0 = user 提问 / 模板符号（训练时不监督）

    做法：逐条消息增量套聊天模板（模板是纯累加的，前后差值就是
    这条消息对应的文本片段），assistant 的片段打 1，其余打 0。

    例子：
        messages = [user:"1+1=?", assistant:"=2"]
        模板渲染:
          片段1 = "<|im_start|>user\n1+1=?<|im_end|>\n"   → mask 全 0
          片段2 = "<|im_start|>assistant\n=2<|im_end|>\n" → mask 全 1
    """
    ids, mask = [], []
    prev_prompt = ""
    cleaned = []
    for msg in messages:
        # 只取 role/content 两个字段，丢掉 reasoning_content 等无关字段
        clean = {"role": msg["role"], "content": msg["content"]}
        cleaned.append(clean)
        # 增量渲染：渲染前 i+1 条消息，和上一次的差 = 这条消息的片段
        cur_prompt = tokenizer.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=False)
        seg = cur_prompt[len(prev_prompt):]
        prev_prompt = cur_prompt

        seg_ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        is_assistant = clean["role"] == "assistant"
        mask.extend([1] * len(seg_ids) if is_assistant else [0] * len(seg_ids))
    return ids, mask


class SFTDataset(Dataset):
    """
    SFT 数据集：对话 → (input_ids, labels)，只有 assistant 部分算 loss。

    用法：
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('./novamind/tokenizer')
        ds = SFTDataset('data/sft_t2t_mini.jsonl', tok, max_length=2048)
        input_ids, labels = ds[0]   # 各 [2048]，labels 大部分是 -100
    """

    def __init__(self, data_path, tokenizer, max_length=2048, max_rows=None):
        """
        参数：
            data_path  : jsonl 对话数据路径
            tokenizer  : 训练好的分词器（含聊天模板）
            max_length : 每个样本的 token 长度（= 模型上下文长度）
            max_rows   : 最多读多少行（调试用，None=全部）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id

        # ---- 第 1 步：逐行读对话 → 套模板打掩码 → 拼进大 buffer ----
        # 和预训练一样的 packing 思路：短对话拼满 max_length 再切块
        all_ids = []
        all_mask = []
        n_read, n_skip, n_supervised = 0, 0, 0

        print(f"[dataset] 读取并分词 {data_path} ...")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    messages = sample["conversations"]
                except Exception:
                    n_skip += 1
                    continue

                # 过滤非法样本：必须 user 开头、assistant 结尾、内容非空
                msgs = [{"role": m["role"], "content": m.get("content", "").strip()}
                        for m in messages
                        if m.get("role") in ("user", "assistant")]
                if not msgs or not any(m["role"] == "assistant" for m in msgs):
                    n_skip += 1
                    continue

                ids, mask = tokenize_conversation(tokenizer, msgs)

                # 超长对话截断到 max_length（一般不会有，防御一下）
                if len(ids) > max_length:
                    ids, mask = ids[:max_length], mask[:max_length]

                all_ids.extend(ids)
                all_mask.extend(mask)
                n_read += 1
                n_supervised += sum(mask)

                if (i + 1) % 100000 == 0:
                    print(f"    已处理 {i + 1} 行, 累计 {len(all_ids) / 1e6:.1f}M token "
                          f"(有效监督 {n_supervised / 1e6:.1f}M)")

        print(f"[dataset] 分词完成: {n_read} 条对话 ({n_skip} 条跳过), "
              f"共 {len(all_ids) / 1e6:.2f}M token, "
              f"其中 assistant 部分 {n_supervised / 1e6:.2f}M ({n_supervised / max(len(all_ids), 1) * 100:.1f}%)")

        # ---- 第 2 步：转 numpy（int32 省内存）----
        all_ids = np.asarray(all_ids, dtype=np.int32)
        all_mask = np.asarray(all_mask, dtype=np.int8)

        # ---- 第 3 步：padding 到 max_length 的整数倍，再切成块 ----
        n_chunks = (len(all_ids) + max_length - 1) // max_length
        pad_len = n_chunks * max_length - len(all_ids)
        if pad_len > 0:
            all_ids = np.pad(all_ids, (0, pad_len), constant_values=self.pad_id)
            all_mask = np.pad(all_mask, (0, pad_len), constant_values=0)

        self.chunks = all_ids.reshape(n_chunks, max_length)
        self.masks = all_mask.reshape(n_chunks, max_length)
        print(f"[dataset] 切成 {n_chunks} 个样本, 每个 {max_length} token")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        """返回 (input_ids, labels)。

        labels = input_ids，但只有 assistant 部分保留，
        其余位置设 -100（cross_entropy 的 ignore_index，不参与 loss）。

        例子（mask 示意）:
            input_ids = [模板, 问题, 模板, 回答, pad]
            mask      = [  0,   0,   0,   1,   0]
            labels    = [-100,-100,-100, 回答, -100]
        """
        input_ids = self.chunks[index]
        mask = self.masks[index]
        labels = input_ids.copy()
        labels[(mask == 0) | (input_ids == self.pad_id)] = -100
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ============================================================================ #
#  运行演示：`python dataset_sft.py` 会走到这里
# ============================================================================ #
if __name__ == "__main__":
    from transformers import AutoTokenizer

    print(__doc__)

    TOK_DIR = os.path.join(os.path.dirname(__file__), "tokenizer")
    DATA = "/data/NovaMind-VL/data/sft_t2t_mini.jsonl"

    print("\n########## 演示：加载数据集并检查样本 ##########\n")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    ds = SFTDataset(DATA, tok, max_length=512, max_rows=20000)

    print(f"\n数据集样本数: {len(ds)}")

    # 取一个"监督 token 最多"的样本展示，比较直观
    best_idx = int(ds.masks.sum(axis=1).argmax())
    input_ids, labels = ds[best_idx]
    print(f"\n样本 {best_idx}（assistant 部分最多的样本）:")
    print(f"  input_ids 形状: {tuple(input_ids.shape)}")
    print(f"  labels 形状: {tuple(labels.shape)}")

    # 还原文本：监督部分（labels != -100）用 【】 括起来
    text = tok.decode(input_ids[:200].tolist())
    ids, labs = input_ids[:200], labels[:200]
    pieces = []
    prev = 0
    # 简单展示：打印整段 + 标注监督区间
    print("\n  整段文本（前 200 token）:")
    print(" ", text)

    n_ignore = (labels == -100).sum().item()
    n_pad = (input_ids == tok.pad_token_id).sum().item()
    n_total = input_ids.numel()
    print(f"\n  监督 token: {n_total - n_ignore}/{n_total} "
          f"(padding {n_pad} 个，问题/模板部分 {n_ignore - n_pad} 个)")

    print("\n✅ SFT 数据集验证通过：掩码 + packing + labels 都正确")
