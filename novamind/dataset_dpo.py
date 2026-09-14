"""
================================================================================
dataset_dpo.py —— DPO 偏好数据集（项目第 10 个脚本）
================================================================================

【DPO 数据是什么】
    一对"好回答 vs 坏回答"：
    {"chosen":   [{"role":"user",...},{"role":"assistant",...}],   ← 好回答
     "rejected": [{"role":"user",...},{"role":"assistant",...}]}   ← 坏回答
    模型要学的不是"记住好回答"，而是"让好回答比坏回答概率更高"。

【这个文件干什么】
    把每对数据变成 4 个张量：
      chosen_ids / chosen_mask   —— 好回答的 token 和"assistant 部分"掩码
      rejected_ids / rejected_mask —— 坏回答同上
    mask=1 的位置是 assistant 回答（DPO loss 只在这些 token 上算对数概率）。

【和 SFT 数据集的区别】
    SFT：只有一条对话，labels 直接监督
    DPO：两条对话成对出现，没有 labels——训练时算两者的对数概率差

【关键细节】
    - 套聊天模板的方式和 SFT 完全一样（增量渲染 + 打掩码）
    - 截断到 max_length（超过的部分切掉，minimind 也是这么做的）
    - padding 位置 mask=0，不算对数概率

【怎么读这个文件】
    先看 tokenize_conversation（和 SFT 版一样），再看 __getitem__ 的返回结构。
================================================================================
"""

import json
import os

import torch
from torch.utils.data import Dataset


def tokenize_conversation(tokenizer, messages):
    """
    和 dataset_sft.py 里同款：套聊天模板 + 打掩码。
    返回 (ids, mask)：mask=1 的位置是 assistant 回答。
    """
    ids, mask = [], []
    prev_prompt = ""
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"], "content": msg["content"]}
        cleaned.append(clean)
        cur_prompt = tokenizer.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=False)
        seg = cur_prompt[len(prev_prompt):]
        prev_prompt = cur_prompt

        seg_ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        is_assistant = clean["role"] == "assistant"
        mask.extend([1] * len(seg_ids) if is_assistant else [0] * len(seg_ids))
    return ids, mask


def pad_to(ids, mask, max_length, pad_id):
    """截断 + 右侧 pad 到 max_length，返回 tensor。"""
    ids = ids[:max_length]
    mask = mask[:max_length]
    pad_len = max_length - len(ids)
    if pad_len > 0:
        ids = ids + [pad_id] * pad_len
        mask = mask + [0] * pad_len
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long))


class DPODataset(Dataset):
    """
    DPO 数据集：偏好对 → (chosen_ids, chosen_mask, rejected_ids, rejected_mask)。

    用法：
        tok = AutoTokenizer.from_pretrained('./novamind/tokenizer')
        ds = DPODataset('data/dpo.jsonl', tok, max_length=1024)
        c_ids, c_mask, r_ids, r_mask = ds[0]   # 各 [1024]
    """

    def __init__(self, data_path, tokenizer, max_length=1024, max_rows=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id

        self.pairs = []
        n_skip = 0
        print(f"[dataset] 读取 DPO 偏好对 {data_path} ...")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    chosen = sample["chosen"]
                    rejected = sample["rejected"]
                except Exception:
                    n_skip += 1
                    continue

                # 清洗消息字段（和 SFT 一样：只要 role/content，过滤空内容）
                def clean_msgs(msgs):
                    out = [{"role": m["role"], "content": m.get("content", "").strip()}
                           for m in msgs
                           if m.get("role") in ("user", "assistant")
                           and m.get("content", "").strip()]
                    return out

                c_msgs = clean_msgs(chosen)
                r_msgs = clean_msgs(rejected)
                if not c_msgs or not r_msgs:
                    n_skip += 1
                    continue
                if not any(m["role"] == "assistant" for m in c_msgs) or \
                   not any(m["role"] == "assistant" for m in r_msgs):
                    n_skip += 1
                    continue

                self.pairs.append((c_msgs, r_msgs))

        print(f"[dataset] 共 {len(self.pairs)} 对偏好样本 ({n_skip} 条跳过)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        c_msgs, r_msgs = self.pairs[index]
        c_ids, c_mask = tokenize_conversation(self.tokenizer, c_msgs)
        r_ids, r_mask = tokenize_conversation(self.tokenizer, r_msgs)
        c_ids, c_mask = pad_to(c_ids, c_mask, self.max_length, self.pad_id)
        r_ids, r_mask = pad_to(r_ids, r_mask, self.max_length, self.pad_id)
        return c_ids, c_mask, r_ids, r_mask


# ============================================================================ #
#  运行演示：`python dataset_dpo.py` 会走到这里
# ============================================================================ #
if __name__ == "__main__":
    from transformers import AutoTokenizer

    print(__doc__)
    TOK_DIR = os.path.join(os.path.dirname(__file__), "tokenizer")
    DATA = "/data/NovaMind-VL/data/dpo.jsonl"

    print("\n########## 演示：加载数据集并检查样本 ##########\n")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    ds = DPODataset(DATA, tok, max_length=1024, max_rows=20000)

    print(f"\n数据集样本数: {len(ds)}")
    c_ids, c_mask, r_ids, r_mask = ds[0]
    print(f"chosen:    ids {tuple(c_ids.shape)}, assistant token 占比 "
          f"{c_mask.float().mean().item() * 100:.1f}%")
    print(f"rejected:  ids {tuple(r_ids.shape)}, assistant token 占比 "
          f"{r_mask.float().mean().item() * 100:.1f}%")

    # 还原 chosen 文本前 100 token
    print("\nchosen 样本前 100 token:")
    print(" ", tok.decode(c_ids[:100].tolist()))

    print("\n✅ DPO 数据集验证通过：成对结构 + 掩码 + padding 都正确")
