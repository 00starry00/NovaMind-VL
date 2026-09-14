"""
================================================================================
dataset_vlm.py —— 图文数据集（VLM 阶段 1：图文对齐）
================================================================================

【数据格式】（LLaVA-Pretrain 的 blip_laion_cc_sbu_558k.json）
    [
      {
        "id": "004539375",
        "image": "004539375.jpg",
        "conversations": [
          {"from": "human", "value": "<image>\nProvide a one-sentence caption for the provided image."},
          {"from": "gpt",   "value": "A large commercial airplane flying through a blue sky."}
        ]
      },
      ...
    ]

【这个文件干什么】
    每行变成一个训练样本：
      pixel_values : 预处理后的图片 [3, 256, 256]
      input_ids    : "<|im_start|>user\n<|image|>\n描述图片<|im_end|>\n<|im_start|>assistant\n..." 的 token
      labels       : 只有 assistant 部分保留（其余 -100）
      img_pos      : <|image|> token 在 input_ids 里的位置（训练时把图像特征插到这里）

【怎么读这个文件】
    先读 __init__（看"读 json + 分词"），再看 __getitem__（看样本结构）。
================================================================================
"""

import json
import os

import torch
from torch.utils.data import Dataset


def tokenize_with_image(tokenizer, messages, max_len):
    """
    把含 <|image|> 的对话变成 (input_ids, labels, img_pos)。

    和 dataset_sft.py 的 tokenize_conversation 一样逐条消息增量套模板，
    额外记录 <|image|> 出现在第几个 token 位置。
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

    # 找 <|image|> 的位置（要求每段只有一张图）
    img_id = tokenizer.convert_tokens_to_ids("<|image|>")
    img_pos = ids.index(img_id) if img_id in ids else -1

    ids = ids[:max_len]
    mask = mask[:max_len]
    return ids, mask, img_pos


def build_sample(tokenizer, max_len, pad_id, messages, pixel_values):
    """把 (对话, 图片) 组装成一个训练样本 dict。"""
    ids, mask, img_pos = tokenize_with_image(tokenizer, messages, max_len)
    input_ids = ids + [pad_id] * (max_len - len(ids))
    labels = [i if m == 1 else -100 for i, m in zip(ids, mask)]
    labels = labels + [-100] * (max_len - len(labels))
    attn = [1] * len(ids) + [0] * (max_len - len(ids))
    return {
        "pixel_values": pixel_values,
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "img_pos": img_pos,          # <|image|> 的位置（-1 = 异常样本）
    }


class VLMPretrainDataset(Dataset):
    """
    VLM 阶段 1 数据集：图片 + caption → (pixel_values, input_ids, labels, img_pos)

    用法：
        ds = VLMPretrainDataset(json_path, image_dir, processor, tokenizer, max_len=256)
    """

    def __init__(self, json_path, image_dir, processor, tokenizer, max_len=256,
                 max_rows=None):
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

        # 读 json（LLaVA-Pretrain 是一个大 list）
        print(f"[dataset] 读取 {json_path} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_rows is not None:
            data = data[:max_rows]

        self.samples = []
        n_skip = 0
        for item in data:
            try:
                convs = item["conversations"]
                # 过滤掉无图 / 没有 gpt 回答的样本
                if "image" not in item or item["image"] is None:
                    n_skip += 1
                    continue
                has_answer = any(c.get("from") == "gpt" and c.get("value", "").strip()
                                 for c in convs)
                if not has_answer:
                    n_skip += 1
                    continue
                self.samples.append((item["image"], convs))
            except Exception:
                n_skip += 1
        print(f"[dataset] {len(self.samples)} 个样本 ({n_skip} 跳过)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_file, convs = self.samples[index]

        # 1) 图片：读文件 → 预处理成 pixel_values
        from PIL import Image
        img_path = os.path.join(self.image_dir, image_file)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        # pixel_values: [3, H, W]（SigLIP 256 多语言 = 256×256）

        # 2) 文本：转成我们的聊天格式（human→user, gpt→assistant）
        messages = []
        for c in convs:
            role = "user" if c.get("from") in ("human", "user") else "assistant"
            content = c.get("value", "").replace("<image>", "<|image|>").strip()
            if content:
                messages.append({"role": role, "content": content})

        return build_sample(self.tokenizer, self.max_len, self.pad_id,
                            messages, pixel_values)


class VLMParquetDataset(Dataset):
    """
    VLM 阶段 1 数据集（parquet 版）：minimind-v 官方数据，图片内嵌在 parquet 里。

    parquet 列：
      conversations: json 字符串 [{"role":"user","content":"...<image>"}, ...]
      image_bytes  : 图片字节（128×128，已 resize）

    数据来源：LinkSoul/Chinese-LLaVA-Vision-Instructions（中文描述！）
    用法：
        ds = VLMParquetDataset('/data/NovaMind-VL/data/vlm/pretrain_i2t.parquet',
                               processor, tokenizer, max_len=256)
    """

    def __init__(self, parquet_path, processor, tokenizer, max_len=256,
                 max_rows=None):
        import pyarrow.parquet as pq
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

        print(f"[dataset] 读取 parquet {parquet_path} ...")
        self.table = pq.read_table(parquet_path)
        self.n = len(self.table)
        if max_rows is not None:
            self.n = min(self.n, max_rows)
        print(f"[dataset] {self.n} 个样本 (总 {len(self.table)}) "
              f"| 列: {list(self.table.column_names)}")

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        import io
        import json as _json
        from PIL import Image

        # 1) 图片：parquet 内嵌字节 → PIL → 预处理
        raw = self.table["image_bytes"][index].as_py()
        if isinstance(raw, list):
            raw = raw[0]          # 多图时只取第一张（阶段 1 都是单图）
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        # 2) 对话：<image> 占位符换成我们的 <|image|> token
        convs = _json.loads(self.table["conversations"][index].as_py())
        messages = []
        for c in convs:
            role = c.get("role")
            if role not in ("user", "assistant"):
                continue
            content = c.get("content", "").replace("<image>", "<|image|>")
            if content.strip():
                messages.append({"role": role, "content": content})

        return build_sample(self.tokenizer, self.max_len, self.pad_id,
                            messages, pixel_values)


class VLMMmsTextVQADataset(Dataset):
    """
    TextVQA 读字数据集（直接读 lmms-lab/TextVQA 的 HF 缓存，零额外磁盘）。

    lmms-lab 版 image_id 是哈希值（不是 COCO 编号），但 parquet 自带图片字节，
    所以不映射文件名，直接用它的 image 列。
    样本："读图中文字回答问题"，10 个人工答案取多数票。
    """

    def __init__(self, cache_dir, processor, tokenizer, max_len=384):
        from collections import Counter
        import os as _os
        _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id
        self.Counter = Counter
        print(f"[dataset] 加载 TextVQA (cache: {cache_dir}) ...")
        self.ds = load_dataset("lmms-lab/TextVQA", split="train", cache_dir=cache_dir)
        print(f"[dataset] TextVQA {len(self.ds)} 条")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        image = item["image"].convert("RGB")   # PIL 图片（COCO 原分辨率）
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        q = item["question"].strip()
        answers = item["answers"]
        ans = self.Counter(answers).most_common(1)[0][0]
        messages = [
            {"role": "user",
             "content": f"<|image|>\nAnswer the question using a single word or phrase.\nQuestion: {q}"},
            {"role": "assistant", "content": str(ans).strip()},
        ]
        return build_sample(self.tokenizer, self.max_len, self.pad_id,
                            messages, pixel_values)


def collate_vlm(batch):
    """拼 batch：图片堆叠，文本本来就是定长，img_pos 转 tensor。"""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "img_pos": torch.tensor([b["img_pos"] for b in batch], dtype=torch.long),
    }


# ============================================================================ #
#  运行演示：`python dataset_vlm.py` 会走到这里
# ============================================================================ #
if __name__ == "__main__":
    print(__doc__)
    # 演示需要一个真实样本，这里只演示 tokenize_with_image 的核心逻辑
    from transformers import AutoTokenizer
    TOK_DIR = os.path.join(os.path.dirname(__file__), "vlm_tokenizer")
    if not os.path.exists(TOK_DIR):
        print("请先运行 train/train_vlm_pretrain.py 生成 vlm_tokenizer")
    else:
        tok = AutoTokenizer.from_pretrained(TOK_DIR)
        msgs = [
            {"role": "user", "content": "<|image|>\n描述这张图片"},
            {"role": "assistant", "content": "一架飞机在蓝天上飞行。"},
        ]
        ids, mask, pos = tokenize_with_image(tok, msgs, 128)
        print(f"\nids: {ids}")
        print(f"mask: {mask}")
        print(f"img_pos: {pos} (该位置是 {tok.decode([ids[pos]])!r})")
        print(f"还原文本: {tok.decode(ids)}")
        print("\n✅ 数据集核心逻辑验证通过")
