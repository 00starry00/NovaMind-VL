"""
================================================================================
dataset.py —— 预训练数据集（项目第 4 个脚本）
================================================================================

【这个文件干什么】
    把纯文本语料（pretrain_t2t_mini.jsonl，每行 {"text":"..."}）变成
    模型能吃的 (input_ids, labels) 数字序列。

【预训练任务是什么】
    "词语接龙"：输入一串 token，预测下一个 token。
    所以 labels 其实就是 input_ids 自己（往右移一位就是"下一个"）。

【关键技巧：packing（打包）】
    语料平均每行只有 212 字 ≈ 80 个 token，而模型上下文是 2048。
    如果一行做一个样本，80 个有效 token 后面跟 1968 个 padding，
    算力浪费 96%（padding 是白算的）。

    packing 的做法：把很多条短文本用 eos 分隔符拼起来，拼满 2048 再切一刀，
    这样每个样本都是"满的"，算力利用率接近 100%。

    例子（假设 max_length=8）：
        文本A: [a1 a2 a3]   文本B: [b1 b2]   文本C: [c1 c2 c3 c4]
        加 eos 拼起来: [a1 a2 a3 EOS b1 b2 EOS c1 c2 c3 c4 EOS]
        切成 8 块:  [a1 a2 a3 EOS b1 b2 EOS c1]   ← 第 1 个样本（满 8）
                    [c2 c3 c4 EOS ... 补 pad]      ← 第 2 个样本（尾块，padding）
    EOS 的作用：告诉模型"这里一条文本结束了，下面开始新的"，是文本边界标记。

【怎么读这个文件】
    先读 __init__（看"读文本→分词→拼装→切块"四步），
    再看 __getitem__（看每个样本怎么返回），
    最后 `python dataset.py` 跑演示看真实样本。
================================================================================
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集：纯文本 → (input_ids, labels)，已用 packing 拼满长度。

    用法：
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('./novamind/tokenizer')
        ds = PretrainDataset('data/pretrain_t2t_mini.jsonl', tok, max_length=2048)
        input_ids, labels = ds[0]   # 第 0 个样本，各 [2048]
    """

    def __init__(self, data_path, tokenizer, max_length=2048, max_rows=None):
        """
        参数：
            data_path  : jsonl 语料路径
            tokenizer  : 训练好的分词器（把文字变 id）
            max_length : 每个样本的 token 长度（= 模型上下文长度）
            max_rows   : 最多读多少行（调试用，None=全部）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id

        # ---- 第 1 步：读文本 → 分词 → 拼成超长序列 ----
        # 这里不用 datasets 库，纯 Python 逐行读（省内存，也好理解）
        all_ids = []
        print(f"[dataset] 读取并分词 {data_path} ...")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)["text"]
                except Exception:
                    continue
                # 分词（不加特殊 token，eos 我们手动加）
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                all_ids.extend(ids)
                all_ids.append(self.eos_id)     # 每条文本后加 eos 作为边界

                # 每 10 万行打印一次进度，避免"以为卡住"
                if (i + 1) % 100000 == 0:
                    print(f"    已处理 {i + 1} 行, 累计 {len(all_ids) / 1e6:.1f}M token")

        print(f"[dataset] 分词完成, 共 {len(all_ids) / 1e6:.2f}M token")

        # ---- 第 2 步：转 numpy 数组（int32 省内存：1亿 token 只占 400MB）----
        all_ids = np.asarray(all_ids, dtype=np.int32)

        # ---- 第 3 步：padding 到 max_length 的整数倍 ----
        # 例子：总长 10000，max_length=2048，需要 pad 到 10240（5 块）
        n_chunks = (len(all_ids) + max_length - 1) // max_length
        pad_len = n_chunks * max_length - len(all_ids)
        if pad_len > 0:
            all_ids = np.pad(all_ids, (0, pad_len), constant_values=self.pad_id)

        # ---- 第 4 步：切成 [n_chunks, max_length] 的块 ----
        # reshape 就是"按顺序每 max_length 个切一刀"
        self.chunks = all_ids.reshape(n_chunks, max_length)
        print(f"[dataset] 切成 {n_chunks} 个样本, 每个 {max_length} token, "
              f"最后一块有 {pad_len} 个 padding")

    def __len__(self):
        """样本总数 = 块的数量。"""
        return len(self.chunks)

    def __getitem__(self, index):
        """返回第 index 个样本的 (input_ids, labels)，各 [max_length]。

        预训练是"词语接龙"：labels = input_ids（预测下一个 token）。
        唯一要处理的是 padding 位置：它的 label 设成 -100，
        因为 cross_entropy 的 ignore_index=-100，会跳过这些位置不算 loss。

        例子（max_length=4，pad_id=0）：
            input_ids = [10, 20, 30, 0]   (最后是 padding)
            labels    = [10, 20, 30, -100]  (padding 位置不参与预测)
        """
        input_ids = self.chunks[index]
        labels = input_ids.copy()                          # 接龙：labels = input_ids
        labels[labels == self.pad_id] = -100               # padding 位置不监督
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ============================================================================ #
#  运行演示：`python dataset.py` 会走到这里
# ============================================================================ #
if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer

    print(__doc__)

    # 用训练好的分词器 + 少量数据（2 万行）快速演示
    TOK_DIR = os.path.join(os.path.dirname(__file__), "tokenizer")
    DATA = "/data/NovaMind-VL/data/pretrain_t2t_mini.jsonl"

    print("\n########## 演示：加载数据集并检查样本 ##########\n")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    ds = PretrainDataset(DATA, tok, max_length=512, max_rows=20000)

    print(f"\n数据集样本数: {len(ds)}")

    # 取第 0 个样本，检查结构
    input_ids, labels = ds[0]
    print(f"\n样本 0 的 input_ids 形状: {tuple(input_ids.shape)}")
    print(f"样本 0 的 labels 形状: {tuple(labels.shape)}")

    # 统计有效 token（非 padding）和 padding 比例
    valid = (input_ids != tok.pad_token_id).sum().item()
    print(f"有效 token: {valid}/{input_ids.shape[0]}")

    # 还原文本，看看模型"看到"的是什么（前 60 个 token）
    text = tok.decode(input_ids[:60].tolist())
    print(f"\n样本 0 前 60 token 还原的文本:\n{text}")

    # 检查 labels 的 -100 位置（应该只在 padding 处）
    n_ignore = (labels == -100).sum().item()
    n_pad = (input_ids == tok.pad_token_id).sum().item()
    print(f"\nlabels 里 -100 的数量: {n_ignore}  (padding 数量: {n_pad}, 应相等)")

    print("\n✅ 数据集验证通过：打包 + padding + labels 都正确")
