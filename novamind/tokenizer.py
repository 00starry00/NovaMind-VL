"""
================================================================================
tokenizer.py —— 分词器：把「文字」变成「数字」（项目第 2 个脚本）
================================================================================

【为什么重要】
    模型不认识字，只认识数字。分词器就是一本「词典」：
        句子  --切分-->  一串 token  --查词典-->  一串 id  --喂给模型-->  一串 id 再还原成文字

    这决定了模型「看世界」的颗粒度：
      - minimind 词表只有 6400 → 中文被拆成很碎的小片段，语义被切烂，理解能力弱。
      - NovaMind 用 32000 → 常用词能整词保留，压缩率高、语义完整。

【BPE 到底在干嘛（一句话）】
    从「每个字符/字节是一个 token」开始，反复把「出现最频繁的相邻 token 对」合并成
    一个新 token，直到词表达到目标大小。这样常见词会被合并成整体，生僻内容保留字符级。

【本文件内容】
    A. BytePairEncoder —— 纯 Python 手写 BPE（教学用，理解算法，小规模可跑）
    B. train_tokenizer() —— 用 HuggingFace tokenizers 训练正式分词器（快、生产可用）
    C. __main__ 演示 —— 先跑玩具 BPE 看合并过程，再训练一个小词表验证全流程

【怎么读】
    先读 A 的 train()（理解「统计→合并→再统计」的循环），
    再看 B（理解正式训练、特殊 token、保存成 HF 格式），
    最后 `python tokenizer.py` 跑一遍看输出。
================================================================================
"""

import os
from collections import Counter

# ============================================================================ #
#  特殊 token 约定（必须和 config.py 里的 id 保持一致）
# ============================================================================ #
# 这些是「控制字符」，不参与语义，专门用来标记句首、句尾、填充。
# 训练时按顺序给 id：0, 1, 2, 3 ...
SPECIAL_TOKENS = [
    "<|endoftext|>",   # 0: pad 填充 / unk 未知词
    "<|im_start|>",    # 1: 句首（也是 bos）
    "<|im_end|>",      # 2: 句尾（也是 eos）
]

# 聊天模板：把多轮对话拼成模型认识的格式（Qwen 风格 im_start/im_end）。
# 例子：user 说"你好"，assistant 回复时，输入会是：
#   <|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


# ============================================================================ #
#  A. 纯 Python 手写 BPE（教学用）
# ============================================================================ #
class BytePairEncoder:
    """
    从零手写的字节级 BPE，用来彻底搞懂算法。规模小，只适合教学，不适合大语料。

    【核心数据】
      self.vocab  : {token_id: bytes}   词表，初始是 256 个单字节
      self.merges : [(a, b, new_id), ...]  按顺序记录的合并规则

    【一个完整的例子】
      语料 "low lower" 的字节序列里，'l' 和 'o' 相邻出现频繁，
      BPE 会把 ('l','o') 合并成新 token "lo"，再把 ('lo','w') 合并成 "low"。
      训练结束时，词表里既有单字节，也有 "low" 这种整词，编码时优先用整词。
    """

    def __init__(self, target_vocab: int = 512):
        # 目标词表大小。256 是起点（所有单字节），所以能学 target_vocab - 256 次合并。
        self.target_vocab = target_vocab
        # 初始词表：256 个单字节 → 对应的 bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []  # 按学习顺序记录 (a, b, new_id)

    def _stats(self, ids):
        """统计相邻 token 对出现的次数。

        例子：ids = [1, 2, 2, 3]
              zip(ids, ids[1:]) = [(1,2), (2,2), (2,3)]
              返回 {(1,2):1, (2,2):1, (2,3):1}
        """
        return Counter(zip(ids, ids[1:]))

    def _merge_once(self, ids, a, b, new_id):
        """把序列里所有相邻的 (a, b) 替换成 new_id（一次从左到右扫描）。

        例子：ids=[65,66,66,67]，合并 (66,67)->256
              → [65, 66, 256]
        注意：每遇到一对 (a,b) 就消耗两个位置，i 跳 2，避免重复合并。
        """
        out = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, verbose: bool = True):
        """训练 BPE：反复「统计 → 合并最频繁的一对 → 更新词表」，直到达到目标大小。

        例子（text = "aaabdaaabac"）：
          第 1 次统计最频繁的是 ('a','a')，合并成新 token "aa"
          第 2 次统计最频繁的是 ('aa','a')，合并成 "aaa" ... 依此类推
        """
        ids = list(text.encode("utf-8"))  # 初始：每个字节一个 token
        while len(self.vocab) < self.target_vocab:
            stats = self._stats(ids)
            if not stats:
                break  # 语料太小，已经没得合并了
            a, b = stats.most_common(1)[0][0]  # 最频繁的一对
            new_id = len(self.vocab)           # 新 token 的 id = 当前词表大小
            # 新 token = 两个旧 token 的字节拼接
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merges.append((a, b, new_id))
            ids = self._merge_once(ids, a, b, new_id)
            if verbose:
                print(f"    合并 {a}({self.vocab[a]!r}) + {b}({self.vocab[b]!r}) "
                      f"-> {new_id}({self.vocab[new_id]!r}), 频次 {stats[(a, b)]}")
        return ids

    def encode(self, text: str):
        """把文字编码成 id 序列：先转字节，再按学习顺序依次应用所有合并规则。"""
        ids = list(text.encode("utf-8"))
        for a, b, new_id in self.merges:
            ids = self._merge_once(ids, a, b, new_id)
        return ids

    def decode(self, ids):
        """把 id 序列还原成文字：查词表拿字节，拼起来解码。"""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


# ============================================================================ #
#  B. 正式分词器训练（HuggingFace tokenizers，快、生产可用）
# ============================================================================ #
def train_tokenizer(corpus_files, vocab_size: int = 32000, output_dir: str = "./tokenizer",
                    min_frequency: int = 2):
    """
    用 HuggingFace `tokenizers` 库训练一个 ByteLevel BPE 分词器，并保存成 HF 格式。

    【为什么自己写 BPE 但正式训练用库】
      手写的 BytePairEncoder 是 O(语料长度) 每轮还要扫全量，大语料上慢到没法用；
      `tokenizers` 是 Rust 实现，几 GB 语料几分钟就能训完，算法本质和手写版一模一样。

    【参数】
      corpus_files : list[str]  语料文件路径列表（.txt 或 .jsonl）
      vocab_size   : int        目标词表大小（NovaMind 用 32000）
      output_dir   : str        输出目录，会生成 tokenizer.json + tokenizer_config.json
      min_frequency: int        一个 pair 至少出现多少次才允许合并（过滤噪声）

    【产物】
      训练 + 保存后，其它脚本用 `AutoTokenizer.from_pretrained(output_dir)` 即可加载。
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    # 1) 建一个空的 BPE 分词器
    tokenizer = Tokenizer(models.BPE())

    # 2) 预分词器：ByteLevel 表示在「字节」层面做 BPE（而不是字符），
    #    这样任何语言（中/英/emoji）都能统一处理，且天然无 OOV（未知词）。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3) 解码器：训练和编码用的是字节，解码时也要用 ByteLevel 还原成正常文字。
    tokenizer.decoder = decoders.ByteLevel()

    # 4) 训练器：指定词表大小、特殊 token、最小频次。
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,   # 会按顺序占 id 0,1,2
        min_frequency=min_frequency,
    )

    # 5) 在语料上训练
    tokenizer.train(corpus_files, trainer)

    # 6) 包成 HF 的 tokenizer，绑定特殊 token 名称，并写入聊天模板
    os.makedirs(output_dir, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(output_dir)
    return hf_tokenizer


def load_tokenizer(tokenizer_dir):
    """加载已训练的分词器（训练脚本和推理脚本都复用这个函数）。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def _iter_texts(jsonl_path, field="text", max_lines=None):
    """从 jsonl 逐行 yield 文本字段（生成器，逐行读不占内存）。

    参数 max_lines: 最多读多少行（None=全部）。
        训分词器不需要全部语料，抽样几十万行就够，还能省内存、加快速度。

    例子：pretrain_t2t_mini.jsonl 每行是 {"text": "..."}，
         yield 出来的就是那个 "..." 字符串，直接喂给分词器训练。
    """
    import json
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)[field]
            except Exception:
                continue


def train_tokenizer_from_jsonl(jsonl_path, vocab_size=32000, output_dir="./tokenizer",
                               min_frequency=2, max_lines=None):
    """从 jsonl 语料（{"text": ...} 格式）训练正式分词器。

    参数 max_lines: 抽样行数（None=全部）。
        全量 127 万行会吃较多内存（容器限制 61G），抽样 40 万行已足够训出 32000 词表。

    和 train_tokenizer() 的区别：
      - train_tokenizer() 读纯文本文件，用 tokenizer.train(文件列表)
      - 本函数读 jsonl，用 train_from_iterator(生成器)，省去中间落盘 .txt 的步骤
    其余逻辑（ByteLevel BPE + 特殊 token + 保存 HF 格式）完全一样。
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(_iter_texts(jsonl_path, max_lines=max_lines), trainer)

    os.makedirs(output_dir, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(output_dir)
    return hf_tokenizer


# ============================================================================ #
#  C. 运行演示
# ============================================================================ #
if __name__ == "__main__":
    print(__doc__)
    print("\n\n########## 演示 1：纯 Python 手写 BPE（看合并过程） ##########\n")

    # 一段中英混合的小语料（故意重复，方便观察合并）
    sample = "low lower lowest 我爱北京天安门 我爱中国 中国 china"
    bpe = BytePairEncoder(target_vocab=300)
    print("训练语料:", sample)
    print("训练过程（前 20 次合并）：")
    bpe.train(sample, verbose=True)
    # 只打印前 20 次（train 已经打印了，这里截断显示）
    print("\n编码 'lower 我爱北京':", bpe.encode("lower 我爱北京"))
    print("解码验证:", bpe.decode(bpe.encode("lower 我爱北京")))
    print("(手写 BPE 演示完毕，能复现「统计→合并→更新」的循环即理解到位)")

    print("\n\n########## 演示 2：正式分词器训练（小词表快速验证全流程） ##########\n")

    # 造一个小语料文件，训练一个 2000 词表的分词器做端到端验证
    demo_corpus = os.path.join(os.path.dirname(__file__), "_demo_corpus.txt")
    demo_lines = [
        "人工智能是研究如何让计算机模拟人类智能的学科。",
        "大语言模型通过海量文本学习语言的规律。",
        "深度学习是机器学习的一个分支，使用多层神经网络。",
        "The transformer architecture powers modern language models.",
        "自然语言处理让机器理解人类的语言。",
        "训练一个语言模型需要高质量的数据和算力。",
        "Hello world, this is a tokenizer demo.",
        "强化学习通过奖励信号来优化模型的策略。",
    ]
    with open(demo_corpus, "w", encoding="utf-8") as f:
        f.write("\n".join(demo_lines) * 50)  # 复制 50 遍凑够频次

    out_dir = os.path.join(os.path.dirname(__file__), "tokenizer_demo")
    tok = train_tokenizer([demo_corpus], vocab_size=2000, output_dir=out_dir, min_frequency=1)

    print(f"词表大小: {tok.vocab_size}")
    for text in ["人工智能", "大语言模型", "Transformer", "强化学习"]:
        ids = tok.encode(text)
        print(f"  '{text}' -> {ids} -> 解码回: {tok.decode(ids)}")
    print(f"\n演示分词器已保存到: {out_dir}")
    print(f"后续用 load_tokenizer('{out_dir}') 即可加载")
