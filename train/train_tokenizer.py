"""
================================================================================
train_tokenizer.py —— 训练 NovaMind 的 32000 词表 BPE 分词器（你手动运行的脚本）
================================================================================

【这个脚本干什么】
    读预训练纯文本语料，训练一个 32000 词表的 ByteLevel BPE 分词器，
    保存成 HuggingFace 格式（tokenizer.json + tokenizer_config.json），
    然后自动验证：词表大小、特殊 token id、中文整词编码、压缩率。

【怎么运行（在训练服务器上）】
    1. 登录训练服务器（服务器地址请见团队内部文档）

    2. 激活环境并进入项目目录
       source /data/miniconda/etc/profile.d/conda.sh && conda activate torch
       cd /data/NovaMind-VL

    3. 运行本脚本
       python train/train_tokenizer.py

【预期耗时】
    1.24GB 语料、127 万行，Rust 版 tokenizers 训练约 2~5 分钟。
================================================================================
"""

import os
import sys
import time
import functools

# 强制 print 立即写盘：否则 nohup 后台运行时，Python 会把输出缓存在内存里，
# 导致 tail -f 什么都看不到（只有进程结束时才一次性刷出来）。
print = functools.partial(print, flush=True)

# 把项目根目录加进模块搜索路径，这样能 import 到 novamind 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novamind.tokenizer import train_tokenizer_from_jsonl

# ---- 路径和超参数（想改就改这里）----
DATA = "/data/NovaMind-VL/data/pretrain_t2t_mini.jsonl"   # 预训练纯文本语料
OUT = "/data/NovaMind-VL/novamind/tokenizer"              # 分词器输出目录
VOCAB_SIZE = 32000                                        # 目标词表大小（和 config.py 一致）
MIN_FREQUENCY = 2                                         # 一个 pair 至少出现几次才允许合并
MAX_LINES = 400000        # 抽样行数：训分词器不用全量，40 万行足够且省内存（容器限 61G）


def main():
    print("=" * 62)
    print(" 训练 32000 词表 BPE 分词器")
    print(f" 语料: {DATA}")
    print(f" 抽样: {MAX_LINES} 行 (None=全量)")
    print(f" 输出: {OUT}")
    print(f" 词表: {VOCAB_SIZE}  最小频次: {MIN_FREQUENCY}")
    print("=" * 62)
    print("\n提示: 训练分三步——预处理 -> 统计词 -> 合并(32000 次迭代)。")
    print("      'Count pairs' 完成后会进入合并阶段，这一阶段【没有进度条】，")
    print("      终端会安静几分钟，请耐心等待，不要以为卡住了。\n")

    t0 = time.time()
    tok = train_tokenizer_from_jsonl(
        DATA,
        vocab_size=VOCAB_SIZE,
        output_dir=OUT,
        min_frequency=MIN_FREQUENCY,
        max_lines=MAX_LINES,
    )
    print(f"\n✅ 训练完成，用时 {time.time() - t0:.1f}s")

    # ============ 自动验证 ============
    print("\n" + "=" * 62)
    print(" 验证结果")
    print("=" * 62)

    print(f"词表大小: {tok.vocab_size}  (目标 {VOCAB_SIZE})")
    if tok.vocab_size < VOCAB_SIZE:
        print("⚠️  词表没涨满，可能是语料不够多样或 min_frequency 设太大")

    # 特殊 token id，必须和 config.py 一致：pad=0, bos=1, eos=2
    print(f"特殊 token id: pad={tok.pad_token_id}, bos={tok.bos_token_id}, eos={tok.eos_token_id}")
    print(f"                (期望 pad=0, bos=1, eos=2)")

    print("\n中文整词编码测试（好的分词器：常用词应是 1 个 token）：")
    for text in ["人工智能", "大语言模型", "深度学习", "机器学习", "Transformer", "你好，世界！"]:
        ids = tok.encode(text)
        print(f"  '{text}' -> {len(ids)} token, {ids} -> 解码回: {tok.decode(ids)}")

    # 压缩率：中文好的分词器约 1.5~2 字/token
    sample = "人工智能正在改变世界，大语言模型让机器能够理解人类的语言并生成自然的回复。"
    n_tok = len(tok.encode(sample))
    print(f"\n压缩率测试: '{sample}'")
    print(f"  {len(sample)} 个字符 -> {n_tok} 个 token（约 {len(sample) / n_tok:.1f} 字/token，期望 1.5~2）")

    print(f"\n✅ 分词器已保存到: {OUT}")
    print(f"   后续脚本用 load_tokenizer('{OUT}') 即可加载")


if __name__ == "__main__":
    main()
