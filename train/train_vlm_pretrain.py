"""
================================================================================
train_vlm_pretrain.py —— VLM 阶段 1：图文对齐训练（项目第 12 个脚本）
================================================================================

【这个脚本干什么】
    冻结 LLM（sft_clean_0.3b）+ 冻结 SigLIP 编码器，只训练投影层。
    数据：LLaVA-Pretrain 595K 图文对（图片 + 一句 caption）。
    训练目标：投影层学会把视觉特征"翻译"成 LLM 能读懂的向量，
              让 LLM 看到图后能生成描述。

【为什么分两阶段】
    阶段 1：先只学"图像特征 → 文字"的对齐（LLM 不动，学得快、不破坏对话能力）
    阶段 2：再一起训（LoRA）学"看图问答"，那时候才有真正的识图对话能力

【数据格式】（LLaVA-Pretrain）
    [{"id": "...", "image": "xxx.jpg",
      "conversations": [{"from": "human", "value": "<image>\n描述图片"},
                        {"from": "gpt", "value": "caption"}]}, ...]

【前置准备】
    1. 下载 SigLIP：/data/models/siglip-256-multilingual
    2. 下载数据：/data/NovaMind-VL/data/vlm/{images/, blip_laion_cc_sbu_558k.json}
    3. 生成 vlm_tokenizer（本脚本第一次跑会自动生成：加 <|image|> 特殊 token）

【怎么运行（在服务器上）】
    cd /data/NovaMind-VL
    nohup /data/miniconda/envs/torch/bin/python train/train_vlm_pretrain.py > vlm_s1.log 2>&1 &
    tail -f vlm_s1.log

【调试】
    加 --smoke 1：不读数据、用随机图片跑通全流程（验证代码，~1 分钟）
================================================================================
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from novamind import ModelConfig, PRESETS, NovaMindForCausalLM
from novamind.vision_model import NovaMindVision
from novamind.dataset_vlm import VLMPretrainDataset, VLMParquetDataset, collate_vlm
from novamind.vlm_utils import (prepare_vlm_tokenizer, resize_llm_vocab,
                                splice_image_features)


def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ============================================================================ #
#  冒烟测试：不依赖真实数据/编码器，验证整个训练管线
# ============================================================================ #
class SmokeDataset(torch.utils.data.Dataset):
    """随机图片 + 随机 caption，只用来验证代码跑得通。"""

    def __init__(self, tokenizer, max_len, n=200):
        self.tok = tokenizer
        self.max_len = max_len
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        msgs = [
            {"role": "user", "content": f"<|image|>\nDescribe image number {index}"},
            {"role": "assistant", "content": f"a random photo showing something interesting {index}"},
        ]
        ids, mask, img_pos = tokenize_smoke(self.tok, msgs, self.max_len)
        input_ids = ids + [self.tok.pad_token_id] * (self.max_len - len(ids))
        labels = [i if m == 1 else -100 for i, m in zip(ids, mask)]
        labels += [-100] * (self.max_len - len(labels))
        attn = [1] * len(ids) + [0] * (self.max_len - len(ids))
        return {
            "pixel_values": torch.randn(3, 256, 256),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "img_pos": img_pos,
        }


def tokenize_smoke(tokenizer, messages, max_len):
    from novamind.dataset_vlm import tokenize_with_image
    return tokenize_with_image(tokenizer, messages, max_len)


# ============================================================================ #
#  主训练
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="NovaMind-VL 阶段 1 图文对齐")
    # 数据与路径
    parser.add_argument("--data_path", type=str,
                        default="/data/NovaMind-VL/data/vlm/pretrain_i2t.parquet",
                        help="parquet 路径（或 json 模式的 json 路径）")
    parser.add_argument("--image_dir", type=str,
                        default="/data/NovaMind-VL/data/vlm/images",
                        help="json 模式下的图片目录")
    parser.add_argument("--data_format", type=str, default="parquet",
                        choices=["parquet", "json"],
                        help="数据格式：parquet=minimind-v 内嵌图片（推荐），json=LLaVA 图片目录版")
    parser.add_argument("--vision_path", type=str,
                        default="/data/models/siglip-384")
    parser.add_argument("--save_projector", type=str, default="vlm_projector_s1",
                        help="投影层保存名（HD 重训请换名避免覆盖旧版）")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/tokenizer")
    parser.add_argument("--vlm_tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/vlm_tokenizer")
    parser.add_argument("--save_dir", type=str, default="/data/NovaMind-VL/out")
    parser.add_argument("--llm_weight", type=str, default="sft_clean_0.3b")
    # 模型
    parser.add_argument("--preset", type=str, default="novamind-0.3b", choices=list(PRESETS.keys()))
    # 训练超参数（阶段 1 只训投影层，lr 可以大）
    # 注意：LLM 虽然冻结，但梯度要穿过它流到投影层，必须保留完整计算图。
    # batch 32 x seq 512 会把 32G 显存吃光（实测 OOM），batch 16 是安全线。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=2000)
    # 调试
    parser.add_argument("--smoke", type=int, default=0, choices=[0, 1],
                        help="1=冒烟模式：随机数据跑通管线，不读真实数据")
    parser.add_argument("--max_rows", type=int, default=0, help="只读前 N 行（0=全量）")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 62)
    print(f" NovaMind-VL 阶段1 图文对齐 | batch={args.batch_size} | lr={args.learning_rate}")
    print("=" * 62)

    # ========== 1. 分词器（+<|image|>） ==========
    tokenizer = prepare_vlm_tokenizer(args.tokenizer_dir, args.vlm_tokenizer_dir)

    # ========== 2. LLM：加载 + 冻结 + 扩词表 ==========
    config = ModelConfig(**PRESETS[args.preset])
    model = NovaMindForCausalLM(config).to(args.device)
    weight_path = os.path.join(args.save_dir, f"{args.llm_weight}.pth")
    state = torch.load(weight_path, map_location=args.device)

    old_vocab = model.config.vocab_size
    new_vocab = len(tokenizer)
    # 先按原词表完整加载权重，再扩词表（新行由 resize 函数用均值初始化）
    model.load_state_dict(state)
    resize_llm_vocab(model, old_vocab, new_vocab)
    print(f"[LLM] 加载 {args.llm_weight}（词表 {old_vocab} → {new_vocab}）")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("[LLM] 已冻结")

    # ========== 3. 视觉塔（SigLIP 冻结 + 投影层可训） ==========
    vision_path = "dummy" if args.smoke else args.vision_path
    vision = NovaMindVision(vision_path, llm_hidden=config.hidden_size).to(args.device)
    n_train = sum(p.numel() for p in vision.parameters() if p.requires_grad) / 1e6
    print(f"[视觉] 可训练参数（投影层）: {n_train:.2f}M")

    # ========== 4. 数据集 ==========
    if args.smoke:
        ds = SmokeDataset(tokenizer, args.max_len)
        print("[冒烟] 使用随机数据（200 样本）验证管线")
    else:
        vision_processor = vision.image_processor
        assert vision_processor is not None, "dummy 模式不支持真实数据"
        if args.data_format == "parquet":
            ds = VLMParquetDataset(
                args.data_path, vision_processor, tokenizer,
                max_len=args.max_len, max_rows=(args.max_rows or None))
        else:
            ds = VLMPretrainDataset(
                args.data_path, args.image_dir, vision_processor, tokenizer,
                max_len=args.max_len, max_rows=(args.max_rows or None))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, collate_fn=collate_vlm)
    print(f"[数据] {len(ds)} 样本, {len(loader)} batch/epoch")

    # ========== 5. 优化器（只训投影层） ==========
    optimizer = optim.AdamW(vision.projector.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    total_steps = len(loader) * args.epochs

    # ========== 6. 训练循环 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    loss_log = open(os.path.join(args.save_dir, f"{args.save_projector}_loss.txt"), "a")
    global_step = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        ema_loss = None
        for step_in_epoch, batch in enumerate(loader):
            pixel = batch["pixel_values"].to(args.device, non_blocking=True)
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            attn = batch["attention_mask"].to(args.device, non_blocking=True)
            img_pos = batch["img_pos"].to(args.device)

            lr = get_lr(global_step, total_steps, args.warmup_steps, args.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr

    # 图像特征（投影层有梯度）
            img_feats = vision(pixel)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 拼接图像特征进 embedding，LLM 前向（LLM 无梯度，省显存）
                inputs_embeds, labels, attn = splice_image_features(
                    model.model.embed_tokens, input_ids, labels, attn, img_feats, img_pos)
                res = model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attn)
                loss = res.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vision.projector.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_in_epoch += 1

            if step_in_epoch % args.log_interval == 0:
                spent = time.time() - epoch_start
                cur = loss.item()
                ema_loss = cur if ema_loss is None else 0.9 * ema_loss + 0.1 * cur
                mem_used = torch.cuda.memory_allocated() / 1e9
                progress = step_in_epoch / len(loader) * 100
                eta_min = spent / step_in_epoch * (len(loader) - step_in_epoch) / 60
                print(f"Epoch[{epoch + 1}/{args.epochs}] {step_in_epoch}/{len(loader)} "
                      f"({progress:4.1f}%) | loss {cur:.4f} (ema {ema_loss:.4f}) | lr {lr:.2e} "
                      f"| 显存 {mem_used:.1f}G | 已用 {spent / 60:.1f}min 剩 {eta_min:.1f}min")
                loss_log.write(f"{global_step} {cur:.6f} 0.0 {ema_loss:.6f}\n")
                loss_log.flush()

            if step_in_epoch % args.save_interval == 0:
                proj_path = os.path.join(args.save_dir, f"{args.save_projector}.pt")
                torch.save(vision.projector.state_dict(), proj_path)
                print(f"[保存] 投影层已存至 {proj_path}")

        print(f"Epoch {epoch + 1} 完成, 用时 {(time.time() - epoch_start) / 60:.1f} 分钟")

    # ========== 7. 保存投影层 ==========
    proj_path = os.path.join(args.save_dir, f"{args.save_projector}.pt")
    torch.save(vision.projector.state_dict(), proj_path)
    loss_log.close()
    print(f"\n✅ VLM 阶段 1 完成！投影层: {proj_path}")
    print("下一步：阶段 2 训练（投影层 + LLM LoRA 联合看图问答）")


if __name__ == "__main__":
    main()
