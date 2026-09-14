"""
================================================================================
train_vlm_sft.py —— VLM 阶段 2：看图问答微调（项目第 13 个脚本）
================================================================================

【这个脚本干什么】
    在阶段 1 训练好的投影层基础上，把 LLM 也解冻一部分（LoRA），
    用 30 万条图文问答数据（sft_i2t.parquet）教模型"看图回答问题"。

【和阶段 1 的区别】
    阶段 1：LLM 全冻结，只训投影层 —— 学"图像特征 ↔ 文字"对齐
    阶段 2：投影层 + LLM LoRA 一起训 —— 学"看图对话/问答"的真实能力

【为什么 LLM 用 LoRA 而不是全参数】
    图文问答数据 30 万条，全参数训会把纯文本对话能力洗掉（又是灾难性遗忘）。
    LoRA 冻结主干只训 4M 适配器，和身份注入同一个道理。

【怎么运行】
    cd /data/NovaMind-VL
    nohup /data/miniconda/envs/torch/bin/python train/train_vlm_sft.py > vlm_s2.log 2>&1 &

【前置】阶段 1 已产出 out/vlm_projector_s1.pt
================================================================================
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from novamind import ModelConfig, PRESETS, NovaMindForCausalLM
from novamind.vision_model import NovaMindVision
from novamind.dataset_vlm import VLMParquetDataset, collate_vlm
from novamind.vlm_utils import prepare_vlm_tokenizer, resize_llm_vocab, splice_image_features
from novamind.lora import apply_lora, save_lora, merge_lora


def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ============================================================================ #
#  冒烟测试：随机数据验证管线
# ============================================================================ #
class SmokeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_len, n=200):
        self.tok = tokenizer
        self.max_len = max_len
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        from novamind.dataset_vlm import tokenize_with_image
        msgs = [
            {"role": "user", "content": f"<|image|>\n这张图片里有什么？第 {index} 号"},
            {"role": "assistant", "content": f"图里有第 {index} 号物体，颜色鲜艳。"},
        ]
        ids, mask, img_pos = tokenize_with_image(self.tok, msgs, self.max_len)
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


def main():
    parser = argparse.ArgumentParser(description="NovaMind-VL 阶段 2 看图问答")
    # 数据与路径
    parser.add_argument("--data_path", type=str,
                        default="/data/NovaMind-VL/data/vlm/sft_i2t.parquet")
    parser.add_argument("--data_format", type=str, default="parquet",
                        choices=["parquet", "json"])
    parser.add_argument("--extra_json_paths", type=str, default="",
                        help="额外 LLaVA 格式 json（逗号分隔），与主数据混合训练，图片在 json_image_dir")
    parser.add_argument("--textvqa_cache", type=str, default="",
                        help="lmms-lab/TextVQA 的 HF 缓存目录（设置了就混入读字数据）")
    parser.add_argument("--json_image_dir", type=str, default="/data/NovaMind-VL/data/vlm_hd/train2014/train2014")
    parser.add_argument("--vision_path", type=str,
                        default="/data/models/siglip-384")
    parser.add_argument("--projector_path", type=str,
                        default="/data/NovaMind-VL/out/vlm_projector_s1.pt",
                        help="阶段 1 训好的投影层")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/tokenizer")
    parser.add_argument("--vlm_tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/vlm_tokenizer")
    parser.add_argument("--save_dir", type=str, default="/data/NovaMind-VL/out")
    parser.add_argument("--save_weight", type=str, default="vlm_s2_0.3b",
                        help="合并权重保存名（HD 重训用 vlm_hd_s2_0.3b 避免覆盖旧版）")
    parser.add_argument("--llm_weight", type=str, default="sft_clean_0.3b",
                        help="LLM 主干（纯文本 SFT 权重，阶段 1 没有动过它）")
    # 模型
    parser.add_argument("--preset", type=str, default="novamind-0.3b", choices=list(PRESETS.keys()))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    # 训练超参数
    # ⚠️ 显存实测：batch 16 x seq 768 不设环境变量 = 31.6G OOM；
    #    batch 12 x max_len 448 + expandable_segments = 2.8G（实测安全）。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=448,
                        help="文本长度上限（sft 数据 95% ≤ 431 token，448 覆盖 95%）")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=2000)
    # 调试
    parser.add_argument("--smoke", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 62)
    print(f" NovaMind-VL 阶段2 看图问答 | LoRA r={args.rank} | batch={args.batch_size} "
          f"| lr={args.learning_rate}")
    print("=" * 62)

    # ========== 1. 分词器 ==========
    tokenizer = prepare_vlm_tokenizer(args.tokenizer_dir, args.vlm_tokenizer_dir)

    # ========== 2. LLM：加载 → 扩词表 → LoRA ==========
    config = ModelConfig(**PRESETS[args.preset])
    model = NovaMindForCausalLM(config).to(args.device)
    weight_path = os.path.join(args.save_dir, f"{args.llm_weight}.pth")
    if not os.path.exists(weight_path):
        print(f"[错误] 主干权重不存在: {weight_path}")
        sys.exit(1)
    state = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(state)
    resize_llm_vocab(model, model.config.vocab_size, len(tokenizer))

    # LoRA 注入（LLM 主干冻结，只训 A/B）
    apply_lora(model, r=args.rank, alpha=args.alpha)

    # ========== 3. 视觉塔：编码器冻结 + 阶段 1 投影层 ==========
    vision_path = "dummy" if args.smoke else args.vision_path
    vision = NovaMindVision(vision_path, llm_hidden=config.hidden_size).to(args.device)
    if not args.smoke:
        proj_state = torch.load(args.projector_path, map_location=args.device)
        vision.projector.load_state_dict(proj_state)
        print(f"[视觉] 投影层加载自 {args.projector_path}")

    # ========== 4. 数据集 ==========
    if args.smoke:
        ds = SmokeDataset(tokenizer, args.max_len)
        print("[冒烟] 使用随机数据（200 样本）验证管线")
    else:
        from torch.utils.data import ConcatDataset
        from novamind.dataset_vlm import VLMPretrainDataset
        ds_list = []
        if args.data_format == "parquet":
            main = VLMParquetDataset(
                args.data_path, vision.image_processor, tokenizer,
                max_len=args.max_len, max_rows=(args.max_rows or None))
        else:
            main = VLMPretrainDataset(
                args.data_path, args.json_image_dir, vision.image_processor, tokenizer,
                max_len=args.max_len, max_rows=(args.max_rows or None))
        ds_list.append(main)
        for jp in args.extra_json_paths.split(","):
            jp = jp.strip()
            if not jp:
                continue
            extra = VLMPretrainDataset(
                jp, args.json_image_dir, vision.image_processor, tokenizer,
                max_len=args.max_len)
            ds_list.append(extra)
            print(f"[数据] 混入 {jp}: {len(extra)} 条")
        if args.textvqa_cache:
            from novamind.dataset_vlm import VLMMmsTextVQADataset
            tvqa = VLMMmsTextVQADataset(
                args.textvqa_cache, vision.image_processor, tokenizer,
                max_len=args.max_len)
            ds_list.append(tvqa)
            print(f"[数据] 混入 TextVQA: {len(tvqa)} 条")
        ds = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, collate_fn=collate_vlm)
    print(f"[数据] {len(ds)} 样本, {len(loader)} batch/epoch")

    # ========== 5. 优化器：投影层 + LoRA A/B ==========
    trainable = [p for p in vision.projector.parameters() if p.requires_grad] + \
                [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable) / 1e6
    print(f"[优化] 可训练参数 {n_train:.2f}M（投影层 + LoRA）")
    optimizer = optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    total_steps = len(loader) * args.epochs

    # ========== 6. 训练循环 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    loss_log = open(os.path.join(args.save_dir, f"{args.save_weight}_loss.txt"), "a")
    global_step = 0
    model.train()
    vision.projector.train()

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

            img_feats = vision(pixel)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                inputs_embeds, labels, attn = splice_image_features(
                    model.model.embed_tokens, input_ids, labels, attn, img_feats, img_pos)
                res = model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attn)
                loss = res.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
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
                save_lora(model, os.path.join(args.save_dir, f"{args.save_weight}_lora.pt"))
                torch.save(vision.projector.state_dict(),
                           os.path.join(args.save_dir, f"{args.save_weight}_projector.pt"))
                print("[保存] LoRA + 投影层已存")

        print(f"Epoch {epoch + 1} 完成, 用时 {(time.time() - epoch_start) / 60:.1f} 分钟")

    # ========== 7. 保存 ==========
    save_lora(model, os.path.join(args.save_dir, f"{args.save_weight}_lora.pt"))
    torch.save(vision.projector.state_dict(),
               os.path.join(args.save_dir, f"{args.save_weight}_projector.pt"))

    # 合并 LoRA 回主干，存完整权重（推理用）
    merged = merge_lora(model)
    merged_path = os.path.join(args.save_dir, f"{args.save_weight}.pth")
    torch.save(merged, merged_path)
    loss_log.close()
    print(f"\n✅ VLM 阶段 2 完成！完整权重: {merged_path}")
    print(f"注意：完整权重不含投影层（{args.save_weight}_projector.pt 单独存）")


if __name__ == "__main__":
    main()
