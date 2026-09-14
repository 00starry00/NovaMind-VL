"""
================================================================================
train_lora.py —— LoRA 微调脚本（身份 / 小数据注入专用，第 8 个脚本）
================================================================================

【这个脚本干什么】
    在冻结的主干模型上，只训练 LoRA 适配器（A/B 低秩矩阵）。
    用来给模型注入小规模知识：身份（你是谁）、医疗问答、考试题等。
    主干权重一个字不动 → 通用能力零风险。

【和全参数 SFT 的区别】
    全参数：303M 参数全部更新 → 小数据多轮必过拟合（背死）
    LoRA ：只更新 ~4M 参数     → 小数据多轮也没事（主干锁死）

【怎么运行（在服务器上）】
    cd /data/NovaMind-VL
    nohup /data/miniconda/envs/torch/bin/python train/train_lora.py > lora.log 2>&1 &

【产物】
    out/lora_identity.pt         只含 A/B 的适配器（几 MB，可热插拔）
    out/sft_identity_0.3b.pth   合并后的完整权重（推理脚本直接用）
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
from novamind.dataset_sft import SFTDataset
from novamind.lora import apply_lora, save_lora, merge_lora


def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    parser = argparse.ArgumentParser(description="NovaMind LoRA 微调")
    # 数据与路径
    parser.add_argument("--data_path", type=str,
                        default="/data/NovaMind-VL/data/identity.jsonl")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/tokenizer")
    parser.add_argument("--save_dir", type=str, default="/data/NovaMind-VL/out")
    parser.add_argument("--base_weight", type=str, default="sft_0.3b",
                        help="主干权重名（不含 .pth）")
    parser.add_argument("--lora_name", type=str, default="lora_identity",
                        help="适配器保存名（也会存一份合并权重 <merged_name>.pth）")
    parser.add_argument("--merged_name", type=str, default="sft_identity_0.3b",
                        help="合并后完整权重的保存名（推理脚本默认加载它）")
    # 模型
    parser.add_argument("--preset", type=str, default="novamind-0.3b", choices=list(PRESETS.keys()))
    # LoRA 超参数
    parser.add_argument("--rank", type=int, default=8, help="LoRA 秩 r")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA 缩放 α")
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=25,
                        help="身份数据实测 25 epochs 最佳：身份记得住、通用能力不受影响")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    # 日志
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=0, help="调试用：只读前 N 行")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 62)
    print(f" NovaMind LoRA 微调 | 主干 {args.base_weight} | r={args.rank} "
          f"alpha={args.alpha} | lr={args.learning_rate}")
    print("=" * 62)

    # ========== 1. 加载主干 + 注入 LoRA ==========
    config = ModelConfig(**PRESETS[args.preset])
    config.max_position_embeddings = args.max_seq_len
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = NovaMindForCausalLM(config).to(args.device)

    weight_path = os.path.join(args.save_dir, f"{args.base_weight}.pth")
    if not os.path.exists(weight_path):
        print(f"[错误] 主干权重不存在: {weight_path}")
        sys.exit(1)
    state = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(state)
    print(f"[权重] 主干加载自 {weight_path}")

    apply_lora(model, r=args.rank, alpha=args.alpha)

    # ========== 2. 数据集（身份等小数据） ==========
    ds = SFTDataset(
        args.data_path, tokenizer,
        max_length=args.max_seq_len,
        max_rows=(args.max_rows if args.max_rows > 0 else None),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"[数据] {len(ds)} 个样本, {len(loader)} 个 batch/epoch")

    # ========== 3. 优化器：只优化可训练参数（LoRA 的 A/B） ==========
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    total_steps = len(loader) * args.epochs // args.accumulation_steps

    # ========== 4. 训练循环 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    loss_log = open(os.path.join(args.save_dir, f"{args.lora_name}_loss.txt"), "a")
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        ema_loss = None
        for step_in_epoch, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            lr = get_lr(global_step, total_steps, args.warmup_steps, args.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.amp.autocast('cuda', dtype=torch.float16):
                res = model(input_ids, labels=labels)
                loss = (res.loss + res.aux_loss) / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step_in_epoch + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if (step_in_epoch + 1) % args.log_interval == 0:
                cur_loss = loss.item() * args.accumulation_steps
                ema_loss = cur_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * cur_loss
                print(f"Epoch[{epoch + 1}/{args.epochs}] {step_in_epoch + 1}/{len(loader)} "
                      f"| loss {cur_loss:.4f} (ema {ema_loss:.4f}) | lr {lr:.2e} | "
                      f"已用 {(time.time() - epoch_start) / 60:.2f}min")
                loss_log.write(f"{global_step} {cur_loss:.6f} 0.0 {ema_loss:.6f}\n")
                loss_log.flush()

        print(f"Epoch {epoch + 1} 完成, 用时 {(time.time() - epoch_start) / 60:.2f} 分钟")

    # ========== 5. 保存：适配器 + 合并权重 ==========
    adapter_path = os.path.join(args.save_dir, f"{args.lora_name}.pt")
    save_lora(model, adapter_path)

    merged = merge_lora(model)
    merged_path = os.path.join(args.save_dir, f"{args.merged_name}.pth")
    torch.save(merged, merged_path)
    print(f"[保存] 合并权重已存至 {merged_path} (完整模型，推理直接用)")
    loss_log.close()
    print("\n✅ LoRA 微调完成！主干权重始终未变，通用能力零风险")


if __name__ == "__main__":
    main()
