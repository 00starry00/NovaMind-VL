"""
================================================================================
train_sft.py —— SFT 指令微调脚本（项目第 7 个脚本）
================================================================================

【这个脚本干什么】
    在预训练权重（pretrain_0.3b_v2.pth）基础上做指令微调：
        读对话数据 → 只对 assistant 部分算 loss → 更新参数 → 定期保存
    训练完模型就"学会对话"了：看到 user 问题，会生成 assistant 回答。

【和预训练脚本的 3 个区别】
    1. 数据集换成 SFTDataset（对话 + 掩码，只学回答部分）
    2. 从预训练权重加载（--from_weight pretrain_0.3b_v2）
    3. 学习率更低（1e-4 vs 3e-4）：预训练已学会语言，微调要"轻拿轻放"
       避免灾难性遗忘（把预训练学到的知识洗掉）

【怎么运行（在服务器上）】
    cd /data/NovaMind-VL
    nohup /data/miniconda/envs/torch/bin/python train/train_sft.py > sft.log 2>&1 &
    tail -f sft.log    # 看进度

【重要参数】
    --from_weight  pretrain_0.3b_v2   从大数据预训练权重开始微调（默认）
    --epochs       1                  1 个 epoch 足够（1.74GB 数据）
    --learning_rate 1e-4              SFT 学习率（预训练的 1/3）
    --max_rows     N                  调试用：只读前 N 行
================================================================================
"""

import argparse
import math
import os
import sys
import time

# 把项目根目录加入 sys.path，这样能 import novamind 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from novamind import ModelConfig, PRESETS, NovaMindForCausalLM
from novamind.dataset_sft import SFTDataset


# ============================================================================ #
#  学习率调度：warmup + cosine（和预训练同一套，数值更温和）
# ============================================================================ #
def get_lr(step, total_steps, warmup_steps, base_lr):
    """
    学习率曲线：先线性升到 base_lr（warmup），再余弦衰减到 base_lr 的 10%。
    """
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ============================================================================ #
#  断点续训：保存 / 加载
# ============================================================================ #
def save_checkpoint(path, model, optimizer, scaler, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)


def load_checkpoint(path, model, optimizer, scaler):
    ckp = torch.load(path, map_location="cpu")
    model.load_state_dict(ckp["model"])
    optimizer.load_state_dict(ckp["optimizer"])
    scaler.load_state_dict(ckp["scaler"])
    return ckp["epoch"], ckp["step"]


# ============================================================================ #
#  主训练
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="NovaMind SFT 指令微调")
    # 数据与路径
    parser.add_argument("--data_path", type=str,
                        default="/data/NovaMind-VL/data/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/tokenizer")
    parser.add_argument("--save_dir", type=str, default="/data/NovaMind-VL/out")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/NovaMind-VL/checkpoints")
    parser.add_argument("--save_weight", type=str, default="sft_0.3b")
    # 模型
    parser.add_argument("--preset", type=str, default="novamind-0.3b", choices=list(PRESETS.keys()))
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="梯度累积步数（等效 batch = batch_size * accumulation_steps）")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    # 断点续训 / 调试
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--from_weight", type=str, default="pretrain_0.3b_v2",
                        help="从哪个预训练权重开始微调（none=从头训练，不推荐）")
    parser.add_argument("--max_rows", type=int, default=0,
                        help="调试用：只读前 N 行数据（0=全量）")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 让 print 实时输出（nohup 重定向时 tail -f 能看到实时进度）
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # ========== 1. 准备配置 / 分词器 / 模型 ==========
    print("=" * 62)
    print(f" NovaMind SFT 指令微调 | 预设 {args.preset} | batch={args.batch_size} "
          f"x{args.accumulation_steps} | seq={args.max_seq_len}")
    print("=" * 62)

    config = ModelConfig(**PRESETS[args.preset])
    config.max_position_embeddings = args.max_seq_len
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = NovaMindForCausalLM(config).to(args.device)

    # 从预训练权重加载（只加载模型权重，优化器从头初始化）
    if args.from_weight != "none":
        weight_path = os.path.join(args.save_dir, f"{args.from_weight}.pth")
        if not os.path.exists(weight_path):
            print(f"[错误] 权重不存在: {weight_path}")
            sys.exit(1)
        state = torch.load(weight_path, map_location=args.device)
        model.load_state_dict(state)
        print(f"[权重] 从 {weight_path} 加载预训练权重（优化器从头初始化）")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[模型] 参数量 {n_params:.2f} M, 词表 {config.vocab_size}, "
          f"层数 {config.num_hidden_layers}, 宽度 {config.hidden_size}")

    # ========== 2. 准备数据集 ==========
    # 注意：SFTDataset 会先把全部对话分词（约 3~8 分钟），请耐心等
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

    # ========== 3. 优化器 + 混合精度 ==========
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    total_steps = len(loader) * args.epochs // args.accumulation_steps

    # ========== 4. 断点续训 ==========
    start_epoch, global_step = 0, 0
    ckp_path = os.path.join(args.checkpoint_dir, f"{args.save_weight}_resume.pth")
    if args.from_resume == 1 and os.path.exists(ckp_path):
        start_epoch, global_step = load_checkpoint(ckp_path, model, optimizer, scaler)
        print(f"[续训] 从 epoch {start_epoch}, step {global_step} 恢复")

    # ========== 5. 训练循环 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    loss_log = open(os.path.join(args.save_dir, f"{args.save_weight}_loss.txt"), "a")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        step_in_epoch = 0
        ema_loss = None
        grad_norm = 0.0
        for input_ids, labels in loader:
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            # ---- 设置当前步的学习率 ----
            lr = get_lr(global_step, total_steps, args.warmup_steps, args.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- 前向 + 反向 ----
            with torch.amp.autocast('cuda', dtype=torch.float16):
                res = model(input_ids, labels=labels)
                loss = (res.loss + res.aux_loss) / args.accumulation_steps

            scaler.scale(loss).backward()

            # ---- 每 accumulation_steps 步更新一次 ----
            if (step_in_epoch + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            step_in_epoch += 1

            # ---- 打印日志 ----
            if step_in_epoch % args.log_interval == 0:
                spent = time.time() - epoch_start
                tokens_per_sec = (step_in_epoch * args.batch_size * args.max_seq_len) / spent
                cur_loss = loss.item() * args.accumulation_steps
                cur_aux = res.aux_loss.item()
                ema_loss = cur_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * cur_loss
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                tokens_seen = step_in_epoch * args.batch_size * args.max_seq_len / 1e6
                progress = step_in_epoch / len(loader) * 100
                eta_min = spent / step_in_epoch * (len(loader) - step_in_epoch) / 60
                print(f"Epoch[{epoch + 1}/{args.epochs}] {step_in_epoch}/{len(loader)} "
                      f"({progress:4.1f}%) | loss {cur_loss:.4f} (ema {ema_loss:.4f}, aux {cur_aux:.4f}) "
                      f"| lr {lr:.2e} | 梯度 {grad_norm:.2f} | {tokens_per_sec:.0f} tok/s | {tokens_seen:.1f}M tok "
                      f"| 显存 {mem_used:.1f}/{mem_total:.0f}G | 已用 {spent / 60:.1f}min 剩 {eta_min:.1f}min")
                loss_log.write(f"{global_step} {cur_loss:.6f} {cur_aux:.6f} {ema_loss:.6f}\n")
                loss_log.flush()

            # ---- 定期保存 ----
            if step_in_epoch % args.save_interval == 0:
                weight_path = os.path.join(args.save_dir, f"{args.save_weight}.pth")
                torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, weight_path)
                save_checkpoint(ckp_path, model, optimizer, scaler, epoch, global_step)
                print(f"[保存] step {global_step} 权重已存至 {weight_path}")

        print(f"Epoch {epoch + 1} 完成, 用时 {(time.time() - epoch_start) / 60:.1f} 分钟")

    # ========== 6. 训练结束，保存最终模型 ==========
    final_path = os.path.join(args.save_dir, f"{args.save_weight}.pth")
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, final_path)
    save_checkpoint(ckp_path, model, optimizer, scaler, args.epochs - 1, global_step)
    loss_log.close()
    print(f"\n✅ SFT 完成！最终模型: {final_path}")
    print("下一步：python test_pretrained.py -i 里把 WEIGHT 改成 sft_0.3b 就能对话了")


if __name__ == "__main__":
    main()
