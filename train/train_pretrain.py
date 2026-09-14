"""
================================================================================
train_pretrain.py —— 预训练脚本（项目第 5 个脚本，跑通它就能开始预训练）
================================================================================

【这个脚本干什么】
    把「分词器 + 模型 + 数据集」串起来，执行预训练（词语接龙）：
        读数据 → 前向算 loss → 反向传播 → 更新参数 → 定期保存

【训练循环的核心流程（一句话）】
    for 每个 batch:
        前向:   res = model(input_ids, labels=labels)   # 算 loss
        反向:   loss.backward()                          # 算梯度
        更新:   optimizer.step()                         # 用梯度更新参数

【本脚本的关键技巧】
    1. fp16 混合精度（AMP）：V100 上 fp16 比 fp32 快 6 倍，用 GradScaler 防精度下溢
    2. 梯度累积：显存不够大 batch 时，累加多个小 batch 的梯度再一起更新（等效大 batch）
    3. warmup + cosine 学习率：先线性升到峰值，再余弦衰减（训练更稳）
    4. 断点续训：保存 model+optimizer+步数，中断后能接着训

【怎么运行（在服务器上）】
    cd /data/NovaMind-VL
    nohup /data/miniconda/envs/torch/bin/python train/train_pretrain.py > pretrain.log 2>&1 &
    tail -f pretrain.log    # 看进度
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
from novamind.dataset import PretrainDataset


# ============================================================================ #
#  学习率调度：warmup + cosine
# ============================================================================ #
def get_lr(step, total_steps, warmup_steps, base_lr):
    """
    学习率曲线：先线性升到 base_lr（warmup），再余弦衰减到 base_lr 的 10%。

    例子（base_lr=3e-4, warmup=100, total=1000）：
        step=0   → lr≈0          (从 0 开始，避免一开始梯度爆炸)
        step=100 → lr=3e-4       (warmup 结束，到达峰值)
        step=550 → lr≈1.5e-4     (cosine 中段)
        step=1000→ lr=3e-5       (结束，降到 10%)
    """
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)          # warmup 线性升
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine 衰减


# ============================================================================ #
#  断点续训：保存 / 加载
# ============================================================================ #
def save_checkpoint(path, model, optimizer, scaler, epoch, step):
    """保存训练状态（模型 + 优化器 + 进度），中断后能接着训。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)


def load_checkpoint(path, model, optimizer, scaler):
    """加载训练状态，返回 (epoch, step)。"""
    ckp = torch.load(path, map_location="cpu")
    model.load_state_dict(ckp["model"])
    optimizer.load_state_dict(ckp["optimizer"])
    scaler.load_state_dict(ckp["scaler"])
    return ckp["epoch"], ckp["step"]


# ============================================================================ #
#  主训练
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="NovaMind 预训练")
    # 数据与路径
    parser.add_argument("--data_path", type=str,
                        default="/data/NovaMind-VL/data/pretrain_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/tokenizer")
    parser.add_argument("--save_dir", type=str, default="/data/NovaMind-VL/out")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/NovaMind-VL/checkpoints")
    parser.add_argument("--save_weight", type=str, default="pretrain_0.3b")
    # 模型
    parser.add_argument("--preset", type=str, default="novamind-0.3b", choices=list(PRESETS.keys()))
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="梯度累积步数（等效 batch = batch_size * accumulation_steps）")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    # 断点续训 / 调试
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--from_weight", type=str, default="none",
                        help="从旧权重继续预训练（只加载模型权重，优化器从头，适合换数据继续训）")
    parser.add_argument("--max_rows", type=int, default=0,
                        help="调试用：只读前 N 行数据（0=全量）")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 让 print 实时输出（否则 nohup 重定向到文件时会有缓冲，tail -f 看不到实时进度）
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # ========== 1. 准备配置 / 分词器 / 模型 ==========
    print("=" * 62)
    print(f" NovaMind 预训练 | 预设 {args.preset} | batch={args.batch_size} "
          f"x{args.accumulation_steps} | seq={args.max_seq_len}")
    print("=" * 62)

    config = ModelConfig(**PRESETS[args.preset])
    config.max_position_embeddings = args.max_seq_len   # 允许用参数覆盖上下文长度
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = NovaMindForCausalLM(config).to(args.device)

    # 从旧权重继续预训练（只加载模型权重，优化器从头初始化，适合"换数据继续训"）
    if args.from_weight != "none":
        weight_path = os.path.join(args.save_dir, f"{args.from_weight}.pth")
        state = torch.load(weight_path, map_location=args.device)
        model.load_state_dict(state)
        print(f"[权重] 从 {weight_path} 继续预训练（优化器从头初始化）")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[模型] 参数量 {n_params:.2f} M, 词表 {config.vocab_size}, "
          f"层数 {config.num_hidden_layers}, 宽度 {config.hidden_size}")

    # ========== 2. 准备数据集 ==========
    # 注意：PretrainDataset 会先把全部文本分词（约 2~5 分钟），请耐心等
    ds = PretrainDataset(
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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=True)   # fp16 梯度缩放

    # 实际更新步数（梯度累积后）= 总 batch 数 * epochs / accumulation
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
        ema_loss = None   # loss 的移动平均（更平滑，方便看趋势）
        grad_norm = 0.0   # 最近一次的梯度范数（判断训练稳不稳定）
        for input_ids, labels in loader:
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            # ---- 设置当前步的学习率（warmup + cosine）----
            lr = get_lr(global_step, total_steps, args.warmup_steps, args.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- 前向 + 反向（fp16 autocast 加速）----
            with torch.amp.autocast('cuda', dtype=torch.float16):
                res = model(input_ids, labels=labels)
                # loss 除以累积步数，这样累加多个小 batch 的梯度 = 一个大 batch
                loss = (res.loss + res.aux_loss) / args.accumulation_steps

            scaler.scale(loss).backward()

            # ---- 每 accumulation_steps 步更新一次参数 ----
            if (step_in_epoch + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()  # 梯度裁剪，返回裁剪前范数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            step_in_epoch += 1

            # ---- 打印日志 ----
            if step_in_epoch % args.log_interval == 0:
                spent = time.time() - epoch_start
                tokens_per_sec = (step_in_epoch * args.batch_size * args.max_seq_len) / spent
                cur_loss = loss.item() * args.accumulation_steps   # 还原真实 loss
                cur_aux = res.aux_loss.item()
                # loss 移动平均：新值权重 0.1，旧值权重 0.9，曲线更平滑
                ema_loss = cur_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * cur_loss
                # 显存占用（当前/总量）
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                # 已训 token 量（M）、进度百分比、剩余时间（分钟）
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
    print(f"\n✅ 预训练完成！最终模型: {final_path}")


if __name__ == "__main__":
    main()
