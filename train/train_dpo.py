"""
================================================================================
train_dpo.py —— DPO 直接偏好优化（项目第 11 个脚本）
================================================================================

【DPO 在干什么】
    不需要奖励模型！直接把"偏好对"变成训练信号：
        好回答(chosen) 的对数概率要升，坏回答(rejected) 的对数概率要降。
    用一个冻结的"参考模型"(ref) 做约束，防止模型为了讨好偏好而乱说话。

【核心公式】
    loss = -log σ( β · [ (logπ(chosen) - logπ(rejected))
                        - (logπ_ref(chosen) - logπ_ref(rejected)) ] )
    其中 logπ 是模型对 assistant 部分的对数概率（除以 assistant token 数）。
    β 控制"偏离参考模型"的强度，越大越敢偏离（但也越容易过拟合偏好）。

【和 SFT 脚本的区别】
    1. 两个模型：policy（要训练的）+ ref（冻结的参考，只算前向）
    2. 没有 labels——loss 从两个模型的 log-prob 差里来
    3. 学习率极低（1e-6）：DPO 是"微调偏好"，不是"学知识"

【教训（首次训练翻车记录）】
    第一次跑：beta=0.1、无锚定损失、lr 1e-6 → 训到后期模型只刷偏好指标，
    生成退化成"构建构建构建"复读（经典 DPO 失败模式）。
    修复：beta 降到 0.05 + SFT 锚定损失（λ=1.0）+ lr 降到 5e-7。

【怎么运行（在服务器上）】
    cd /data/NovaMind-VL
    nohup /data/miniconda/envs/torch/bin/python train/train_dpo.py > dpo.log 2>&1 &
    tail -f dpo.log
================================================================================
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from novamind import ModelConfig, PRESETS, NovaMindForCausalLM
from novamind.dataset_dpo import DPODataset


def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def compute_log_probs(model, input_ids, mask):
    """
    计算模型对 assistant 部分的平均对数概率。

    步骤：前向拿 logits → 逐 token 算 cross_entropy（不归约）
         → 只留 mask=1 的位置 → 除以 assistant token 数（每个样本自己平均）。

    返回 shape [B] 的 log-prob。
    """
    # 注意：这里不能套 no_grad！policy 需要梯度流回 logits。
    # ref 模型在调用处用 torch.no_grad() 包住。
    logits = model(input_ids).logits
    # 语言模型任务：logits[:-1] 预测 input_ids[1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # 逐 token 的负对数似然 [B, L-1]（不归约）
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_logits.shape[0], -1)

    # 只对 assistant 位置求和，除以各自的 assistant token 数
    nll = (nll * shift_mask).sum(dim=1)
    n_tokens = shift_mask.sum(dim=1).clamp(min=1)
    return -nll / n_tokens   # log-prob（平均到每个 assistant token）


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta):
    """
    标准 DPO 损失。

    logits = 隐式奖励差
        r_diff_policy = logπ(chosen) - logπ(rejected)   （policy 的偏好差）
        r_diff_ref    = logπ_ref(chosen) - logπ_ref(rejected)（ref 的偏好差）
    loss = -log σ(β · (r_diff_policy - r_diff_ref))
    """
    logits = (policy_chosen_lp - policy_rejected_lp) - (ref_chosen_lp - ref_rejected_lp)
    loss = -F.logsigmoid(beta * logits)
    return loss.mean(), logits


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


def main():
    parser = argparse.ArgumentParser(description="NovaMind DPO 直接偏好优化")
    # 数据与路径
    parser.add_argument("--data_path", type=str,
                        default="/data/NovaMind-VL/data/dpo.jsonl")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/data/NovaMind-VL/novamind/tokenizer")
    parser.add_argument("--save_dir", type=str, default="/data/NovaMind-VL/out")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/NovaMind-VL/checkpoints")
    parser.add_argument("--save_weight", type=str, default="dpo_0.3b")
    # 模型
    parser.add_argument("--preset", type=str, default="novamind-0.3b", choices=list(PRESETS.keys()))
    parser.add_argument("--from_weight", type=str, default="sft_clean_0.3b",
                        help="基座权重名（policy 和 ref 都从它初始化）")
    # DPO 超参数
    parser.add_argument("--beta", type=float, default=0.05,
                        help="DPO 偏离强度 β（首次用 0.1 训崩了：模型钻空子过拟合偏好，生成退化。0.05 更保守）")
    parser.add_argument("--anchor_lambda", type=float, default=1.0,
                        help="SFT 锚定损失系数 λ：loss = DPO + λ·NLL(chosen)。防止模型只刷偏好指标而生成退化")
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="每个 batch 里 chosen+rejected 各 batch_size 条")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="偏好对单侧最大长度（数据中位数约 350 token/侧，1024 覆盖大部分）")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    # 断点续训 / 调试
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_rows", type=int, default=0, help="调试用：只读前 N 行")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 62)
    print(f" NovaMind DPO | 基座 {args.from_weight} | beta={args.beta} | "
          f"lr={args.learning_rate} | batch={args.batch_size} | seq={args.max_seq_len}")
    print("=" * 62)

    # ========== 1. 加载基座 → policy，复制 → ref（冻结） ==========
    config = ModelConfig(**PRESETS[args.preset])
    config.max_position_embeddings = args.max_seq_len
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    model = NovaMindForCausalLM(config).to(args.device)
    weight_path = os.path.join(args.save_dir, f"{args.from_weight}.pth")
    if not os.path.exists(weight_path):
        print(f"[错误] 基座权重不存在: {weight_path}")
        sys.exit(1)
    state = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(state)
    print(f"[权重] policy 加载自 {weight_path}")

    # ref 模型：同一权重、eval 模式、不参与梯度
    ref_model = NovaMindForCausalLM(config).to(args.device)
    ref_model.load_state_dict(state)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("[模型] ref 模型已创建并冻结")

    # ========== 2. 数据集 ==========
    ds = DPODataset(
        args.data_path, tokenizer,
        max_length=args.max_seq_len,
        max_rows=(args.max_rows if args.max_rows > 0 else None),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"[数据] {len(ds)} 对偏好样本, {len(loader)} 个 batch/epoch")

    # ========== 3. 优化器 + 混合精度 ==========
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    total_steps = len(loader) * args.epochs

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
        ema_loss = None
        ema_acc = None
        for step_in_epoch, (c_ids, c_mask, r_ids, r_mask) in enumerate(loader):
            c_ids = c_ids.to(args.device, non_blocking=True)
            c_mask = c_mask.to(args.device, non_blocking=True)
            r_ids = r_ids.to(args.device, non_blocking=True)
            r_mask = r_mask.to(args.device, non_blocking=True)

            lr = get_lr(global_step, total_steps, args.warmup_steps, args.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- 参考模型对数概率（不参与梯度，fp16 前向） ----
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                ref_c_lp = compute_log_probs(ref_model, c_ids, c_mask)
                ref_r_lp = compute_log_probs(ref_model, r_ids, r_mask)

            # ---- policy 对数概率（要梯度） ----
            with torch.amp.autocast('cuda', dtype=torch.float16):
                policy_c_lp = compute_log_probs(model, c_ids, c_mask)
                policy_r_lp = compute_log_probs(model, r_ids, r_mask)
                dpo, logits = dpo_loss(policy_c_lp, policy_r_lp,
                                       ref_c_lp, ref_r_lp, args.beta)

                # SFT 锚定损失：让模型继续"学好回答"，防止为了刷偏好而生成退化。
                # NLL(chosen) 越小 = 模型越会生成好回答。
                if args.anchor_lambda > 0:
                    nll_chosen = -policy_c_lp  # log-prob 的负数就是 NLL
                    loss = dpo + args.anchor_lambda * nll_chosen.mean()
                else:
                    loss = dpo

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_in_epoch += 1

            # ---- 打印日志 ----
            if step_in_epoch % args.log_interval == 0:
                spent = time.time() - epoch_start
                # 偏好准确率：logits>0 说明 policy 把 chosen 相对 rejected 抬高了
                acc = (logits > 0).float().mean().item()
                cur_loss = loss.item()
                cur_dpo = dpo.item() if isinstance(dpo, torch.Tensor) else dpo
                ema_loss = cur_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * cur_loss
                ema_acc = acc if ema_acc is None else 0.9 * ema_acc + 0.1 * acc
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                progress = step_in_epoch / len(loader) * 100
                eta_min = spent / step_in_epoch * (len(loader) - step_in_epoch) / 60
                print(f"Epoch[{epoch + 1}/{args.epochs}] {step_in_epoch}/{len(loader)} "
                      f"({progress:4.1f}%) | loss {cur_loss:.4f} (ema {ema_loss:.4f}, dpo {cur_dpo:.4f}) "
                      f"| acc {acc:.3f} (ema {ema_acc:.3f}) | lr {lr:.2e} "
                      f"| 显存 {mem_used:.1f}/{mem_total:.0f}G | 已用 {spent / 60:.1f}min 剩 {eta_min:.1f}min")
                loss_log.write(f"{global_step} {cur_loss:.6f} {cur_dpo:.6f} {ema_loss:.6f} {acc:.4f}\n")
                loss_log.flush()

            # ---- 定期保存 ----
            if step_in_epoch % args.save_interval == 0:
                weight_path = os.path.join(args.save_dir, f"{args.save_weight}.pth")
                torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, weight_path)
                save_checkpoint(ckp_path, model, optimizer, scaler, epoch, global_step)
                print(f"[保存] step {global_step} 权重已存至 {weight_path}")

        print(f"Epoch {epoch + 1} 完成, 用时 {(time.time() - epoch_start) / 60:.1f} 分钟")

    # ========== 6. 保存最终模型 ==========
    final_path = os.path.join(args.save_dir, f"{args.save_weight}.pth")
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, final_path)
    save_checkpoint(ckp_path, model, optimizer, scaler, args.epochs - 1, global_step)
    loss_log.close()
    print(f"\n✅ DPO 完成！最终模型: {final_path}")
    print("测试：python test_pretrained.py -i --weight dpo_0.3b")


if __name__ == "__main__":
    main()
