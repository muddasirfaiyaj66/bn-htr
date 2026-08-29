"""
train.py

Trains the CRNN on BN-HTRd line images using CTC loss.

Usage (from F:\\BN-HTR):
    python train.py --train_csv data\\train.csv --val_csv data\\val.csv --vocab vocab.json ^
        --epochs 60 --batch_size 32 --lr 1e-3 --out_dir checkpoints
"""

import argparse
import os
import time

import editdistance
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BNHTRDataset, collate_fn
from model import CRNN
from vocab import load_vocab


def ctc_greedy_decode(logits, idx2char):
    # logits: (B, T, C)
    preds = logits.argmax(dim=2)  # (B, T)
    results = []
    for seq in preds.tolist():
        prev = -1
        chars = []
        for idx in seq:
            if idx != prev and idx != 0:  # 0 = blank
                chars.append(idx2char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results


def evaluate(model, loader, idx2char, device):
    model.eval()
    total_cer_dist, total_cer_len = 0, 0
    total_wer_dist, total_wer_len = 0, 0
    with torch.no_grad():
        for imgs, labels_concat, label_lengths, texts in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = ctc_greedy_decode(logits, idx2char)
            for pred, target in zip(preds, texts):
                total_cer_dist += editdistance.eval(pred, target)
                total_cer_len += max(len(target), 1)
                pred_words = pred.split()
                target_words = target.split()
                total_wer_dist += editdistance.eval(pred_words, target_words)
                total_wer_len += max(len(target_words), 1)
    cer = total_cer_dist / total_cer_len
    wer = total_wer_dist / total_wer_len
    return cer, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    char2idx, idx2char = load_vocab(args.vocab)
    num_classes = len(char2idx) + 1  # +1 for CTC blank

    train_ds = BNHTRDataset(args.train_csv, char2idx, augment=True)
    val_ds = BNHTRDataset(args.val_csv, char2idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = CRNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 0
    best_cer = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels_concat, label_lengths, texts in pbar:
            imgs = imgs.to(device)
            labels_concat = labels_concat.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(imgs)  # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C) for CTCLoss
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device)

            loss = criterion(log_probs, labels_concat, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        cer, wer = evaluate(model, val_loader, idx2char, device)
        scheduler.step(cer)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}  val_CER={cer:.4f}  val_WER={wer:.4f}  time={elapsed:.1f}s")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_cer": best_cer,
            "char2idx": char2idx,
            "idx2char": idx2char,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))

        if cer < best_cer:
            best_cer = cer
            ckpt["best_cer"] = best_cer
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> New best CER {best_cer:.4f}, saved best.pt")


if __name__ == "__main__":
    main()
