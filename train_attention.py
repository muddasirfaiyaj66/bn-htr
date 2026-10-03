"""
train_attention.py

Train the attention sequence model on the combined line dataset.
Images are deskewed, illumination-flattened, and Otsu-binarized.
The CNN encoder can be initialized from a trained CRNN checkpoint.

Usage (from the bn-htr folder):
    python train_attention.py --train_csv data\\train.csv --val_csv data\\val.csv --vocab data\\vocab.json ^
        --init_crnn F:\\BN-HTR\\models\\v4\\best.pt --epochs 20 --out_dir F:\\BN-HTR\\models\\attention
"""

import argparse
import os
import time

import editdistance
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BNHTRDataset
from model_attention import AttentionHTR
from vocab import load_vocab


def collate_attention(batch, sos_idx, eos_idx):
    imgs, labels, texts = zip(*batch)
    seqs = []
    for label in labels:
        seqs.append([sos_idx] + label.tolist() + [eos_idx])
    width = max(len(seq) for seq in seqs)
    targets = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, seq in enumerate(seqs):
        targets[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return torch.stack(imgs, dim=0), targets, texts


def ids_to_text(ids, idx2char):
    chars = []
    for idx in ids:
        ch = idx2char.get(idx, "")
        if ch:
            chars.append(ch)
    return "".join(chars)


def evaluate(model, loader, idx2char, device):
    model.eval()
    total_cer_dist, total_cer_len = 0, 0
    total_wer_dist, total_wer_len = 0, 0
    with torch.no_grad():
        for imgs, _targets, texts in loader:
            imgs = imgs.to(device)
            decoded = model.greedy_decode(imgs)
            for ids, target in zip(decoded, texts):
                pred = ids_to_text(ids, idx2char)
                total_cer_dist += editdistance.eval(pred, target)
                total_cer_len += max(len(target), 1)
                total_wer_dist += editdistance.eval(pred.split(), target.split())
                total_wer_len += max(len(target.split()), 1)
    return total_cer_dist / total_cer_len, total_wer_dist / total_wer_len


def init_from_crnn(model, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    src = ckpt["model_state"]
    own = model.state_dict()
    copied = 0
    for key, value in src.items():
        target = key.replace("rnn.", "encoder.", 1) if key.startswith("rnn.") else key
        if target in own and own[target].shape == value.shape:
            own[target] = value
            copied += 1
    model.load_state_dict(own)
    print(f"Copied {copied} tensors from {checkpoint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="checkpoints_attention")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--init_crnn", default=None, help="CRNN checkpoint used to initialize the CNN and encoder")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    char2idx, idx2char = load_vocab(args.vocab)
    sos_idx = max(char2idx.values()) + 1
    eos_idx = sos_idx + 1
    num_classes = eos_idx + 1

    train_ds = BNHTRDataset(args.train_csv, char2idx, augment=True, clean=True, binarize=True)
    val_ds = BNHTRDataset(args.val_csv, char2idx, augment=False, clean=True, binarize=True)

    def _collate(batch):
        return collate_attention(batch, sos_idx, eos_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate, pin_memory=False,
    )

    model = AttentionHTR(num_classes, sos_idx, eos_idx).to(device)
    if args.init_crnn:
        init_from_crnn(model, args.init_crnn, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_cer = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Attention {epoch + 1}/{args.epochs}")
        for imgs, targets, _texts in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs, targets)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        cer, wer = evaluate(model, val_loader, idx2char, device)
        scheduler.step(cer)
        print(
            f"Epoch {epoch + 1}: train_loss={epoch_loss / len(train_loader):.4f}  "
            f"val_CER={cer:.4f}  val_WER={wer:.4f}  time={time.time() - t0:.1f}s"
        )
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_cer": best_cer,
            "char2idx": char2idx,
            "idx2char": idx2char,
            "sos_idx": sos_idx,
            "eos_idx": eos_idx,
            "num_classes": num_classes,
            "clean": True,
            "binarize": True,
            "architecture": "attention",
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if cer < best_cer:
            best_cer = cer
            ckpt["best_cer"] = best_cer
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> New best CER {best_cer:.4f}, saved best.pt")


if __name__ == "__main__":
    main()
