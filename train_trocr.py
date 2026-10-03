"""
train_trocr.py

Fine-tune a pretrained TrOCR model (ViT encoder + text decoder) on the
combined Bangla line dataset. Images are deskewed and Otsu-binarized.

Usage (from the bn-htr folder):
    python train_trocr.py --train_csv data\\train.csv --val_csv data\\val.csv ^
        --epochs 3 --batch_size 4 --out_dir F:\\BN-HTR\\models\\trocr
"""

import argparse
import os
import time

import cv2
import editdistance
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from dataset import _augment, clean_line, load_manifest


class TrocrLineDataset(Dataset):
    def __init__(self, csv_path, augment=False, clean=True, binarize=True):
        self.rows = load_manifest(csv_path)
        self.augment = augment
        self.clean = clean
        self.binarize = binarize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, text = self.rows[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if self.clean or self.binarize:
            img = clean_line(img, binarize=self.binarize)
        if self.augment:
            img = _augment(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb), text


def make_collate(processor, max_length):
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        images, texts = zip(*batch)
        pixels = processor(list(images), return_tensors="pt").pixel_values
        labels = processor.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids
        labels[labels == pad_id] = -100
        return pixels, labels, texts

    return collate


@torch.no_grad()
def evaluate(model, processor, loader, device, max_length):
    model.eval()
    total_cer_dist, total_cer_len = 0, 0
    total_wer_dist, total_wer_len = 0, 0
    seen = 0
    for pixels, _labels, texts in loader:
        if seen >= 200:
            break
        pixels = pixels.to(device)
        generated = model.generate(pixels, max_length=max_length, num_beams=1)
        preds = processor.batch_decode(generated, skip_special_tokens=True)
        for pred, target in zip(preds, texts):
            pred = " ".join(pred.split())
            total_cer_dist += editdistance.eval(pred, target)
            total_cer_len += max(len(target), 1)
            total_wer_dist += editdistance.eval(pred.split(), target.split())
            total_wer_len += max(len(target.split()), 1)
            seen += 1
    model.train()
    return total_cer_dist / max(total_cer_len, 1), total_wer_dist / max(total_wer_len, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--model_name", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--out_dir", default="checkpoints_trocr")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--encoder", default=None, help="Vision encoder (ViT, DeiT, or Swin). Pair with --decoder.")
    ap.add_argument("--decoder", default=None, help="Text decoder/tokenizer, for example a Bangla BERT or mBART")
    ap.add_argument("--lora", action="store_true", help="Train the decoder with LoRA via the optional peft package")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = ap.parse_args()
    if bool(args.encoder) != bool(args.decoder):
        ap.error("Pass both --encoder and --decoder, or neither to fine-tune the TrOCR checkpoint")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading {args.model_name}")

    if args.encoder and args.decoder:
        from transformers import AutoImageProcessor, AutoTokenizer

        image_processor = AutoImageProcessor.from_pretrained(args.encoder)
        tokenizer = AutoTokenizer.from_pretrained(args.decoder)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.decoder)

        class _PairProcessor:
            """Enough of the TrOCR processor interface for the existing collate."""

            def __init__(self, image_processor, tokenizer):
                self.image_processor = image_processor
                self.tokenizer = tokenizer

            def __call__(self, images, return_tensors="pt"):
                return self.image_processor(images, return_tensors=return_tensors)

        processor = _PairProcessor(image_processor, tokenizer)
        start_id = tokenizer.cls_token_id
        if start_id is None:
            start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    else:
        processor = TrOCRProcessor.from_pretrained(args.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
        start_id = processor.tokenizer.cls_token_id
        pad_id = processor.tokenizer.pad_token_id
        eos_id = processor.tokenizer.sep_token_id
    model.config.decoder_start_token_id = start_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id
    model.generation_config.decoder_start_token_id = start_id
    model.generation_config.pad_token_id = pad_id
    model.generation_config.eos_token_id = eos_id
    if args.lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise SystemExit("Install peft to use --lora. The default full fine-tune does not need it.") from exc
        model.decoder = get_peft_model(
            model.decoder,
            LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none", target_modules=["q_proj", "v_proj"]),
        )
        print(f"LoRA on the decoder, r={args.lora_r}")
    model.generation_config.max_length = args.max_length
    model.generation_config.num_beams = 4
    model.to(device)

    collate = make_collate(processor, args.max_length)
    train_loader = DataLoader(
        TrocrLineDataset(args.train_csv, augment=True, clean=True, binarize=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        TrocrLineDataset(args.val_csv, augment=False, clean=True, binarize=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_cer = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"TrOCR {epoch + 1}/{args.epochs}")
        for pixels, labels, _texts in pbar:
            pixels = pixels.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(pixel_values=pixels, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.save_pretrained(os.path.join(args.out_dir, "last"))
        processor.save_pretrained(os.path.join(args.out_dir, "last"))

        cer, wer = evaluate(model, processor, val_loader, device, args.max_length)
        print(
            f"Epoch {epoch + 1}: train_loss={epoch_loss / len(train_loader):.4f}  "
            f"val_CER={cer:.4f}  val_WER={wer:.4f}  time={time.time() - t0:.1f}s"
        )
        if cer < best_cer:
            best_cer = cer
            model.save_pretrained(os.path.join(args.out_dir, "best"))
            processor.save_pretrained(os.path.join(args.out_dir, "best"))
            print(f"  -> New best CER {best_cer:.4f}, saved best/")


if __name__ == "__main__":
    main()
