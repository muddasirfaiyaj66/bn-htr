"""
prepare_combined.py

Build train/val/test CSVs for the CRNN from three line-level sources:

  1. BN-HTRd — existing line crops and Excel word labels
  2. BanglaWriting — word boxes grouped into lines and cropped
  3. Bongabdo — page photos segmented into lines, then aligned to the
     page transcript with the current recognizer

Each dataset is split by writer (or by page, where each page is one writer)
so validation handwriting is unseen. Splits are merged into one manifest.

Usage (from the bn-htr folder):
    python prepare_combined.py --checkpoint F:\\BN-HTR\\models\\v3\\best.pt --out_dir data
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import editdistance
import numpy as np
import torch
from tqdm import tqdm

from dataset import preprocess_array
from infer import ctc_greedy_decode_single
from model import CRNN
from prepare_manifest import collect_pairs, split_by_writer

PUNCT = set(".,;:!?।॥'\"“”‘’`-–—()[]{}…")
MAX_LABEL_LEN = 160


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    for ch in ("\u200c", "\u200d", "\ufeff", "\u00a0"):
        text = text.replace(ch, " " if ch == "\u00a0" else "")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split())


def _is_punct_only(label):
    stripped = label.strip()
    if not stripped:
        return True
    return all((not ch.isalnum()) and not ("\u0980" <= ch <= "\u09FF") for ch in stripped)


def _join_tokens(labels):
    out = ""
    for lab in labels:
        lab = clean_text(lab)
        if not lab:
            continue
        if out and _is_punct_only(lab):
            out += lab
        elif out:
            out += " " + lab
        else:
            out = lab
    return clean_text(out)


def group_words_into_lines(words):
    """Cluster word boxes into horizontal lines. Each word is a dict with x0,y0,x1,y1,label."""
    words = sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2.0, w["x0"]))
    lines = []
    for w in words:
        w_cy = (w["y0"] + w["y1"]) / 2.0
        w_h = max(1.0, w["y1"] - w["y0"])
        if lines:
            line = lines[-1]
            centers = [(item["y0"] + item["y1"]) / 2.0 for item in line]
            heights = [max(1.0, item["y1"] - item["y0"]) for item in line]
            line_cy = float(np.median(centers))
            line_h = float(np.median(heights))
            if abs(w_cy - line_cy) <= 0.55 * max(line_h, w_h):
                line.append(w)
                continue
        lines.append([w])

    merged = []
    for line in lines:
        labels = [item["label"] for item in line]
        if merged and all(_is_punct_only(lab) for lab in labels):
            merged[-1].extend(line)
        else:
            merged.append(line)

    merged.sort(key=lambda line: float(np.mean([(w["y0"] + w["y1"]) / 2.0 for w in line])))
    for line in merged:
        line.sort(key=lambda w: w["x0"])
    return merged


def crop_line(img, line, pad=6):
    x0 = max(0, int(min(w["x0"] for w in line)) - pad)
    y0 = max(0, int(min(w["y0"] for w in line)) - pad)
    x1 = min(img.shape[1], int(max(w["x1"] for w in line)) + pad)
    y1 = min(img.shape[0], int(max(w["y1"] for w in line)) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img[y0:y1, x0:x1]


def collect_banglawriting(root, out_dir):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    jsons = sorted(root.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No JSON annotations in {root}")

    for jp in tqdm(jsons, desc="BanglaWriting"):
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_name = data.get("imagePath") or (jp.stem + ".jpg")
        img_path = root / image_name
        if not img_path.is_file():
            alt = jp.with_suffix(".jpg")
            img_path = alt if alt.is_file() else img_path
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [warn] missing image for {jp.name}")
            continue

        jw = float(data.get("imageWidth") or img.shape[1])
        jh = float(data.get("imageHeight") or img.shape[0])
        sx = img.shape[1] / jw if jw else 1.0
        sy = img.shape[0] / jh if jh else 1.0

        words = []
        for shape in data.get("shapes", []):
            label = clean_text(shape.get("label", ""))
            pts = shape.get("points") or []
            if not label or len(pts) < 2:
                continue
            xs = [p[0] * sx for p in pts]
            ys = [p[1] * sy for p in pts]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            words.append({"label": label, "x0": x0, "y0": y0, "x1": x1, "y1": y1})

        if not words:
            continue

        writer = f"bw:{jp.stem}"
        for i, line in enumerate(group_words_into_lines(words)):
            text = _join_tokens(w["label"] for w in line)
            if len(text) < 2 or len(text) > MAX_LABEL_LEN or _is_punct_only(text):
                continue
            crop = crop_line(img, line)
            if crop is None:
                continue
            dest = out_dir / f"{jp.stem}_L{i:03d}.jpg"
            cv2.imwrite(str(dest), crop)
            pairs.append((str(dest), text, writer))

    return pairs


def _load_bongabdo_meta(root):
    meta_path = Path(root) / "Bongabdo_Metadata.csv"
    writers = {}
    if not meta_path.is_file():
        return writers
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Filename") or "").strip()
            user = (row.get("Username") or name).strip()
            if name:
                writers[name] = user or name
    return writers


def _snap_to_space(text, pos, window=18):
    pos = max(0, min(len(text), pos))
    if pos == 0 or pos == len(text):
        return pos
    for d in range(0, window + 1):
        for p in (pos - d, pos + d):
            if 0 < p < len(text) and text[p - 1] == " ":
                return p
    return pos


def align_predictions_to_transcript(preds, gt):
    """
    Split `gt` into one string per prediction using edit alignment.
    Returns a list of strings the same length as `preds`, or None.
    """
    preds = [clean_text(p) for p in preds]
    gt = clean_text(gt)
    if not gt or not preds or not any(preds):
        return None

    sep = "\x01"
    pred = sep.join(preds)
    n, m = len(pred), len(gt)
    if n == 0 or m == 0 or n > 4000 or m > 4000:
        return None

    back = np.zeros((n + 1, m + 1), dtype=np.uint8)
    prev = np.arange(m + 1, dtype=np.int32)
    back[0, 1:] = 2
    gt_chars = gt

    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = i
        back[i, 0] = 1
        pc = pred[i - 1]
        sep_char = pc == sep
        for j in range(1, m + 1):
            if sep_char:
                sub = prev[j - 1] + 2
            else:
                sub = prev[j - 1] + (0 if pc == gt_chars[j - 1] else 1)
            delete = prev[j] + 1
            insert = cur[j - 1] + 1
            if sub <= delete and sub <= insert:
                cur[j] = sub
                back[i, j] = 0
            elif delete <= insert:
                cur[j] = delete
                back[i, j] = 1
            else:
                cur[j] = insert
                back[i, j] = 2
        prev = cur

    gt_at_pred = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        move = int(back[i, j])
        if move == 0:
            gt_at_pred[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1

    anchors = []
    cursor = 0
    for line_i, piece in enumerate(preds):
        end = cursor + len(piece)
        for k in range(cursor, end):
            g = gt_at_pred[k]
            if g is not None:
                anchors.append((g, line_i))
        cursor = end + 1
    if len(anchors) < max(4, len(preds)):
        return None
    anchors.sort()

    assign = np.zeros(m, dtype=np.int32)
    ai = 0
    for g in range(m):
        while ai + 1 < len(anchors) and anchors[ai + 1][0] <= g:
            ai += 1
        choice = anchors[ai]
        if ai + 1 < len(anchors):
            nxt = anchors[ai + 1]
            if abs(nxt[0] - g) < abs(choice[0] - g):
                choice = nxt
        assign[g] = choice[1]
    for g in range(1, m):
        if assign[g] < assign[g - 1]:
            assign[g] = assign[g - 1]

    texts = []
    start = 0
    for line_i in range(len(preds) - 1):
        end = start
        while end < m and int(assign[end]) <= line_i:
            end += 1
        end = _snap_to_space(gt, end)
        end = max(start, min(m, end))
        texts.append(gt[start:end].strip())
        start = end
    texts.append(gt[start:].strip())
    if len(texts) != len(preds):
        return None
    return texts


def _page_cer(preds, gt):
    hyp = clean_text(" ".join(preds))
    ref = clean_text(gt)
    if not ref:
        return 1.0
    return editdistance.eval(hyp, ref) / len(ref)


def _recognize_crops(model, crops, idx2char, device):
    if not crops:
        return []
    batch = np.stack([preprocess_array(c, augment=False) for c in crops], axis=0)
    tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(tensor)
    preds = []
    for row in logits:
        preds.append(ctc_greedy_decode_single(row, idx2char))
    return preds


def _load_recognizer(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    char2idx = ckpt["char2idx"]
    idx2char = {int(k): v for k, v in ckpt["idx2char"].items()}
    model = CRNN(num_classes=len(char2idx) + 1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx2char


def collect_bongabdo(root, out_dir, checkpoint, cer_max, device):
    from segment_lines import segment_page

    root = Path(root)
    img_dir = root / "Images"
    ann_dir = root / "Annotations"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = _load_bongabdo_meta(root)

    model, idx2char = _load_recognizer(checkpoint, device)
    pairs = []
    skipped = {"segment": 0, "align": 0, "cer": 0, "read": 0}
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}")

    for img_path in tqdm(images, desc="Bongabdo"):
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if not ann_path.is_file():
            skipped["read"] += 1
            continue
        gt = clean_text(ann_path.read_text(encoding="utf-8"))
        if len(gt) < 20:
            skipped["read"] += 1
            continue

        try:
            _, crops, _method = segment_page(str(img_path), force_classical=True)
        except Exception as exc:
            print(f"  [warn] segment failed {img_path.name}: {exc}")
            skipped["segment"] += 1
            continue

        crops = [c for c in crops if c is not None and c.size > 0 and c.shape[0] >= 12 and c.shape[1] >= 40]
        expected = max(3, int(round(len(gt) / 45)))
        if len(crops) < 3 or not (0.45 * expected <= len(crops) <= 2.2 * expected):
            skipped["segment"] += 1
            continue

        preds = _recognize_crops(model, crops, idx2char, device)
        if _page_cer(preds, gt) > cer_max:
            skipped["cer"] += 1
            continue

        texts = align_predictions_to_transcript(preds, gt)
        if not texts:
            skipped["align"] += 1
            continue

        writer = "bong:" + writers.get(img_path.stem, img_path.stem)
        kept = 0
        for i, (crop, text) in enumerate(zip(crops, texts)):
            text = clean_text(text)
            if len(text) < 2 or len(text) > MAX_LABEL_LEN:
                continue
            dest = out_dir / f"{img_path.stem}_L{i:03d}.jpg"
            cv2.imwrite(str(dest), crop)
            pairs.append((str(dest), text, writer))
            kept += 1
        if kept == 0:
            skipped["align"] += 1

    print(
        "Bongabdo skipped — "
        f"segment: {skipped['segment']}, weak page match: {skipped['cer']}, "
        f"alignment: {skipped['align']}, unreadable: {skipped['read']}"
    )
    return pairs


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        writer.writerows(rows)


def _merge_splits(named_pairs, val_frac, test_frac):
    train, val, test = [], [], []
    summary = {}
    for name, pairs in named_pairs:
        tr, va, te = split_by_writer(pairs, val_frac=val_frac, test_frac=test_frac)
        train.extend(tr)
        val.extend(va)
        test.extend(te)
        summary[name] = {
            "lines": len(pairs),
            "writers": len(set(p[2] for p in pairs)),
            "train": len(tr),
            "val": len(va),
            "test": len(te),
        }
        print(
            f"{name}: {len(pairs)} lines, {summary[name]['writers']} writers "
            f"-> train {len(tr)}, val {len(va)}, test {len(te)}"
        )
    return train, val, test, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bnhrd_root",
        default=r"F:\BN-HTR\BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)\BN-HTR_Dataset",
    )
    ap.add_argument("--bongabdo_root", default=r"F:\BN-HTR\Bongabdo")
    ap.add_argument("--banglawriting_root", default=r"F:\BN-HTR\BanglaWriting\converted")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--checkpoint", default=r"F:\BN-HTR\models\v3\best.pt")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--cer_max", type=float, default=0.55, help="Drop a Bongabdo page when recognizer CER is above this")
    ap.add_argument("--skip_bnhrd", action="store_true")
    ap.add_argument("--skip_banglawriting", action="store_true")
    ap.add_argument("--skip_bongabdo", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    named = []
    if not args.skip_bnhrd:
        print("Reading BN-HTRd line labels...")
        bn_pairs = collect_pairs(args.bnhrd_root)
        bn_pairs = [
            (img, clean_text(text), f"bnhrd:{writer}")
            for img, text, writer in bn_pairs
            if 2 <= len(clean_text(text)) <= MAX_LABEL_LEN
        ]
        named.append(("BN-HTRd", bn_pairs))

    if not args.skip_banglawriting:
        bw_pairs = collect_banglawriting(args.banglawriting_root, out_dir / "lines" / "banglawriting")
        named.append(("BanglaWriting", bw_pairs))

    if not args.skip_bongabdo:
        if not os.path.isfile(args.checkpoint):
            print(f"Missing checkpoint {args.checkpoint}; cannot align Bongabdo transcripts.", file=sys.stderr)
            sys.exit(1)
        bong_pairs = collect_bongabdo(
            args.bongabdo_root,
            out_dir / "lines" / "bongabdo",
            args.checkpoint,
            args.cer_max,
            device,
        )
        named.append(("Bongabdo", bong_pairs))

    if not named:
        print("Nothing to prepare.", file=sys.stderr)
        sys.exit(1)

    train, val, test, summary = _merge_splits(named, args.val_frac, args.test_frac)
    write_csv(train, out_dir / "train.csv")
    write_csv(val, out_dir / "val.csv")
    write_csv(test, out_dir / "test.csv")
    with open(out_dir / "prepare_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Combined -> train {len(train)}, val {len(val)}, test {len(test)}")
    print(f"Wrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
