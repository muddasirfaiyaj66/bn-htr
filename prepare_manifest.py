"""
prepare_manifest.py

Builds train/val/test manifest CSVs mapping each line-crop image -> its
ground-truth Bangla text, reconstructed from the dataset's per-word Excel
annotations.

CONFIRMED actual dataset layout (from the real BN-HTR_Dataset download,
which differs from both the paper's diagram AND the first cut of this
script):

  <dataset_root>/
    Segmentation_Images/
      Lines/
        <doc>/
          <doc>_<page>.jpg          <- FULL PAGE image (not a single line!)
          <doc>_<page>.txt          <- YOLO bounding boxes for all lines on the page
          <doc>_<page>/
            <doc>_<page>_<line>.jpg   <- an actual single cropped line image
            ...
      Words/
        <doc>/
          <doc>_<page>/
            <doc>_<page>_<line>/
              <doc>_<page>_<line>_<word>.jpg   <- a single word crop
    Recognition_Ground_Truth_Texts/
      <doc>/
        <doc>.xlsx     <- columns: Id (e.g. "1_1_1_1"), Word (Bangla text)
                          Id = <doc>_<page>_<line>_<word>

There is NO direct transcription file for a line. Instead, we reconstruct
each line's text by taking every row in <doc>.xlsx whose Id starts with
"<doc>_<page>_<line>_", sorting by the trailing word index, and joining
the words with spaces. That reconstructed text is paired with the real
line-crop image at:
    Segmentation_Images/Lines/<doc>/<doc>_<page>/<doc>_<page>_<line>.jpg

Usage:
    python prepare_manifest.py --dataset_root "F:\\...\\BN-HTR_Dataset" --out_dir F:\\BN-HTR\\data
"""

import os
import csv
import argparse
import random
from collections import defaultdict

import pandas as pd

LINES_SUBPATH = os.path.join("Segmentation_Images", "Lines")
GT_SUBPATH = "Recognition_Ground_Truth_Texts"


def reconstruct_line_texts_for_doc(xlsx_path):
    """
    Returns { "<doc>_<page>_<line>": "word1 word2 word3 ..." }
    built from every row in the doc's xlsx, grouped by the first 3
    underscore-separated parts of Id and ordered by the 4th (word index).
    """
    df = pd.read_excel(xlsx_path)
    if "Id" not in df.columns or "Word" not in df.columns:
        raise ValueError(f"Unexpected columns in {xlsx_path}: {df.columns.tolist()}")

    groups = defaultdict(list)  # line_key -> [(word_idx, word_text), ...]

    for _, row in df.iterrows():
        id_str = str(row["Id"]).strip()
        word = str(row["Word"]).strip()
        if not id_str or word == "" or word.lower() == "nan":
            continue

        parts = id_str.split("_")
        if len(parts) != 4:
            continue  # skip malformed ids rather than crash the whole run

        doc, page, line, word_idx = parts
        line_key = f"{doc}_{page}_{line}"
        try:
            idx = int(word_idx)
        except ValueError:
            idx = 0
        groups[line_key].append((idx, word))

    line_texts = {}
    for line_key, words in groups.items():
        words.sort(key=lambda t: t[0])
        line_texts[line_key] = " ".join(w for _, w in words)

    return line_texts


def collect_pairs(dataset_root):
    lines_root = os.path.join(dataset_root, LINES_SUBPATH)
    gt_root = os.path.join(dataset_root, GT_SUBPATH)

    if not os.path.isdir(lines_root):
        raise FileNotFoundError(f"Could not find {lines_root}")
    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"Could not find {gt_root}")

    pairs = []
    skipped_missing_image = 0
    doc_ids = sorted(os.listdir(gt_root), key=lambda x: (len(x), x))

    for doc in doc_ids:
        doc_gt_dir = os.path.join(gt_root, doc)
        if not os.path.isdir(doc_gt_dir):
            continue

        xlsx_path = os.path.join(doc_gt_dir, f"{doc}.xlsx")
        if not os.path.exists(xlsx_path):
            print(f"  [warn] no xlsx for doc {doc}, skipping")
            continue

        try:
            line_texts = reconstruct_line_texts_for_doc(xlsx_path)
        except Exception as e:
            print(f"  [warn] failed to read {xlsx_path}: {e}")
            continue

        for line_key, text in line_texts.items():
            doc_part, page_part, line_part = line_key.split("_")
            page_key = f"{doc_part}_{page_part}"
            img_path = os.path.join(lines_root, doc_part, page_key, f"{line_key}.jpg")

            if not os.path.exists(img_path):
                skipped_missing_image += 1
                continue

            if len(text.strip()) == 0:
                continue

            # 'doc' (the top-level numbered folder) is the writer/document id,
            # used later for a writer-disjoint split
            pairs.append((img_path, text, doc))

    if skipped_missing_image:
        print(f"  [info] {skipped_missing_image} reconstructed lines had no matching image file (skipped)")

    return pairs


def split_by_writer(pairs, val_frac=0.1, test_frac=0.1, seed=42):
    writers = sorted(set(p[2] for p in pairs))
    random.Random(seed).shuffle(writers)

    n = len(writers)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_writers = set(writers[:n_test])
    val_writers = set(writers[n_test:n_test + n_val])

    train, val, test = [], [], []
    for img_path, text, writer in pairs:
        if writer in test_writers:
            test.append((img_path, text))
        elif writer in val_writers:
            val.append((img_path, text))
        else:
            train.append((img_path, text))

    return train, val, test


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Path to the 'BN-HTR_Dataset' folder")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = collect_pairs(args.dataset_root)
    print(f"Found {len(pairs)} (image, reconstructed-line-text) pairs across "
          f"{len(set(p[2] for p in pairs))} writer folders.")

    if len(pairs) < 1000:
        print("  [warn] this looks low compared to the paper's reported 13,867 lines — "
              "double check the dataset extracted correctly before training.")

    train, val, test = split_by_writer(pairs, args.val_frac, args.test_frac)
    print(f"Split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print("Note: split is done BY WRITER FOLDER, not randomly by line, "
          "so validation/test handwriting styles are unseen during training.")

    write_csv(train, os.path.join(args.out_dir, "train.csv"))
    write_csv(val, os.path.join(args.out_dir, "val.csv"))
    write_csv(test, os.path.join(args.out_dir, "test.csv"))

    print(f"Wrote train.csv / val.csv / test.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
