"""
Build data/train_all.csv from every labeled dataset.

Line text (BN-HTRd, BanglaWriting, Bongabdo) is kept in full.
Isolated-character sets are capped per class so they do not drown out lines.
BanglaLekha and Ekush are stored dark-ink on white, matching the line images.
"""

import csv
import os
import random
import zipfile
from collections import defaultdict
from xml.etree import ElementTree as ET

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
PER_CLASS = 40
SEED = 1
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def xlsx_rows(path, sheet="xl/worksheets/sheet1.xml"):
    z = zipfile.ZipFile(path)
    shared = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings = []
    for si in shared.findall(NS + "si"):
        strings.append("".join((t.text or "") for t in si.iter(NS + "t")))
    sheet_xml = ET.fromstring(z.read(sheet))
    rows = []
    for row in sheet_xml.iter(NS + "row"):
        vals = []
        for cell in row.findall(NS + "c"):
            node = cell.find(NS + "v")
            if node is None or node.text is None:
                vals.append("")
                continue
            raw = node.text
            vals.append(strings[int(raw)] if cell.get("t") == "s" else raw)
        if any(vals):
            rows.append(vals)
    return rows


def ekush_labels():
    rows = xlsx_rows(r"F:\BN-HTR\Ekush\Untitled spreadsheet.xlsx", "xl/worksheets/sheet2.xml")
    labels = {}
    for vals in rows[1:]:
        if len(vals) < 2 or not vals[0]:
            continue
        labels[str(int(float(vals[0])))] = vals[1].strip()
    return labels


def basic_and_digits():
    ek = ekush_labels()
    # BanglaLekha folders 1-50 follow Ekush 10-59 (50 basic characters).
    # Folders 51-60 are the ten digits.
    labels = {}
    for i in range(1, 51):
        labels[str(i)] = ek[str(i + 9)]
    for i, digit in enumerate("০১২৩৪৫৬৭৮৯"):
        labels[str(51 + i)] = digit
    return labels


def matrivasha_labels():
    rows = xlsx_rows(r"F:\BN-HTR\MatriVasha\compound_character.xlsx")
    labels = {}
    for vals in rows[1:]:
        if len(vals) < 2 or not vals[0] or vals[0] == "Folder Name":
            continue
        labels[str(int(float(vals[0])))] = vals[1].strip()
    return labels


def save_inverted(src, dest):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    if float(img.mean()) < 140:
        img = 255 - img
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cv2.imwrite(dest, img)
    return True


def sample_folders(root, labels, out_dir, rng, invert):
    chosen = []
    for folder, text in labels.items():
        src_dir = os.path.join(root, folder)
        if not os.path.isdir(src_dir):
            continue
        names = [n for n in os.listdir(src_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        pick = names if len(names) <= PER_CLASS else rng.sample(names, PER_CLASS)
        for name in pick:
            src = os.path.join(src_dir, name)
            dest = os.path.join(out_dir, folder, name)
            if not os.path.isfile(dest):
                if invert:
                    if not save_inverted(src, dest):
                        continue
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    cv2.imwrite(dest, img)
            if text:
                chosen.append((dest, text))
    return chosen


def main():
    rng = random.Random(SEED)
    rows = []

    with open(os.path.join(DATA, "train.csv"), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["image_path"], row["text"]))
    n_lines = len(rows)

    mv = sample_folders(
        r"F:\BN-HTR\MatriVasha\sampled",
        matrivasha_labels(),
        r"F:\BN-HTR\MatriVasha\sampled",
        rng,
        invert=False,
    )
    ek = sample_folders(
        r"F:\BN-HTR\Ekush\dataset",
        ekush_labels(),
        r"F:\BN-HTR\Ekush\sampled",
        rng,
        invert=True,
    )
    bl = sample_folders(
        r"F:\BN-HTR\BanglaLekha-Isolated\BanglaLekha-Isolated\Images",
        basic_and_digits(),
        r"F:\BN-HTR\BanglaLekha-Isolated\sampled",
        rng,
        invert=True,
    )
    rows.extend(mv)
    rows.extend(ek)
    rows.extend(bl)

    out = os.path.join(DATA, "train_all.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        writer.writerows(rows)
    print(f"lines={n_lines} matrivasha={len(mv)} ekush={len(ek)} banglalekha={len(bl)}")
    print(f"total={len(rows)} wrote {out}")


if __name__ == "__main__":
    main()
