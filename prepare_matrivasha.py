"""
Sample MatriVasha compound-character images and append them to the line
training manifest. The full zips stay in F:\\BN-HTR\\MatriVasha. Validation
stays the existing line set so CER stays comparable.
"""

import csv
import os
import random
import zipfile
from collections import defaultdict
from xml.etree import ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SRC = r"F:\BN-HTR\MatriVasha"
OUT_IMG = os.path.join(SRC, "sampled")
XLSX = os.path.join(SRC, "compound_character.xlsx")
PER_CLASS_PER_ZIP = 50
SEED = 0
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def class_map(path):
    z = zipfile.ZipFile(path)
    shared = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings = []
    for si in shared.findall(NS + "si"):
        strings.append("".join(t.text or "" for t in si.iter(NS + "t")))
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    mapping = {}
    for row in sheet.iter(NS + "row"):
        vals = []
        for cell in row.findall(NS + "c"):
            node = cell.find(NS + "v")
            if node is None:
                vals.append("")
                continue
            raw = node.text or ""
            vals.append(strings[int(raw)] if cell.get("t") == "s" else raw)
        if len(vals) < 2 or not vals[0] or vals[0] == "Folder Name":
            continue
        mapping[str(int(float(vals[0])))] = vals[1].strip()
    return mapping


def sample_zip(zip_path, labels, rng):
    chosen = []
    with zipfile.ZipFile(zip_path) as zf:
        groups = defaultdict(list)
        for name in zf.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            folder = parts[-2]
            if folder in labels:
                groups[folder].append(name)
        for folder, names in groups.items():
            pick = names if len(names) <= PER_CLASS_PER_ZIP else rng.sample(names, PER_CLASS_PER_ZIP)
            text = labels[folder]
            for name in pick:
                dest = os.path.join(OUT_IMG, folder, os.path.basename(name))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.isfile(dest):
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                chosen.append((dest, text))
    return chosen


def main():
    rng = random.Random(SEED)
    labels = class_map(XLSX)
    rows = []
    for zip_name in ("female.zip", "male.zip"):
        rows.extend(sample_zip(os.path.join(SRC, zip_name), labels, rng))
    print(f"classes={len(labels)} sampled={len(rows)}")

    out_csv = os.path.join(DATA_DIR, "train_v6.csv")
    base = os.path.join(DATA_DIR, "train.csv")
    with open(base, "r", encoding="utf-8", newline="") as src, open(
        out_csv, "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=["image_path", "text"])
        writer.writeheader()
        n_base = 0
        for row in reader:
            writer.writerow({"image_path": row["image_path"], "text": row["text"]})
            n_base += 1
        for path, text in rows:
            writer.writerow({"image_path": path, "text": text})
    print(f"lines={n_base} conjuncts={len(rows)} wrote {out_csv}")


if __name__ == "__main__":
    main()
