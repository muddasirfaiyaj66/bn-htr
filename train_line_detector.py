"""
train_line_detector.py

Train YOLOv8 for page → line detection on BN-HTRd.

Usage:
    python prepare_yolo_dataset.py --dataset_root path\\to\\BN-HTR_Dataset --out_dir data\\yolo_lines
    python train_line_detector.py --data data\\yolo_lines\\data.yaml --epochs 40
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo_lines/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="Base YOLO weights")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="bn_htr_lines")
    ap.add_argument(
        "--export_to",
        default="checkpoints/line_detector.pt",
        help="Copy best.pt here for inference",
    )
    args = ap.parse_args()

    if not Path(args.data).is_file():
        raise FileNotFoundError(
            f"{args.data} missing. Run prepare_yolo_dataset.py first."
        )

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=12,
        workers=4,
        degrees=2.0,
        shear=1.0,
        perspective=0.0005,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    export = Path(args.export_to)
    export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export)
    print(f"Exported line detector → {export.resolve()}")


if __name__ == "__main__":
    main()
