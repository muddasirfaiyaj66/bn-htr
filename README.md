# BN-HTR

Offline Bangla handwritten text recognition (HTR) for full-page and single-line images.

Built on the [BN-HTRd](https://data.mendeley.com/datasets/743k6dm543) dataset. The pipeline detects text lines, recognizes each line with a CRNN+CTC model, and returns Bangla text.

**Live demo:** [huggingface.co/spaces/muddasir-faiyaj/BN-HTR](https://huggingface.co/spaces/muddasir-faiyaj/BN-HTR)

## Features

- End-to-end recognition from a page or line image
- YOLOv8 line detector (with classical CV fallback)
- CRNN + CTC character recognition
- Flask web UI and REST API
- Gradio app for Hugging Face Spaces
- Training and evaluation scripts

## Pipeline

```
Input image
    │
    ├─ single-line heuristic match ──► CRNN + CTC
    │
    └─ otherwise
            │
            ▼
      Line detection (YOLO, else projection/OPTICS)
            │
            ▼
      Per-line CRNN + CTC
            │
            ▼
      Joined Bangla text
```

The recognizer is trained on single-line crops. Full pages are segmented into lines before recognition.

## Architecture

| Stage | Method | Notes |
|-------|--------|--------|
| Line detection | YOLOv8n fine-tuned on BN-HTRd boxes | Falls back to ink projection + OPTICS |
| Recognition | CRNN (CNN + BiLSTM) + CTC | Character-level vocabulary |
| Decoding | Greedy CTC | Beam search / LM optional later |

**CRNN input:** grayscale, height 64px, width padded (800 train / 1280 infer).

## Dataset

[BN-HTRd](https://data.mendeley.com/datasets/743k6dm543) — document-level offline Bangla HTR and line segmentation.

| | |
|--|--|
| Pages | 788 |
| Writers | 150 |
| Lines | 13,867 |
| Words | 108,147 |

Dataset layout (as shipped on Mendeley):

```
BN-HTR_Dataset/
├── Segmentation_Images/
│   ├── Lines/<doc>/
│   │   ├── <doc>_<page>.jpg      # page image
│   │   ├── <doc>_<page>.txt      # YOLO line boxes
│   │   └── <doc>_<page>/         # cropped line images
│   └── Words/...
└── Recognition_Ground_Truth_Texts/
    └── <doc>/
        ├── <doc>.xlsx            # word-level labels
        └── <doc>.txt
```

Line transcripts are rebuilt from `<doc>.xlsx` by `prepare_manifest.py` (group by line id, join words).

## Repository layout

```
bn-htr-project/
├── prepare_manifest.py       # CRNN train/val/test CSVs
├── prepare_yolo_dataset.py   # YOLO detection dataset
├── train_line_detector.py    # train YOLOv8 line detector
├── vocab.py
├── dataset.py
├── model.py                  # CRNN
├── train.py                  # CRNN training
├── infer.py                  # single-line CLI
├── evaluate.py
├── segment_lines.py          # page → line crops
├── recognize.py              # recognition API used by apps
├── recognize_page.py         # end-to-end CLI
├── app.py                    # Flask UI + REST
├── gradio_app.py
├── templates/index.html
├── requirements.txt
└── README.md
```

## Setup

### 1. Install

```bash
git clone https://github.com/<your-username>/bn-htr-project.git
cd bn-htr-project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify CUDA if available:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### 2. Download the dataset

Download `BN-HTR_Dataset.zip` from [Mendeley](https://data.mendeley.com/datasets/743k6dm543) and extract it (example path: `data/BN-HTR_Dataset/`).

### 3. Build manifests and vocabulary

```bash
python prepare_manifest.py --dataset_root data/BN-HTR_Dataset --out_dir data
python vocab.py --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv --out vocab.json
```

Splits are writer-disjoint.

### 4. Train the recognizer

```bash
python train.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --vocab vocab.json \
  --epochs 60 \
  --batch_size 32 \
  --lr 1e-3 \
  --out_dir checkpoints
```

- Writes `checkpoints/last.pt` each epoch and `checkpoints/best.pt` on best validation CER
- Resume with `--resume checkpoints/last.pt`

### 5. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --test_csv data/test.csv
```

Reports CER (character error rate), WER (word error rate), and sample predictions.

### 6. Train the line detector

```bash
python prepare_yolo_dataset.py --dataset_root data/BN-HTR_Dataset --out_dir data/yolo_lines
python train_line_detector.py --data data/yolo_lines/data.yaml --epochs 40
```

Exports `checkpoints/line_detector.pt`. Without this file, page segmentation uses classical CV only.

## Inference

### CLI

```bash
# Page or line image
python recognize_page.py --checkpoint checkpoints/best.pt --image path/to/image.jpg

# Single line only
python infer.py --checkpoint checkpoints/best.pt --image path/to/line.jpg
```

Optional: `--save_lines_dir out/` to save detected line crops.

### Python

```python
import torch
from model import CRNN
from recognize import Recognizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)

model = CRNN(num_classes=len(ckpt["char2idx"]) + 1).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

rec = Recognizer(model, ckpt["idx2char"], device)
result = rec.recognize("path/to/image.jpg")
print(result["full_text"])
```

### REST API / web UI

```bash
python app.py --checkpoint checkpoints/best.pt --port 5000
```

Open `http://localhost:5000/`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/recognize` | Auto page/line recognition |
| `POST` | `/recognize_page` | Force page pipeline |
| `POST` | `/recognize_line` | Force single-line recognition |
| `GET` | `/health` | Status |

```bash
curl -X POST http://localhost:5000/recognize -F "image=@page.jpg"
```

Example response:

```json
{
  "lines": ["...", "..."],
  "full_text": "...\n...",
  "line_count": 2,
  "mode": "page",
  "segmenter": "yolo"
}
```

Environment variables: `CHECKPOINT`, `LINE_DETECTOR`, `HOST`, `PORT`.

### Gradio (local)

```bash
python gradio_app.py
```

## Deploy on Hugging Face Spaces

Public Space: [https://huggingface.co/spaces/muddasir-faiyaj/BN-HTR](https://huggingface.co/spaces/muddasir-faiyaj/BN-HTR)

Create a new Space with **SDK: Gradio**, then upload the files below into the Space repository.

### Files to upload

| Local file | Upload as (Space path) | Required |
|------------|------------------------|----------|
| `gradio_app.py` | `app.py` | Yes — rename to `app.py` |
| `requirements_hf.txt` | `requirements.txt` | Yes — rename to `requirements.txt` |
| `dataset.py` | `dataset.py` | Yes |
| `model.py` | `model.py` | Yes |
| `infer.py` | `infer.py` | Yes |
| `recognize.py` | `recognize.py` | Yes |
| `segment_lines.py` | `segment_lines.py` | Yes |
| `checkpoints/best.pt` | `checkpoints/best.pt` | Yes — CRNN weights |
| `checkpoints/line_detector.pt` | `checkpoints/line_detector.pt` | Recommended — YOLO line detector |

Without `line_detector.pt`, page segmentation falls back to classical CV.

### Do not upload

Training scripts, dataset folders, Flask UI, and local artifacts are not needed on the Space:

- `app.py` (Flask), `templates/`
- `train.py`, `train_line_detector.py`, `evaluate.py`, `prepare_*.py`
- `data/`, `runs/`, `segmented_debug/`
- Source BN-HTRd dataset

### Space layout

```
your-space/
├── app.py                 # from gradio_app.py
├── requirements.txt       # from requirements_hf.txt
├── dataset.py
├── model.py
├── infer.py
├── recognize.py
├── segment_lines.py
└── checkpoints/
    ├── best.pt
    └── line_detector.pt   # optional but recommended
```

### Notes

- Large `.pt` files are stored with Git LFS on Hugging Face; the web uploader handles this automatically.
- Free **CPU** hardware works; **ZeroGPU** or a dedicated GPU Space is faster.
- For **ZeroGPU**, `gradio_app.py` already decorates inference with `@spaces.GPU`. Do not add `spaces` to `requirements.txt` (the Gradio image provides it). After uploading the updated `app.py`, restart the Space.
- After the build finishes, open `https://huggingface.co/spaces/<user>/<space>` or `https://<user>-<space>.hf.space`.
- Local Flask UI can call the Space by pasting that URL into **API URL**.

## Checkpoints

| File | Role |
|------|------|
| `checkpoints/best.pt` | CRNN recognizer |
| `checkpoints/line_detector.pt` | YOLO line detector |

Neither is committed to git (see `.gitignore`). Train locally or attach your own weights.

## Notes

- CTC models need several epochs before CER drops meaningfully.
- Vocabulary is character-level (not grapheme clusters).
- Page quality depends heavily on line segmentation; use the YOLO detector when possible.
- A few documents with broken Excel labels are skipped during manifest build.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use BN-HTRd, cite:

```
Rahman, M. A., Tabassum, N., Paul, M., Pal, R., & Islam, M. K.
BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten
Text Recognition (HTR) and Line Segmentation.
In Computer Vision and Image Analysis for Industry 4.0. Taylor & Francis.
```
