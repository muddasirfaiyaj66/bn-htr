# BN-HTR: Bangla Handwritten Text Recognition

An end-to-end pipeline for **offline Bangla handwritten line recognition**, built on the
[BN-HTRd dataset](https://data.mendeley.com/datasets/743k6dm543). Takes a full handwritten
page image and outputs recognized Bangla text — line segmentation, model training,
and inference are all included.

## Model, algorithms, and why they were chosen

This project has two separate stages, each solving a different problem, each with its
own algorithm:

### Stage 1 — Line segmentation (page → individual line images)

**Algorithm: a classical, unsupervised computer-vision pipeline** — no training required.

Pipeline: Otsu thresholding → Canny edge detection → morphological opening/dilation
(noise removal) → Hough line transform (thickens the Bangla "matra" head-stroke so each
word becomes one connected blob) → Hough circle transform (breaks apart the circular/loop
glyph shapes common in Bangla that otherwise fuse adjacent lines together) → connected-component
analysis (draws a box around each word-blob and finds its center point) → **OPTICS clustering**
on the Y-coordinate of those center points (groups blobs into lines) → full-width crop of
each cluster.

**Why this approach instead of training a segmentation model:**
- No line-level detector training data is strictly required — it works directly on a raw
  page with zero labeled examples, which matters because (as documented further down) the
  dataset does **not** actually ship a clean line-detector-ready label file for every image.
- It is the same method validated in the original BN-HTRd paper, reporting ~81.6% FM score
  on this exact dataset, so it's a known, reasonable baseline rather than an untested guess.
- **Why OPTICS specifically, not k-means**: k-means requires you to specify the number of
  clusters (lines) up front. The number of lines per page varies with handwriting size and
  page length, so there's no fixed `k` to give it. OPTICS discovers the number of clusters
  from the data's density structure, which fits this problem naturally.

**Trade-off to be aware of**: being unsupervised and rule-based, it can misfire on cramped
or unusually-spaced handwriting (two lines merging, or a single line splitting into two).
If this becomes a bottleneck for your own images, the natural upgrade is a trained detector
(e.g. YOLO) using the dataset's existing YOLO-format bounding box annotations — that's a
drop-in replacement for `segment_lines.py`, everything downstream stays the same.

### Stage 2 — Text recognition (line image → Bangla text)

**Algorithm: CRNN (Convolutional Recurrent Neural Network) trained with CTC loss.**

- **CNN backbone** (7 conv layers, batch-normalized): extracts visual features from the
  line image and progressively collapses its height down to a single-pixel-tall feature
  sequence — effectively turning a 2D image into a 1D sequence of "columns" read left to right.
- **BiLSTM** (2 layers, bidirectional, 256 hidden units): reads that column sequence forward
  and backward, so the network can use context from both directions — e.g. what a stroke
  looks like given the shapes before *and* after it, which matters a lot for cursive/joined
  handwriting.
- **CTC (Connectionist Temporal Classification) loss**: the key piece that makes this
  trainable without expensive per-character bounding-box labels. It only needs the full line
  image and its transcription as a whole — it automatically learns the alignment between
  image columns and output characters during training.

**Why CRNN+CTC instead of alternatives:**
- **vs. HMMs** (used historically for handwriting recognition): HMMs need heavier
  hand-engineered feature extraction and generally recognize less context than an RNN;
  CRNN+CTC has replaced them as the standard approach since ~2017.
- **vs. Transformer-based OCR** (e.g. TrOCR-style encoder-decoder): generally higher accuracy
  ceiling, but needs substantially more training data and compute to reach it. With ~14K
  training lines and a single consumer GPU (not a large pretraining corpus), a CRNN converges
  faster and more reliably. This is a solid path to upgrade to later once the CRNN baseline
  is working and you want to push accuracy further, ideally by fine-tuning an existing
  pretrained multilingual OCR transformer rather than training one from scratch.
- **vs. plain CNN classifier**: a fixed-output CNN needs pre-segmented, fixed-length inputs
  (one character or one fixed-length word at a time). Lines are variable-length sequences of
  characters — CTC is specifically built to handle that without needing word/character-level
  segmentation as a prerequisite.
- **CTC's biggest practical advantage for this project specifically**: the dataset does not
  provide per-character alignment — only whole-line text (reconstructed from word-level
  Excel labels, see below). CTC is exactly the loss function designed for this situation:
  sequence-level supervision, no explicit alignment needed.

Input: grayscale line image, fixed height (64px), width resized/padded up to 800px.
Output: a sequence of per-character probabilities, decoded into text via **greedy CTC decoding**
(picks the highest-probability character at each timestep, collapses repeats, removes blanks).
Greedy decoding is fast and simple; if you want a small accuracy boost later, beam search
decoding (optionally with a Bangla language model rescoring the beams) is the standard upgrade.

## Dataset

**[BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text
Recognition (HTR) and Line Segmentation](https://data.mendeley.com/datasets/743k6dm543)**

- 788 full-page handwritten images from 150 writers
- 13,867 lines, 108,147 words, 23,115 unique words
- Ground-truth text sourced from BBC Bangla News, hand-copied by volunteer writers
- Paper: Rahman et al., *"BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla
  Handwritten Text Recognition (HTR) and Line Segmentation"*, in *Computer Vision and Image
  Analysis for Industry 4.0*, Taylor & Francis.

The dataset is **not included in this repo** (too large for git) — download it from the
Mendeley link above and follow the setup steps below.

### Actual dataset structure (as distributed on Mendeley)

This differs slightly from the diagram in the paper — documented here from direct inspection
of the real download, since it isn't obvious from the paper alone:

```
BN-HTR_Dataset/
├── Segmentation_Images/
│   ├── Lines/
│   │   └── <doc>/
│   │       ├── <doc>_<page>.jpg        # FULL PAGE image (not a single line)
│   │       ├── <doc>_<page>.txt        # YOLO bounding boxes for all lines on the page
│   │       └── <doc>_<page>/
│   │           └── <doc>_<page>_<line>.jpg   # an actual single cropped line image
│   └── Words/
│       └── <doc>/<doc>_<page>/<doc>_<page>_<line>/
│           └── <doc>_<page>_<line>_<word>.jpg   # a single word crop
└── Recognition_Ground_Truth_Texts/
    └── <doc>/
        ├── <doc>.pdf
        ├── <doc>.txt      # full-document ground truth text
        └── <doc>.xlsx     # columns: Id ("<doc>_<page>_<line>_<word>"), Word (Bangla text)
```

There is **no direct per-line transcription file** — line-level ground truth is
reconstructed by grouping the rows of `<doc>.xlsx` by their shared
`<doc>_<page>_<line>` prefix, ordering by word index, and joining the words. This
repo's `prepare_manifest.py` does that automatically and pairs each reconstructed
line with its real cropped line image.

## Repo structure

```
bn-htr-project/
├── prepare_manifest.py   # scans dataset, reconstructs line text, builds train/val/test CSVs
├── vocab.py               # builds the Bangla character vocabulary for CTC
├── dataset.py              # PyTorch Dataset: preprocessing + augmentation
├── model.py                 # CRNN architecture
├── train.py                  # training loop (CTC loss, CER/WER validation, checkpointing)
├── infer.py                   # run the trained model on a single line image or folder
├── evaluate.py                 # final CER/WER on the held-out test set + example predictions
├── segment_lines.py              # classical unsupervised page → lines segmentation
├── recognize_page.py              # full pipeline: raw page photo → segmented lines → text
├── app.py                          # minimal REST API to serve the model to other applications
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/bn-htr-project.git
cd bn-htr-project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> Use a PyTorch CUDA build matching your driver. cu121/cu124 wheels work with most recent
> NVIDIA drivers (including CUDA 12.x/13.x drivers, which are backward compatible).
> Verify with:
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> ```

### 2. Download the dataset

Download `BN-HTR_Dataset.zip` from the
[Mendeley dataset page](https://data.mendeley.com/datasets/743k6dm543) and extract it
somewhere on disk, e.g.:

```
data/BN-HTR_Dataset/
```

### 3. Build the training manifest

Scans the dataset, reconstructs line-level transcriptions from the per-word Excel files,
and writes `train.csv` / `val.csv` / `test.csv` (split **by writer**, so validation/test
handwriting styles are never seen during training):

```bash
python prepare_manifest.py --dataset_root data/BN-HTR_Dataset --out_dir data
```

### 4. Build the character vocabulary

```bash
python vocab.py --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv --out vocab.json
```

### 5. Train

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

- Prints training loss and validation CER/WER every epoch.
- Saves `checkpoints/last.pt` every epoch and `checkpoints/best.pt` whenever validation
  CER improves.
- Resume an interrupted run with `--resume checkpoints/last.pt`.
- Lower `--batch_size` if you hit a CUDA out-of-memory error.

### 6. Evaluate on the held-out test set

`train.py` only ever reports validation numbers during training. Once training finishes,
check the model's real performance on `test.csv` — writers it has **never seen** in either
training or validation:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --test_csv data/test.csv
```

This prints:
- **Character Error Rate (CER)** — the percentage of characters that would need to be
  inserted, deleted, or substituted to turn the prediction into the ground truth. This is
  the primary metric for handwriting recognition quality.
- **Word Error Rate (WER)** — the same idea at the word level. Naturally higher than CER,
  since a single wrong character makes the whole word wrong.
- A random sample of predictions next to their ground truth, so you can eyeball where it's
  going right or wrong (e.g. confusing visually similar conjuncts, dropping matras, etc.)

**What to expect**: there's no single "good" number for Bangla handwriting CER — it depends
heavily on training time, augmentation, and how legible the writers' handwriting is. Treat
the first run as a baseline: if CER is very high (>40%), it's worth training for more epochs
or checking the manifest/vocab were built correctly before assuming the architecture is at fault.

### 7. Run inference

On a single line image:

```bash
python infer.py --checkpoint checkpoints/best.pt --image path/to/line.jpg
```

On a folder of line images:

```bash
python infer.py --checkpoint checkpoints/best.pt --dir path/to/folder_of_lines
```

### 8. Full pipeline — raw page photo → recognized text

Segments a full handwritten page into lines (classical CV pipeline) and recognizes each one:

```bash
python recognize_page.py --checkpoint checkpoints/best.pt --image path/to/page.jpg --save_lines_dir segmented_out
```

Or run segmentation alone, to inspect/debug line splitting before recognition:

```bash
python segment_lines.py --image path/to/page.jpg --out_dir segmented_out
```

## Using the trained model in an application

Once you have a `checkpoints/best.pt` you're happy with, there are two ways to use it
outside of the command line:

### Option A — call the Python functions directly (embed it in another Python app)

If your application is also Python (a desktop tool, a data pipeline, a Jupyter notebook,
another backend), skip the API layer entirely and reuse the same loading/inference code
`infer.py` and `recognize_page.py` already use:

```python
import torch
from model import CRNN
from dataset import preprocess_image
from infer import ctc_greedy_decode_single

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("checkpoints/best.pt", map_location=device)

model = CRNN(num_classes=len(ckpt["char2idx"]) + 1).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

img = preprocess_image("path/to/line.jpg", augment=False)
tensor = torch.from_numpy(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(tensor)[0]

text = ctc_greedy_decode_single(logits, ckpt["idx2char"])
print(text)
```

For a full page instead of a pre-cropped line, swap in `segment_lines.segment_page()` first
(see `recognize_page.py` for the complete example — it's the same pattern applied to every
detected line).

### Option B — run it as a REST API (call it from anything: a web app, mobile app, another service)

`app.py` wraps the model in a small Flask server with two endpoints. Start it with:

```bash
python app.py --checkpoint checkpoints/best.pt --port 5000
```

**Recognize a single pre-cropped line image:**

```bash
curl -X POST http://localhost:5000/recognize_line -F "image=@path/to/line.jpg"
```
```json
{ "text": "recognized bangla text here" }
```

**Recognize a full page** (runs segmentation + recognition on every detected line):

```bash
curl -X POST http://localhost:5000/recognize_page -F "image=@path/to/page.jpg"
```
```json
{
  "lines": ["line 1 text", "line 2 text", "..."],
  "full_text": "line 1 text\nline 2 text\n..."
}
```

A health check is available at `GET /health`.

This makes the model usable from anything that can make an HTTP request — a web frontend
file-upload form, a mobile app, a Postman collection, another backend service, etc. For a
production deployment (rather than local testing), put this behind a proper WSGI server
(e.g. `gunicorn`) and a reverse proxy, and consider batching requests if you expect
meaningful traffic — the current server processes one request at a time.

## Notes

- **CER/WER**: Character Error Rate / Word Error Rate are the metrics printed during
  validation. Expect noticeably high error in the first several epochs — CTC models
  typically take a while to start aligning correctly, then improve quickly.
- **Vocabulary**: character-level (not grapheme-cluster), which works well with CTC but
  means conjunct consonants are modeled as sequences of components rather than single
  units — a reasonable starting point, upgradeable later.
- **Segmentation limits**: the classical pipeline occasionally merges or splits lines
  incorrectly on cramped handwriting. A trained detector (e.g. YOLO, using the dataset's
  existing YOLO-format line annotations) is a natural upgrade if this becomes a bottleneck.
- A small number of documents (a handful out of 150) have malformed or missing Excel
  ground-truth files and are automatically skipped with a warning during manifest building.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use the BN-HTRd dataset, please cite the original paper:

```
Rahman, M. A., Tabassum, N., Paul, M., Pal, R., & Islam, M. K.
BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten
Text Recognition (HTR) and Line Segmentation.
In Computer Vision and Image Analysis for Industry 4.0. Taylor & Francis.
```
