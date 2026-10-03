# BN-HTR: Offline Bangla Handwritten Text Recognition

Muddasir Faiyaj

BN-HTR reads a handwritten Bangla page or a single line and returns Unicode text. The approach is segmentation then recognition. A page is cut into lines first. Each line is read on its own. The lines are joined in order. The recognizer never sees a full page.

Live demo: [huggingface.co/spaces/muddasir-faiyaj/BN-HTR](https://huggingface.co/spaces/muddasir-faiyaj/BN-HTR)

## Approach

| Stage | Algorithm | What it does |
| --- | --- | --- |
| 1. Line split | YOLOv8n, else ink projection | Finds each text line on a page. A short, wide photo is already one line, so this stage is skipped. |
| 2. Line image | Deskew, illumination flatten, Otsu | Straightens the line and turns the ink into a black-on-white image of height 64. |
| 3. Recognition | CRNN: CNN + 2-layer bidirectional LSTM, trained with CTC | Reads the line one character step at a time, without a box around each letter. |
| 4. Decoding | CTC prefix beam search + Bangla character trigram | Keeps several character prefixes and picks the one the language model scores highest. |
| 5. Text | Unicode normalisation | Writes ড়, ঢ়, and য় as the single characters stored in the vocabulary, then joins the lines. |

The released weights are the CRNN checkpoint `checkpoints/best.pt` (v10). An attention decoder and a TrOCR fine-tune are in the repository and are not loaded unless you ask for them.

## Flow

```
page or line image
        │
        ├─ already one line
        │
        └─ a page ──► YOLOv8n line detector
                      (else ink projection)
                │
                ▼
        deskew + Otsu, height 64
                │
                ▼
        CRNN (CNN + BiLSTM) + CTC
                │
                ▼
        prefix beam + character trigram
                │
                ▼
        Bangla text, one line per row
```

1. The image is a page or a line. A short, wide image is one line. A taller image is a page.
2. A page goes through YOLOv8n, trained on BN-HTRd line boxes. If that file is missing, ink projection splits the page.
3. Each crop is deskewed, flattened, and Otsu-binarized, then resized to height 64.
4. The CRNN emits a character distribution at each step. CTC training is what allows those steps to line up with the label when the exact character boundaries are unknown.
5. Prefix beam search (width 8) scores those steps with the character trigram in `data/lm.json`. If that file is missing, decoding is greedy CTC.
6. The line strings are normalised and stacked in top-to-bottom order.

## 1. System

The implementation is:

| Piece | File | Role |
| --- | --- | --- |
| Line crops and writer-disjoint splits | `prepare_manifest.py`, `prepare_combined.py` | BN-HTRd, BanglaWriting, and Bongabdo line manifests |
| Isolated-character sample | `prepare_matrivasha.py`, `prepare_all_datasets.py` | Capped samples mixed into later fine-tunes |
| Vocabulary | `vocab.py` | 173 characters, plus the CTC blank |
| Images and augmentation | `dataset.py`, `preprocess.py`, `normalize.py` | Height 64, deskew, Otsu, Bangla normalisation |
| Recognizer | `model.py`, `train.py` | CRNN trained with CTC |
| Line detector | `prepare_yolo_dataset.py`, `train_line_detector.py`, `segment_lines.py` | YOLOv8n on BN-HTRd line boxes |
| Decoder | `decode.py` | Character trigram, prefix beam, optional lexicon rescoring |
| Applications | `recognize.py`, `recognize_page.py`, `infer.py`, `app.py`, `gradio_app.py` | CLI, Flask UI, Gradio |
| Scoring | `evaluate.py` | CER and word error rate (WER) |

The released engine is the CRNN. An attention decoder and a TrOCR fine-tune exist in the repository and are not loaded unless you ask for them.

## 2. Recognizer

The network is a CRNN [6]. A grayscale line of height 64 is passed through seven convolutional layers. The feature map is collapsed to a sequence and read by a two-layer bidirectional LSTM with hidden size 256 and dropout 0.25. A linear layer emits one distribution per time step over the character vocabulary plus a CTC blank at index 0.

Training width is padded to 800. Inference allows width 1280 so a long line is not squeezed. The loss is CTC with `zero_infinity` [7]. The optimizer is Adam. The learning rate is reduced by half when validation CER has not improved for four epochs. Gradients are clipped at norm 5.

From v5 onward, every line is deskewed, its illumination is flattened, and it is Otsu-binarized before the network. Those two flags are stored in the checkpoint (`clean`, `binarize`). Inference must keep both on for these weights. The released path does not use the optional colour preprocessor in `preprocess.py`.

Training augmentation, applied only on the training split, is small rotation, shear, elastic warp, perspective, blur, JPEG recompression, extra margins, and a random black, blue, or red ink colour on textured paper.

Labels and predictions pass through `normalize.py`: Unicode NFC, removal of the zero-width joiner and non-joiner, and recomposition of ড় (U+09DC), ঢ় (U+09DD), and য় (U+09DF). NFC alone does not recompose those three characters. A grapheme-cluster vocabulary was counted and not adopted: on the 18,696 line images it would grow the output layer from 171 observed character types to 1,394 clusters.

## 3. Decoding

`decode.py` estimates a character trigram on the training transcripts and writes `data/lm.json`. Decoding is CTC prefix beam search with that trigram. If the file is missing, decoding falls back to greedy CTC. The released file contains a lexicon of 8,942 words. A word-level edit-distance corrector exists (`--post_correct`) and is off by default: it replaces a token only when one lexicon neighbour at distance 1 or 2 scores clearly better. The default beam width is 8 and the language-model weight is 0.15.

KenLM is declared for non-Windows installs. It does not build in this Windows environment, and it was not used for the released decoder. `pyctcdecode` can apply the unigram list; on this machine the prefix-beam decoder remains the one that runs.

## 4. Data

Only the six datasets in Section 9 were used. Line images come from BN-HTRd, BanglaWriting, and Bongabdo. Isolated characters from MatriVasha, Ekush, and part of BanglaLekha-Isolated were added in some fine-tunes, at a cap of 40 images per class. Validation and test stay the writer-disjoint line split, so CER can be compared from v4 onward.

| Split | Rows | Contents |
| --- | ---: | --- |
| `data/train.csv` | 15,103 | Lines from the three document datasets |
| `data/val.csv` | 1,781 | Writer-disjoint lines |
| `data/test.csv` | 1,812 | Writer-disjoint lines |
| `data/train_v6.csv` | 27,103 | Training lines plus 12,000 MatriVasha images |
| `data/train_all.csv` | 27,183 | Training lines plus MatriVasha (4,800), Ekush (4,880), and BanglaLekha-Isolated (2,400) |
| `data/train_v10.csv` | 3,400 | Short line fine-tune used for the released checkpoint |

BanglaLekha-Isolated folders 1–60 are the 50 basic characters and 10 digits. Folders 61–84 have no character key in the label sheet used here, so those images were not included. BanglaLekha and Ekush are stored as white ink on black; samples with mean intensity below 140 are inverted so the ink is dark on white, matching the line images. MatriVasha samples were already dark on white.

The line detector uses the BN-HTRd page images and their line boxes. It does not use the isolated-character sets.

## 5. Training

Hardware for the later runs was one NVIDIA GeForce RTX 4060 (8 GB). On this Windows machine `num_workers` must be 0; a larger value crashes the data loader. All commands below are run from `bn-htr`.

The CRNN optimizer is Adam. Each epoch log prints the learning rate in use. `ReduceLROnPlateau` halves that rate when validation CER does not improve for 4 epochs, and the next line of the log says the new rate. The attention trainer uses the same rule with patience 3. TrOCR keeps a constant rate.

Early checkpoints v1–v3 were trained on BN-HTRd lines. v3 is the starting point for the combined-data model. From v4 on, validation is the writer-disjoint line split. The file copied to `checkpoints/best.pt` is v10.

## 6. Methods used for accuracy

The released system keeps the following, in order.

1. **Line detection before recognition.** YOLOv8n finds each text line on a page. A single-line photo skips the detector. The CRNN is trained and run on one line at a time, which is the unit CTC can align.
2. **CRNN + CTC.** Convolutional features, a bidirectional LSTM, and CTC loss let the model read a line without a hand-drawn character box for every letter.
3. **Deskew, illumination flattening, and Otsu binarization.** From v5 on, training and inference share this preprocessing. It reduces tilt and paper shading before the network.
4. **Ink and paper augmentation.** Training randomly rotates, shears, blurs, and recolors ink (black, blue, or red) on textured paper so the recognizer is less tied to one scan.
5. **Character-trigram prefix beam search.** Instead of taking the single best character at each step, the decoder keeps several prefixes and scores them with a Bangla character language model (`data/lm.json`).
6. **Bangla Unicode normalisation.** ড়, ঢ়, and য় are written as the single characters the vocabulary already contains, on both labels and predictions.

## 7. Using the released weights

```powershell
Copy-Item ..\models\v10\best.pt checkpoints\best.pt
python app.py --checkpoint checkpoints\best.pt --port 5000
```

Open `http://127.0.0.1:5000/`. `POST /recognize` accepts a page or a line. `POST /recognize_page` forces page segmentation. `POST /recognize_line` forces a single line. `GET /health` reports the device and the decoder. The checkpoint resolution order is `--checkpoint`, then the `CHECKPOINT` environment variable, then `checkpoints/best.pt`, then `checkpoints/last.pt`. The line detector is `LINE_DETECTOR` or `checkpoints/line_detector.pt`.

```powershell
python recognize_page.py --checkpoint checkpoints\best.pt --image path\to\page.jpg
python infer.py --checkpoint checkpoints\best.pt --image path\to\line.jpg
python gradio_app.py
```

## 8. Commands

Run these from `F:\BN-HTR\bn-htr`. On this machine use `num_workers` 0 whenever the flag exists.

### 8.1 Environment

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The training machine used PyTorch 2.11 with the cu128 wheel. `requirements.txt` installs OpenCV, Flask, Ultralytics, Transformers, RapidFuzz, and pyctcdecode. The `kenlm` line is skipped on Windows.

### 8.2 Line manifests

BN-HTRd alone. Line text is rebuilt from each document workbook by grouping words on the line id.

```powershell
python prepare_manifest.py --dataset_root "F:\BN-HTR\BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)\BN-HTR_Dataset" --out_dir data --val_frac 0.1 --test_frac 0.1
```

BN-HTRd, BanglaWriting, and Bongabdo together. Bongabdo pages are split into lines and kept when the current recognizer is within `--cer_max` of the page transcript. Splits are writer-disjoint inside each dataset, then merged.

```powershell
python prepare_combined.py --bnhrd_root "F:\BN-HTR\BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)\BN-HTR_Dataset" --bongabdo_root F:\BN-HTR\Bongabdo --banglawriting_root F:\BN-HTR\BanglaWriting\converted --checkpoint ..\models\v3\best.pt --out_dir data --val_frac 0.1 --test_frac 0.1 --cer_max 0.55
```

### 8.3 Isolated-character manifests

MatriVasha only, 50 images per class from each of the male and female archives, appended to the line training manifest. This writes `data/train_v6.csv`.

```powershell
python prepare_matrivasha.py
```

All three isolated sets, 40 images per class, seed 1. BanglaLekha folders 61–84 are skipped. This writes `data/train_all.csv`.

```powershell
python prepare_all_datasets.py
```

`data/train_v10.csv` is the short line set used to fine-tune v10. Validation remains `data/val.csv`.

### 8.4 Vocabulary and character language model

```powershell
python vocab.py --train_csv data\train.csv --val_csv data\val.csv --test_csv data\test.csv --out data\vocab.json
python vocab.py --train_csv data\train.csv --val_csv data\val.csv --test_csv data\test.csv --graphemes --out results\grapheme_report.json
python decode.py --train_csv data\train.csv --out data\lm.json
```

`--graphemes` writes a report and does not replace `vocab.json`.

### 8.5 Line detector

```powershell
python prepare_yolo_dataset.py --dataset_root "F:\BN-HTR\BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)\BN-HTR_Dataset" --out_dir data\yolo_lines --val_ratio 0.1 --seed 42
python train_line_detector.py --data data\yolo_lines\data.yaml --model yolov8n.pt --epochs 40 --imgsz 1024 --batch 8 --device 0 --project runs\detect --name bn_htr_lines
python segment_lines.py --image path\to\page.jpg --out_dir out\lines --detector checkpoints\line_detector.pt
python segment_lines.py --image path\to\page.jpg --out_dir out\lines --classical
```

The second segmentation command ignores the detector and uses ink projection.

### 8.6 CRNN

v4, combined lines, no deskew, new optimizer, validation CER allowed to reset:

```powershell
python train.py --train_csv data\train.csv --val_csv data\val.csv --vocab data\vocab.json --resume ..\models\v3\best.pt --fresh_optim --reset_best --epochs 20 --batch_size 64 --lr 1e-4 --num_workers 0 --out_dir ..\models\v4
```

v5 through v9 use `--clean --binarize` and start the epoch counter again (`--fresh_optim --reset_best`), which is why their saved epoch index is not a continuation of the previous file. v6 trains on `data\train_v6.csv`. v7 and v9 train on `data\train.csv`. v8 trains on `data\train_all.csv`. The learning rate of these five runs is not stored in the checkpoint, so it is left off the commands below. Omitting `--lr` here is not a claim that the script default was used.

```powershell
python train.py --train_csv data\train.csv --val_csv data\val.csv --vocab data\vocab.json --resume ..\models\v4\best.pt --fresh_optim --reset_best --clean --binarize --epochs 20 --batch_size 64 --num_workers 0 --out_dir ..\models\v5
python train.py --train_csv data\train_v6.csv --val_csv data\val.csv --vocab data\vocab.json --resume ..\models\v5\best.pt --fresh_optim --reset_best --clean --binarize --epochs 20 --batch_size 64 --num_workers 0 --out_dir ..\models\v6
python train.py --train_csv data\train.csv --val_csv data\val.csv --vocab data\vocab.json --resume ..\models\v6\best.pt --fresh_optim --reset_best --clean --binarize --epochs 8 --batch_size 64 --num_workers 0 --out_dir ..\models\v7
python train.py --train_csv data\train_all.csv --val_csv data\val.csv --vocab data\vocab.json --resume ..\models\v7\best.pt --fresh_optim --reset_best --clean --binarize --epochs 8 --batch_size 64 --num_workers 0 --out_dir ..\models\v8
python train.py --train_csv data\train.csv --val_csv data\val.csv --vocab data\vocab.json --resume ..\models\v8\best.pt --fresh_optim --reset_best --clean --binarize --epochs 8 --batch_size 64 --num_workers 0 --out_dir ..\models\v9
```

v10, the released fine-tune. `--fresh_optim` keeps the new learning rate. `--reset_best` lets the new run replace `best.pt`. Seven epochs were saved; the best CER is 0.1038.

```powershell
python train.py --train_csv data\train_v10.csv --val_csv data\val.csv --vocab data\vocab.json --resume ..\models\v9\best.pt --fresh_optim --reset_best --clean --binarize --epochs 8 --batch_size 64 --lr 5e-5 --num_workers 0 --out_dir ..\models\v10
Copy-Item ..\models\v10\best.pt checkpoints\best.pt
```

Flags that exist and were not used for the released weights: `--enhanced`, `--sauvola`, and `--channel`. Turning them on trains and evaluates with `preprocess.py` instead of deskew plus Otsu.

### 8.7 Recognizers that were not released

Attention decoder. `--init_crnn` copies a CRNN convolutional stack into the attention model. The script's own example points that flag at the v4 checkpoint. This model is not loaded by the application.

```powershell
python train_attention.py --train_csv data\train.csv --val_csv data\val.csv --vocab data\vocab.json --init_crnn ..\models\v4\best.pt --epochs 20 --batch_size 32 --lr 1e-3 --num_workers 0 --out_dir ..\models\attention
```

TrOCR. A fine-tune of `microsoft/trocr-small-handwritten` can be loaded with `--engine trocr`. The default engine stays `crnn`.

```powershell
python train_trocr.py --train_csv data\train.csv --val_csv data\val.csv --model_name microsoft/trocr-small-handwritten --epochs 3 --batch_size 4 --lr 5e-5 --max_length 128 --num_workers 0 --out_dir ..\models\trocr
```

`--encoder` and `--decoder` together build a custom vision-encoder / text-decoder pair. `--lora` needs the optional `peft` package, which is not in `requirements.txt`. `--no_amp` turns mixed precision off.

### 8.8 Evaluation

```powershell
python evaluate.py --checkpoint checkpoints\best.pt --test_csv data\test.csv --batch_size 32 --num_workers 0 --num_examples 10
python evaluate.py --checkpoint checkpoints\best.pt --test_csv data\test.csv --beam --lm data\lm.json --num_workers 0
python evaluate.py --checkpoint checkpoints\best.pt --samples_dir path\to\labeled_lines --out_json results\baseline.json
python evaluate.py --checkpoint checkpoints\best.pt --samples_dir path\to\labeled_lines --enhanced --channel auto --out_json results\preprocess_enhanced.json
python evaluate.py --checkpoint checkpoints\best.pt --samples_dir path\to\labeled_lines --enhanced --sauvola --out_json results\preprocess_sauvola.json
python evaluate.py --checkpoint checkpoints\best.pt --samples_dir path\to\labeled_lines --tune_decode
python evaluate.py --checkpoint ..\models\trocr\best --samples_dir path\to\labeled_lines --engine trocr --trocr_dir ..\models\trocr\best --out_json results\trocr_sample.json
```

`--samples_dir` expects each line image beside a UTF-8 `.txt` of the same stem. `--tune_decode` searches beam width, language-model weight, post-correction, and pyctcdecode settings, and writes `decode_config.json` only when CER improves. `--lexicon` is accepted by the scorer. CSV scoring is greedy unless `--beam` is set.

### 8.9 Inference and the two applications

```powershell
python infer.py --checkpoint checkpoints\best.pt --image path\to\line.jpg
python infer.py --checkpoint checkpoints\best.pt --dir path\to\lines --lm data\lm.json
python infer.py --checkpoint checkpoints\best.pt --image path\to\line.jpg --greedy
python infer.py --checkpoint checkpoints\best.pt --image path\to\line.jpg --enhanced --channel auto
python infer.py --checkpoint checkpoints\best.pt --image path\to\line.jpg --enhanced --sauvola
python infer.py --checkpoint checkpoints\best.pt --image path\to\line.jpg --post_correct
python recognize.py --image path\to\page.jpg --checkpoint checkpoints\best.pt --engine crnn
python recognize.py --image path\to\page.jpg --engine trocr --trocr_dir ..\models\trocr\best
python recognize_page.py --checkpoint checkpoints\best.pt --image path\to\page.jpg --save_lines_dir out\lines
python recognize_page.py --checkpoint checkpoints\best.pt --image path\to\page.jpg --force_page
python recognize_page.py --checkpoint checkpoints\best.pt --image path\to\line.jpg --force_line
python app.py --checkpoint checkpoints\best.pt --host 0.0.0.0 --port 5000 --engine crnn
python gradio_app.py
```

`app.py` reads `HOST`, `PORT`, `CHECKPOINT`, `LINE_DETECTOR`, `ENGINE`, and `TROCR_DIR`. Gradio uses `ENGINE` and `TROCR_DIR` the same way. Leave `ENGINE` unset to serve the CRNN.

```powershell
curl.exe -X POST http://127.0.0.1:5000/recognize -F "image=@page.jpg"
curl.exe http://127.0.0.1:5000/health
```

### 8.10 Optional tools that did not train the released weights

A word n-gram builder. The released decoder uses the character trigram from `decode.py`, not this file.

```powershell
python build_lm.py --corpus data\train.csv --order 3 --min_count 1 --out data\word_lm
```

`--order` is 3, 4, or 5. The script writes an ARPA file, a lexicon, and a JSON table. It does not parse Wikipedia XML.

Synthetic lines. The renderer was not used to produce `checkpoints/best.pt`.

```powershell
python synth_lines.py --corpus data\train.csv --out_dir data\synth --count 100 --seed 1
```

Repeat `--font` to choose typefaces. Conjunct shaping needs libraqm.

## 9. Dataset credits

These are the only datasets whose images entered training or line-detector training.

**BN-HTRd.** Document pages, line crops, word labels, and line boxes. 788 pages, about 150 writers, 13,867 lines, and 108,147 words in the dataset paper. Used for the line recognizer and for the YOLOv8n line detector.

Rahman, M. A., Tabassum, N., Paul, M., Pal, R., and Islam, M. K. BN-HTRd: A benchmark dataset for document level offline Bangla handwritten text recognition (HTR) and line segmentation. In *Computer Vision and Image Analysis for Industry 4.0*, CRC Press, 2023. Preprint: [arXiv:2206.08977](https://arxiv.org/abs/2206.08977). Data: [10.17632/743k6dm543](https://data.mendeley.com/datasets/743k6dm543).

**BanglaWriting.** Single-page handwriting with word boxes, grouped into lines for this system.

Mridha, M. F., Ohi, A. Q., Ali, M. A., Emon, M. I., and Kabir, M. M. BanglaWriting: A multi-purpose offline Bangla handwriting dataset. *Data in Brief*, 34, 106633, 2021. [10.1016/j.dib.2020.106633](https://doi.org/10.1016/j.dib.2020.106633). Data: [10.17632/r43wkvdk4w.1](https://data.mendeley.com/datasets/r43wkvdk4w/1).

**Bongabdo.** Phone and scan images of handwritten pages, split into lines and filtered by transcript agreement before they entered the manifest.

Ghosh, A. Towards full-page offline Bangla handwritten text recognition using an image-to-sequence architecture. *IEEE Silchar Subsection Conference (SILCON)*, 2023. [10.1109/SILCON59133.2023.10404241](https://doi.org/10.1109/SILCON59133.2023.10404241). Data: Ghosh, A. (2023). Bongabdo. UCI Machine Learning Repository. [10.24432/C5XK7S](https://doi.org/10.24432/C5XK7S).

**MatriVasha.** Isolated handwritten compound characters. 120 classes. A capped sample was mixed into v6 and v8, then followed by line-only fine-tuning.

Ferdous, J., Karmaker, S., Rabby, A. K. M. S. A., and Hossain, S. A. MatriVasha: A multipurpose comprehensive database for Bangla handwritten compound characters. In *Emerging Technologies in Data Mining and Information Security*, Lecture Notes in Networks and Systems, vol. 164. Springer, 2021. [10.1007/978-981-15-9774-9_74](https://doi.org/10.1007/978-981-15-9774-9_74). Data: [10.17632/v39pc2g2wp.1](https://data.mendeley.com/datasets/v39pc2g2wp/1).

**Ekush.** Isolated modifiers, vowels, consonants, compounds, and digits. A capped sample of the labeled folders was inverted to dark-on-white and mixed into v8.

Rabby, A. K. M. S. A., Haque, S., Islam, M. S., Abujar, S., and Hossain, S. A. Ekush: A multipurpose and multitype comprehensive database for online off-line Bangla handwritten characters. In *Recent Trends in Image Processing and Pattern Recognition*, Communications in Computer and Information Science, vol. 1037, pp. 149–158. Springer, 2019. [10.1007/978-981-13-9187-3_14](https://doi.org/10.1007/978-981-13-9187-3_14).

**BanglaLekha-Isolated.** Folders 1–60 only: 50 basic characters and 10 Bangla digits, inverted to dark-on-white, capped at 40 images per class, and mixed into v8. The compound-character folders were not given a character key by the label map used here, so they were not included.

Biswas, M., Islam, R., Shom, G. K., Shopon, M., Mohammed, N., Momen, S., and Abedin, A. BanglaLekha-Isolated: A multi-purpose comprehensive dataset of handwritten Bangla isolated characters. *Data in Brief*, 12, 103–107, 2017. [10.1016/j.dib.2017.03.035](https://doi.org/10.1016/j.dib.2017.03.035). Data: [10.17632/hf6sf8zrkc.2](https://data.mendeley.com/datasets/hf6sf8zrkc/2).

## 10. Method references

[6] Shi, B., Bai, X., and Yao, C. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(11), 2298–2304, 2017.

[7] Graves, A., Fernández, S., Gomez, F., and Schmidhuber, J. Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *ICML*, 2006.

The line detector is Ultralytics YOLOv8n, fine-tuned on the BN-HTRd line boxes.

## License

MIT. See [LICENSE](LICENSE).

The datasets remain under the terms set by their authors. This repository's license does not relicense those images.
