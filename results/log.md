# Evaluation log

Held-out real lines: `tests/real_samples` (1 red-ink photo). Checkpoint: `checkpoints/best.pt`, which is what `python app.py` loads. Scoring uses that checkpoint's deskew/Otsu flags and the character LM at `data/lm.json` when it is present.

Ground truth: `কিংকর্তব্যবিমূঢ়`. Baseline prediction: `কিংবর্তব্যবিসুড়`.

| Step | CER | WER | Delta CER | Delta WER | What changed |
|------|-----|-----|-----------|-----------|--------------|
| baseline | 0.2667 | 1.0000 | 0 | 0 | Current CRNN path. 4 of 15 characters wrong. |
