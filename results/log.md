# Evaluation log

Held-out real lines: `tests/real_samples` (1 red-ink photo). Checkpoint: `checkpoints/best.pt`, which is what `python app.py` loads. Scoring uses that checkpoint's deskew/Otsu flags and the character LM at `data/lm.json` when it is present.

Ground truth: `কিংকর্তব্যবিমূঢ়`. Baseline prediction: `কিংবর্তব্যবিসুড়`.

| Step | CER | WER | Delta CER | Delta WER | What changed |
|------|-----|-----|-----------|-----------|--------------|
| baseline | 0.2667 | 1.0000 | 0 | 0 | Current CRNN path. 4 of 15 characters wrong. |
| preprocess enhanced | 0.4000 | 2.0000 | +0.1333 | +1.0000 | Channel pick, illumination divide, CLAHE, crop, deskew. Prediction `কর্তব্য বিমু`. Worse, so the app default stays the legacy path. |
| preprocess enhanced + Sauvola | 0.2667 | 2.0000 | 0 | +1.0000 | Same CER as baseline, extra space splits the word (`কিং কর্তব্যবিশু`). Sauvola stays off by default. |
| ink channel only | 0.2667 | 1.0000 | 0 | 0 | gray / min / green / Lab-L / auto all match the baseline prediction on this photo. |
| decoder grid | 0.2667 | 1.0000 | 0 | 0 | Beam, LM weight, post-correction, and pyctcdecode unigrams (KenLM is not installed on Windows). Best tie is the current prefix beam. `decode_config.json` was left unchanged. The UI status now names the decoder instead of a bare segmenter `none`. |
| unicode normalisation | 0.2667 | 1.0000 | 0 | 0 | NFC plus composed ড় ঢ় য় on labels and predictions. This sample was already composed, so CER did not move. Grapheme clusters would grow the vocab from 171 characters to 1394 and were not adopted, because that needs a retrain before CER can be measured. |
| augmentation | 0.2667 | 1.0000 | 0 | 0 | New training augmentations and `synth_lines.py` do not change the current checkpoint. Inference CER stays on the legacy preprocess. |
