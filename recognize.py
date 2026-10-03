"""
recognize.py

Recognize Bangla handwritten text from a page or line image.
"""

from __future__ import annotations

import argparse
import logging
import os

import cv2
import numpy as np
import torch

from dataset import IMG_MAX_WIDTH_INFER, preprocess_array
from decode import (
    correct_with_lm,
    decode_log_probs,
    describe_decoder,
    find_lm_path,
    load_lm,
    logits_to_log_probs,
    pyctc_decode,
)
from infer import ctc_greedy_decode_single
from model import CRNN
from normalize import normalize_bangla
from segment_lines import looks_like_single_line, segment_page


def pack_result(lines, mode, segmenter, decoder, lm_loaded, lm_path):
    """JSON payload shared by the CRNN and TrOCR engines."""
    lines = [t if t is not None else "" for t in lines]
    return {
        "lines": lines,
        "full_text": "\n".join(lines),
        "line_count": len(lines),
        "mode": mode,
        "segmenter": segmenter,
        "decoder": decoder,
        "lm_loaded": bool(lm_loaded),
        "lm_path": lm_path,
    }

logger = logging.getLogger(__name__)


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_crnn_recognizer(ckpt, device, enhanced=False, sauvola=False, channel="auto", post_correct=False, decode_engine="prefix", beam_width=8, lm_weight=0.15, alpha=0.5, beta=1.5, hotwords=None, kenlm_path=None):
    idx2char = ckpt["idx2char"]
    if idx2char and isinstance(next(iter(idx2char)), str):
        idx2char = {int(k): v for k, v in idx2char.items()}
    num_classes = len(ckpt["char2idx"]) + 1
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return Recognizer(
        model,
        idx2char,
        device,
        clean=bool(ckpt.get("clean", False)),
        binarize=bool(ckpt.get("binarize", False)),
        enhanced=enhanced,
        sauvola=sauvola,
        channel=channel,
        post_correct=post_correct,
        decode_engine=decode_engine,
        beam_width=beam_width,
        lm_weight=lm_weight,
        alpha=alpha,
        beta=beta,
        hotwords=hotwords,
        kenlm_path=kenlm_path,
    )


class Recognizer:
    def __init__(self, model, idx2char, device, lm=None, beam_width=8, lm_weight=0.15, lexicon=True, clean=False, binarize=False, enhanced=False, sauvola=False, channel="auto", post_correct=False, lm_margin=1.0, decode_engine="prefix", alpha=0.5, beta=1.5, hotwords=None, kenlm_path=None, lm_path=None):
        self.model = model
        self.idx2char = idx2char
        self.device = device
        self.lm_path = lm_path or find_lm_path()
        self.lm = lm if lm is not None else load_lm(self.lm_path)
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.lexicon = lexicon
        self.clean = clean or binarize
        self.binarize = binarize
        self.enhanced = enhanced
        self.sauvola = sauvola
        self.channel = channel
        self.post_correct = post_correct
        self.lm_margin = lm_margin
        self.decode_engine = decode_engine
        self.alpha = alpha
        self.beta = beta
        self.hotwords = list(hotwords or [])
        self.kenlm_path = kenlm_path
        self.lm_loaded = self.lm is not None
        self.decoder_name = describe_decoder(
            self.lm, self.lm_path, beam_width=self.beam_width, post_correct=self.post_correct, engine=self.decode_engine
        )
        logger.info("Active decoder: %s", self.decoder_name)

    def _prepare(self, image):
        if image is not None and image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        return preprocess_array(
            image,
            max_w=IMG_MAX_WIDTH_INFER,
            augment=False,
            clean=self.clean,
            binarize=self.binarize,
            enhanced=self.enhanced,
            sauvola=self.sauvola,
            channel=self.channel,
            allow_wide=True,
        )

    def decode_arrays(self, log_probs, logits=None):
        """Decode one line. `logits` are the raw network outputs for pyctcdecode."""
        text = None
        if self.decode_engine == "pyctc" and logits is not None:
            unigrams = list(self.lm.lexicon) if self.lm is not None else None
            text = pyctc_decode(
                logits,
                self.idx2char,
                kenlm_path=self.kenlm_path,
                unigrams=unigrams,
                alpha=self.alpha,
                beta=self.beta,
                beam_width=self.beam_width,
                hotwords=self.hotwords,
            )
        if text is None:
            if self.lm is None or self.beam_width <= 1:
                text = ctc_greedy_decode_single(torch.from_numpy(np.asarray(log_probs)), self.idx2char)
            else:
                text = decode_log_probs(
                    log_probs,
                    self.idx2char,
                    lm=self.lm,
                    beam_width=self.beam_width,
                    lm_weight=self.lm_weight,
                    lexicon=self.lexicon and not self.post_correct,
                )
        if self.post_correct and self.lm is not None:
            text = correct_with_lm(text, self.lm, margin=self.lm_margin)
        return normalize_bangla(text)

    def recognize_detail(self, image):
        """Return text plus the CTC arrays for tuning and confidence."""
        arr = self._prepare(image)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_t = self.model(tensor)[0]
        logits = logits_t.detach().float().cpu().numpy()
        log_probs = logits_to_log_probs(logits)
        text = self.decode_arrays(log_probs, logits)
        return {"text": text, "logits": logits, "log_probs": log_probs}

    def recognize_array(self, gray_img):
        return self.recognize_detail(gray_img)["text"]

    def _read_image(self, img_path):
        flag = cv2.IMREAD_COLOR if self.enhanced else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(img_path, flag)
        if img is None:
            raise FileNotFoundError(img_path)
        return img

    def recognize_line_path(self, img_path):
        return self.recognize_array(self._read_image(img_path))

    def recognize(self, img_path, force_mode=None, detector_weights=None):
        """
        Recognize text from an image path.

        force_mode: None | "line" | "page"
        Returns dict: lines, full_text, line_count, mode, segmenter
        """
        img = self._read_image(img_path)

        if force_mode == "line" or (
            force_mode is None and looks_like_single_line(img)
        ):
            text = self.recognize_array(img)
            return self._pack([text], "line", "none")

        _, line_imgs, segmenter = segment_page(
            img_path, detector_weights=detector_weights
        )

        if not line_imgs:
            text = self.recognize_array(img)
            return self._pack([text], "page", f"{segmenter}+fallback_whole")

        results = [self.recognize_array(line) for line in line_imgs]
        return self._pack(results, "page", segmenter)

    def _pack(self, lines, mode, segmenter):
        return pack_result(lines, mode, segmenter, self.decoder_name, self.lm_loaded, self.lm_path)


class TrocrRecognizer:
    """Optional VisionEncoderDecoder. Not the default; the CRNN stays in front until it wins the eval set."""

    def __init__(self, model_dir, device):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"TrOCR directory not found: {model_dir}")
        self.processor = TrOCRProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.device = device
        self.model_dir = model_dir
        self.decoder_name = f"trocr ({model_dir})"
        self.lm_loaded = False
        self.lm_path = None
        self.enhanced = False
        logger.info("Active decoder: %s", self.decoder_name)

    def recognize_array(self, image):
        from PIL import Image

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = self.processor(Image.fromarray(rgb), return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated = self.model.generate(pixels)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return normalize_bangla(" ".join(text.split()))

    def _read_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        return img

    def recognize_line_path(self, img_path):
        return self.recognize_array(self._read_image(img_path))

    def recognize(self, img_path, force_mode=None, detector_weights=None):
        img = self._read_image(img_path)
        if force_mode == "line" or (force_mode is None and looks_like_single_line(img)):
            return pack_result([self.recognize_array(img)], "line", "none", self.decoder_name, False, None)
        _page, line_imgs, segmenter = segment_page(img_path, detector_weights=detector_weights)
        if not line_imgs:
            return pack_result(
                [self.recognize_array(img)], "page", f"{segmenter}+fallback_whole", self.decoder_name, False, None
            )
        lines = [self.recognize_array(line) for line in line_imgs]
        return pack_result(lines, "page", segmenter, self.decoder_name, False, None)


def main():
    """CLI: python recognize.py --image line.jpg --checkpoint checkpoints\\best.pt"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Recognize a Bangla handwriting image")
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--engine", choices=("crnn", "trocr"), default="crnn")
    ap.add_argument("--trocr_dir", default=None)
    ap.add_argument("--enhanced", action="store_true")
    ap.add_argument("--sauvola", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.engine == "trocr":
        if not args.trocr_dir:
            ap.error("--engine trocr needs --trocr_dir")
        recognizer = TrocrRecognizer(args.trocr_dir, device)
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required for the CRNN")
        ckpt = load_torch_checkpoint(args.checkpoint, device)
        recognizer = build_crnn_recognizer(ckpt, device, enhanced=args.enhanced, sauvola=args.sauvola)
    result = recognizer.recognize(args.image)
    print(result["full_text"])
    print(result["decoder"])


if __name__ == "__main__":
    main()
