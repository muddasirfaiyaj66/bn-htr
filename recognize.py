"""
recognize.py

Recognize Bangla handwritten text from a page or line image.
"""

from __future__ import annotations

import argparse
import logging

import cv2
import torch

from dataset import IMG_MAX_WIDTH_INFER, preprocess_array
from decode import decode_log_probs, load_lm, logits_to_log_probs
from infer import ctc_greedy_decode_single
from model import CRNN
from segment_lines import looks_like_single_line, segment_page

logger = logging.getLogger(__name__)


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_crnn_recognizer(ckpt, device, enhanced=False, sauvola=False, channel="auto"):
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
    )


class Recognizer:
    def __init__(self, model, idx2char, device, lm=None, beam_width=8, lm_weight=0.15, lexicon=True, clean=False, binarize=False, enhanced=False, sauvola=False, channel="auto"):
        self.model = model
        self.idx2char = idx2char
        self.device = device
        self.lm = lm if lm is not None else load_lm()
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.lexicon = lexicon
        self.clean = clean or binarize
        self.binarize = binarize
        self.enhanced = enhanced
        self.sauvola = sauvola
        self.channel = channel
        self.lm_loaded = self.lm is not None
        self.decoder_name = "ctc-prefix-beam+char-lm" if self.lm_loaded else "greedy"
        if self.lm_loaded:
            logger.info("Language model loaded from data/lm.json. Decoder: %s", self.decoder_name)
        else:
            logger.info("Language model not loaded. Decoder: greedy CTC")

    def recognize_array(self, gray_img):
        if gray_img is not None and gray_img.ndim == 3 and gray_img.shape[2] == 1:
            gray_img = gray_img[:, :, 0]
        arr = preprocess_array(
            gray_img,
            max_w=IMG_MAX_WIDTH_INFER,
            augment=False,
            clean=self.clean,
            binarize=self.binarize,
            enhanced=self.enhanced,
            sauvola=self.sauvola,
            channel=self.channel,
            allow_wide=True,
        )
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)[0]
        if self.lm is None or self.beam_width <= 1:
            return ctc_greedy_decode_single(logits, self.idx2char)
        log_probs = logits_to_log_probs(logits.detach().float().cpu().numpy())
        return decode_log_probs(
            log_probs,
            self.idx2char,
            lm=self.lm,
            beam_width=self.beam_width,
            lm_weight=self.lm_weight,
            lexicon=self.lexicon,
        )

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
            return {
                "lines": [text],
                "full_text": text,
                "line_count": 1,
                "mode": "line",
                "segmenter": "none",
            }

        _, line_imgs, segmenter = segment_page(
            img_path, detector_weights=detector_weights
        )

        if not line_imgs:
            text = self.recognize_array(img)
            return {
                "lines": [text],
                "full_text": text,
                "line_count": 1,
                "mode": "page",
                "segmenter": f"{segmenter}+fallback_whole",
            }

        results = [self.recognize_array(line) for line in line_imgs]
        results = [t if t is not None else "" for t in results]
        return {
            "lines": results,
            "full_text": "\n".join(results),
            "line_count": len(results),
            "mode": "page",
            "segmenter": segmenter,
        }
