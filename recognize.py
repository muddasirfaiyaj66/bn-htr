"""
recognize.py

Recognize Bangla handwritten text from a page or line image.
"""

from __future__ import annotations

import argparse
import logging

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
from segment_lines import looks_like_single_line, segment_page

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
        return text

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
        lines = [t if t is not None else "" for t in lines]
        return {
            "lines": lines,
            "full_text": "\n".join(lines),
            "line_count": len(lines),
            "mode": mode,
            "segmenter": segmenter,
            "decoder": self.decoder_name,
            "lm_loaded": self.lm_loaded,
            "lm_path": self.lm_path,
        }
