"""
recognize.py

Recognize Bangla handwritten text from a page or line image.
"""

import cv2
import torch

from dataset import IMG_MAX_WIDTH_INFER, preprocess_array
from infer import ctc_greedy_decode_single
from segment_lines import looks_like_single_line, segment_page


class Recognizer:
    def __init__(self, model, idx2char, device):
        self.model = model
        self.idx2char = idx2char
        self.device = device

    def recognize_array(self, gray_img):
        if gray_img is not None and gray_img.ndim == 3 and gray_img.shape[2] == 1:
            gray_img = gray_img[:, :, 0]
        arr = preprocess_array(gray_img, max_w=IMG_MAX_WIDTH_INFER, augment=False)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)[0]
        return ctc_greedy_decode_single(logits, self.idx2char)

    def recognize_line_path(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        return self.recognize_array(img)

    def recognize(self, img_path, force_mode=None, detector_weights=None):
        """
        Recognize text from an image path.

        force_mode: None | "line" | "page"
        Returns dict: lines, full_text, line_count, mode, segmenter
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)

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
