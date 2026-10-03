"""Unit tests for coloured-ink line preprocessing."""

import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocess import (  # noqa: E402
    PreprocessConfig,
    _contrast,
    crop_to_ink,
    preprocess_line,
    sauvola_threshold,
    select_ink_channel,
)
from segment_lines import looks_like_single_line  # noqa: E402


class PreprocessTest(unittest.TestCase):
    def test_coloured_ink_beats_plain_grayscale(self):
        paper = np.full((80, 400, 3), 150, np.uint8)
        paper[30:50, 40:360] = (50, 50, 180)  # red ink, BGR
        gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
        chosen = select_ink_channel(paper, "auto")
        self.assertGreaterEqual(_contrast(chosen), _contrast(gray) - 0.5)

    def test_resize_pads_on_the_right_without_stretching(self):
        img = np.full((40, 200), 255, np.uint8)
        img[12:28, 20:180] = 20
        out = preprocess_line(img, PreprocessConfig(target_h=64, max_w=1280, clahe=False))
        self.assertEqual(out.shape, (64, 1280))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out[:, -1].min()), 250)
        self.assertLess(int(out[:, 40].min()), 250)

    def test_very_wide_line_is_not_squashed(self):
        img = np.full((30, 3000), 255, np.uint8)
        img[8:22, 10:2990] = 0
        out = preprocess_line(img, PreprocessConfig(target_h=64, max_w=1280, allow_wide=True, clahe=False))
        self.assertEqual(out.shape[0], 64)
        self.assertGreater(out.shape[1], 1280)

    def test_crop_ignores_specks(self):
        img = np.full((120, 400), 255, np.uint8)
        img[50:70, 80:320] = 0
        img[2:4, 2:4] = 0
        cropped = crop_to_ink(img, margin=4)
        self.assertLess(cropped.shape[0], 50)
        self.assertLess(cropped.shape[1], 280)
        self.assertGreater(cropped.shape[1], 200)

    def test_sauvola_is_binary(self):
        img = np.full((64, 320), 210, np.uint8)
        img[24:40, 20:300] = 30
        binary = sauvola_threshold(img)
        values = set(np.unique(binary).tolist())
        self.assertTrue(values.issubset({0, 255}))
        self.assertIn(0, values)

    def test_single_line_heuristic_still_accepts_a_line(self):
        line = np.full((80, 640), 230, np.uint8)
        line[36:44, 20:620] = 15
        self.assertTrue(looks_like_single_line(line))
        prepared = preprocess_line(line, PreprocessConfig(clahe=False, deskew=False))
        self.assertTrue(looks_like_single_line(prepared))
        page = np.full((700, 500), 230, np.uint8)
        page[40:70, 30:470] = 15
        page[220:250, 30:470] = 15
        page[400:430, 30:470] = 15
        self.assertFalse(looks_like_single_line(page))


if __name__ == "__main__":
    unittest.main()
