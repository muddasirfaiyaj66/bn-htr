"""Training augmentation and synthetic-line rendering."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import _augment, _augment_color  # noqa: E402
from synth_lines import find_fonts, render_line  # noqa: E402


class AugmentTest(unittest.TestCase):
    def test_color_and_gray_augment_stay_usable(self):
        img = np.full((48, 220), 255, np.uint8)
        img[16:32, 20:200] = 15
        color = _augment_color(img)
        self.assertEqual(color.ndim, 3)
        self.assertEqual(color.shape[2], 3)
        gray = _augment(img)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.dtype, np.uint8)

    def test_synth_render_has_ink(self):
        fonts = find_fonts()
        if not fonts:
            self.skipTest("no Bangla font installed")
        rendered = render_line("কিংকর্তব্য", fonts[0], size=40)
        self.assertEqual(rendered.ndim, 2)
        self.assertLess(int(rendered.min()), 80)


if __name__ == "__main__":
    unittest.main()
