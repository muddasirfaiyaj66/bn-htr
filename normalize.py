"""
normalize.py

Bangla Unicode normalisation shared by decoding, evaluation, and training labels.

NFC does not recompose ড় ঢ় য়: those letters are in the Unicode composition
exclusion list, so decomposed nukta forms are mapped back to the precomposed
characters the CRNN vocabulary already contains.
"""

from __future__ import annotations

import unicodedata

_COMPOSE = (
    ("\u09a1\u09bc", "\u09dc"),  # ড + ় -> ড়
    ("\u09a2\u09bc", "\u09dd"),  # ঢ + ় -> ঢ়
    ("\u09af\u09bc", "\u09df"),  # য + ় -> য়
)


def normalize_bangla(text: str) -> str:
    """NFC, drop joiners, and unify precomposed vs decomposed ড় ঢ় য়."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    for decomposed, composed in _COMPOSE:
        text = text.replace(decomposed, composed)
    return " ".join(text.split())
