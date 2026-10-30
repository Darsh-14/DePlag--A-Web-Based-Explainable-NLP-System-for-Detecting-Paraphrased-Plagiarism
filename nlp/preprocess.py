"""
nlp/preprocess.py
=================
Single canonical text normalization used by data-building, fine-tuning,
classifier training, and inference — ensuring no train/inference mismatch.

IMPORTANT: As of the DeBERTa-v3 cross-encoder upgrade, we use whitespace-only
normalization. Punctuation is PRESERVED because the tokenizer was pre-trained
on text with punctuation. Stripping it destroys subword boundaries and costs
4-5% accuracy.
"""

import re


def normalize(text: str) -> str:
    """
    Collapse whitespace only. Preserves punctuation and case-sensitive
    characters for the tokenizer.

    Must be applied identically to training data AND inference inputs.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())    # collapse whitespace only


def normalize_legacy(text: str) -> str:
    """Legacy version that strips punctuation. DO NOT use for DeBERTa models."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Returns a list of normalized word tokens."""
    return normalize(text).split()
