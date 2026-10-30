"""
app.py
======
DePlag Plagiarism Detection — Inference Pipeline

Uses the fine-tuned cross-encoder (RoBERTa-base, 90.84% test accuracy) for
classification.  The SBERT model (deplag_brain) is kept loaded for the XAI
layer (GWIE counterfactuals / Integrated Gradients attribution).

Cross-encoder weights:  models/config.json + model.safetensors + tokenizer.*
SBERT weights:          models/deplag_brain/
"""

import os
import sys
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from nlp.preprocess import normalize

# ── Model Paths ────────────────────────────────────────────────────────────────
CROSS_ENCODER_PATH = "models"                # config.json + model.safetensors
SBERT_PATH         = "models/deplag_brain"   # kept for XAI (GWIE / IG)
RESULTS_PATH       = "models/results.json"   # threshold + metadata
MAX_LENGTH         = 192                     # must match training config

# ── Module-level handles (load once) ───────────────────────────────────────────
_model     = None   # cross-encoder
_tokenizer = None   # RobertaTokenizer
_threshold = None   # F1-optimal decision threshold (0.435)
_sbert     = None   # SBERT for XAI — exposed for server.py
_device    = None


def _load_models():
    global _model, _tokenizer, _threshold, _sbert, _device

    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DePlag] Loading cross-encoder from '{CROSS_ENCODER_PATH}' on {_device} ...")
        _tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_PATH)
        _model.to(_device)
        _model.eval()
        print(f"[DePlag] Cross-encoder loaded ({_model.config.model_type}, "
              f"{sum(p.numel() for p in _model.parameters())/1e6:.0f}M params)")

    if _threshold is None:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        _threshold = results["threshold"]
        print(f"[DePlag] Decision threshold: {_threshold:.4f}  (F1-optimal)")

    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        print(f"[DePlag] Loading SBERT from '{SBERT_PATH}' (for XAI) ...")
        _sbert = SentenceTransformer(SBERT_PATH)


# ── Cross-encoder inference ────────────────────────────────────────────────────

@torch.no_grad()
def _predict_single(text_a: str, text_b: str) -> float:
    """Returns P(plagiarism) from the cross-encoder for one pair."""
    enc = _tokenizer(
        text_a, text_b,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(_device)

    logits = _model(**enc).logits

    if logits.shape[-1] == 1:
        return torch.sigmoid(logits).item()
    return torch.softmax(logits, dim=-1)[0, 1].item()


@torch.no_grad()
def _predict_batch(pairs: list[tuple[str, str]]) -> np.ndarray:
    """Returns P(plagiarism) array for a batch of text pairs."""
    texts_a = [a for a, _ in pairs]
    texts_b = [b for _, b in pairs]

    enc = _tokenizer(
        texts_a, texts_b,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(_device)

    logits = _model(**enc).logits

    if logits.shape[-1] == 1:
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_plagiarism(text_a: str, text_b: str) -> dict:
    """
    Detects plagiarism between two texts using the cross-encoder.

    Returns:
        {
          "is_plagiarism": bool,
          "confidence":    float,   # P(plagiarism) [0, 1]
          "label":         str,
          "plagiarism_type": str,
          "cosine_similarity": float,
          "word_overlap":  float,
        }
    """
    _load_models()

    text_a_n, text_b_n = normalize(text_a), normalize(text_b)

    # Cross-encoder classification
    proba   = _predict_single(text_a_n, text_b_n)
    is_plag = bool(proba >= _threshold)

    # SBERT cosine similarity (for sub-typing + UI display)
    embeddings = _sbert.encode(
        [text_a_n, text_b_n],
        batch_size=2,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
    jaccard    = _jaccard(text_a_n, text_b_n)

    return {
        "is_plagiarism":     is_plag,
        "confidence":        round(float(proba), 4),
        "label":             "Plagiarism" if is_plag else "Not Plagiarism",
        "plagiarism_type":   _classify_type(is_plag, cosine_sim, jaccard),
        "cosine_similarity": round(cosine_sim, 4),
        "word_overlap":      round(jaccard, 4),
    }


def detect_batch(pairs: list[tuple[str, str]]) -> list[dict]:
    """
    Efficient batch detection for multiple text pairs.

    Args:
        pairs: list of (text_a, text_b) tuples

    Returns:
        list of result dicts (same schema as detect_plagiarism)
    """
    _load_models()

    pairs_n = [(normalize(a), normalize(b)) for a, b in pairs]

    # Cross-encoder batch prediction
    probas = _predict_batch(pairs_n)

    # SBERT batch encode for cosine similarity
    texts_flat = [t for pair in pairs_n for t in pair]
    embeddings = _sbert.encode(
        texts_flat,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb_a = embeddings[0::2]
    emb_b = embeddings[1::2]

    results = []
    for i, (text_a, text_b) in enumerate(pairs_n):
        proba      = float(probas[i])
        is_plag    = bool(proba >= _threshold)
        cosine_sim = float(np.dot(emb_a[i], emb_b[i]))
        jaccard    = _jaccard(text_a, text_b)
        results.append({
            "is_plagiarism":     is_plag,
            "confidence":        round(proba, 4),
            "label":             "Plagiarism" if is_plag else "Not Plagiarism",
            "plagiarism_type":   _classify_type(is_plag, cosine_sim, jaccard),
            "cosine_similarity": round(cosine_sim, 4),
            "word_overlap":      round(jaccard, 4),
        })
    return results


# ── Utility ────────────────────────────────────────────────────────────────────
def _jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity on word sets."""
    a_words = set(text_a.lower().split())
    b_words = set(text_b.lower().split())
    if not a_words and not b_words:
        return 0.0
    intersection = len(a_words & b_words)
    union        = len(a_words | b_words)
    return intersection / union if union else 0.0


def _classify_type(is_plag: bool, cosine_sim: float, jaccard: float) -> str:
    """Heuristic plagiarism sub-type based on similarity signals."""
    if not is_plag:
        return "None"
    if jaccard > 0.7:
        return "Verbatim Copy"
    if jaccard > 0.3:
        return "Patchwriting"
    if cosine_sim > 0.85:
        return "Structural Plagiarism"
    return "Paraphrase Plagiarism"


# ── CLI Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        (
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn fox leaps across a sleepy canine.",
        ),
        (
            "Climate change is accelerating due to greenhouse gas emissions.",
            "Global warming is speeding up because of carbon dioxide in the atmosphere.",
        ),
        (
            "I love eating pizza on Friday nights.",
            "Quantum mechanics describes the behavior of subatomic particles.",
        ),
    ]

    print("\n" + "="*60)
    print("DePlag — Plagiarism Detection Demo (Cross-Encoder)")
    print("="*60)
    for i, (a, b) in enumerate(examples, 1):
        result = detect_plagiarism(a, b)
        print(f"\nPair {i}:")
        print(f"  A: {a}")
        print(f"  B: {b}")
        print(f"  -> {result['label']} ({result['confidence']:.1%} confidence)")
        print(f"     Type: {result['plagiarism_type']}")
        print(f"     Cosine sim: {result['cosine_similarity']}  |  Word overlap: {result['word_overlap']}")
