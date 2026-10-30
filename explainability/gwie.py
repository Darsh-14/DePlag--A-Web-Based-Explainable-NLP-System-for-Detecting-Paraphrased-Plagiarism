"""
explainability/gwie.py
======================
Gradient-Weighted Iterative Erasure (GWIE) with practical guards for
interactive use.
"""

import re
from collections import Counter

import numpy as np
import torch

from explainability.integrated_gradients import IntegratedGradients, aggregate_to_words

MASK_TOKEN = "[ERASED]"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "being",
    "but", "by", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "hers", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "itself", "me", "might", "my", "no", "not", "of", "on", "or",
    "our", "ours", "she", "should", "since", "so", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "to", "too", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with", "would",
    "you", "your", "yours",
}


class GWIE:
    def __init__(
        self,
        sbert_model,
        alpha: float = 0.75,
        max_iter: int = 20,
        ig_steps: int = 30,
    ):
        self.model = sbert_model
        self.alpha = alpha
        self.max_iter = max_iter
        self.ig = IntegratedGradients(sbert_model, n_steps=ig_steps)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def explain(self, suspicious_text: str, source_embedding: np.ndarray) -> dict:
        e_src = torch.tensor(source_embedding, dtype=torch.float32, device=self._device)
        current_text = suspicious_text
        current_sim = self._encode_cosine(current_text, e_src)
        initial_sim = float(current_sim)
        chosen_words = []
        chosen_keys = set()
        iterations = []
        converged = False

        for step in range(self.max_iter):
            if current_sim < self.alpha:
                converged = True
                break

            token_attrs = self.ig.attribute(current_text, source_embedding, device=self._device)
            word_attrs = aggregate_to_words(token_attrs, current_text)
            candidates = [
                (word, score)
                for word, score in word_attrs
                if self._is_valid_candidate(word, chosen_keys)
            ]
            if not candidates:
                break

            pivot_word, pivot_score = max(candidates, key=lambda item: item[1])
            next_text = _mask_word(current_text, pivot_word)
            if next_text == current_text:
                chosen_keys.add(_canonical_word(pivot_word))
                continue

            next_sim = self._encode_cosine(next_text, e_src)
            iterations.append({
                "step": step + 1,
                "pivot_word": pivot_word,
                "attribution": round(float(pivot_score), 4),
                "sim_before": round(float(current_sim), 4),
                "sim_after": round(float(next_sim), 4),
            })
            chosen_words.append(pivot_word)
            chosen_keys.add(_canonical_word(pivot_word))
            current_text = next_text
            current_sim = float(next_sim)

        if chosen_words:
            chosen_words, current_text, current_sim = self._prune_counterfactual(
                suspicious_text,
                chosen_words,
                e_src,
            )
            converged = current_sim < self.alpha

        return {
            "counterfactual_set": chosen_words,
            "counterfactual_text": current_text,
            "initial_similarity": round(initial_sim, 4),
            "final_similarity": round(float(current_sim), 4),
            "iterations": iterations,
            "converged": converged,
        }

    def _encode_cosine(self, text: str, e_src: torch.Tensor) -> float:
        emb = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        e = torch.tensor(emb, dtype=torch.float32, device=self._device)
        e = torch.nn.functional.normalize(e.unsqueeze(0), dim=-1)
        e_s = torch.nn.functional.normalize(
            e_src.unsqueeze(0) if e_src.ndim == 1 else e_src,
            dim=-1,
        )
        return float((e * e_s).sum().item())

    def _is_valid_candidate(self, word: str, chosen_keys: set[str]) -> bool:
        key = _canonical_word(word)
        if not key or key in chosen_keys:
            return False
        if key == _canonical_word(MASK_TOKEN) or "erased" in key:
            return False
        if key in STOPWORDS:
            return False
        if len(key) <= 2 and not key.isupper():
            return False
        return True

    def _prune_counterfactual(
        self,
        original_text: str,
        chosen_words: list[str],
        e_src: torch.Tensor,
    ) -> tuple[list[str], str, float]:
        retained = list(chosen_words)
        for word in chosen_words:
            candidate_words = [w for w in retained if w != word]
            candidate_text = _mask_words(original_text, candidate_words)
            candidate_sim = self._encode_cosine(candidate_text, e_src)
            if candidate_sim < self.alpha:
                retained = candidate_words

        final_text = _mask_words(original_text, retained)
        final_sim = self._encode_cosine(final_text, e_src)
        return retained, final_text, float(final_sim)


def _canonical_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def _mask_words(text: str, words: list[str]) -> str:
    masked = text
    counts = Counter(words)
    for word, count in counts.items():
        for _ in range(count):
            updated = _mask_word(masked, word)
            if updated == masked:
                break
            masked = updated
    return masked


def _mask_word(text: str, word: str) -> str:
    escaped = re.escape(word.strip())
    if not escaped:
        return text
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.sub(pattern, MASK_TOKEN, text, count=1, flags=re.IGNORECASE)
