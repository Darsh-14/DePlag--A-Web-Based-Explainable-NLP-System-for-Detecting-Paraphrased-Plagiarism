"""
explainability/integrated_gradients.py
=======================================
Integrated Gradients (IG) token attribution for SBERT.

Given a suspicious sentence s and a source embedding e_src, computes a
scalar attribution score φ_k for each token w_k indicating how much that
token contributes to the cosine similarity score S_sim = cos(e_s, e_src).

The IG formula (Sundararajan et al., 2017):
    φ_k = (x_k − x̃_k) × ∫₀¹ ∂S_sim(x̃ + α(x − x̃)) / ∂x_k  dα

where x̃ is a zero-vector baseline and the integral is approximated by
m=50 Riemann steps.

Usage
-----
    from explainability.integrated_gradients import IntegratedGradients
    ig = IntegratedGradients(sbert_model)
    scores = ig.attribute(suspicious_text, source_embedding)
    # scores: list of (token_string, attribution_float)
"""

import torch
import numpy as np
from typing import Optional


class IntegratedGradients:
    """
    Computes token-level Integrated Gradients w.r.t. cosine similarity
    between the SBERT encoding of a text and a fixed source embedding.
    """

    def __init__(self, sbert_model, n_steps: int = 50):
        """
        Parameters
        ----------
        sbert_model : SentenceTransformer
            The loaded fine-tuned SBERT instance (deplag_brain).
        n_steps : int
            Number of interpolation steps for the Riemann approximation.
            50 is a good balance between accuracy and speed.
        """
        self.model   = sbert_model
        self.n_steps = n_steps

        # Grab the underlying HuggingFace transformer and its embedding table
        self._transformer = sbert_model._first_module()
        self._hf_model    = self._transformer.auto_model
        self._tokenizer   = self._transformer.tokenizer
        self._embed_table = self._hf_model.get_input_embeddings()

    # ── Public API ────────────────────────────────────────────────────────────

    def attribute(
        self,
        text: str,
        source_embedding: np.ndarray,
        device: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """
        Returns
        -------
        list of (token_string, attribution_score)
            Tokens are the raw tokenizer sub-words; scores are L1-normalised
            so the highest-attribution token always has score 1.0.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        e_src = torch.tensor(source_embedding, dtype=torch.float32, device=device)
        if e_src.ndim == 1:
            e_src = e_src.unsqueeze(0)           # (1, D)

        # Tokenize
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        ).to(device)
        input_ids  = enc["input_ids"]            # (1, T)
        token_strs = self._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        # Baseline: zero embedding (same shape as token embedding table row)
        embed_dim = self._embed_table.weight.shape[1]
        baseline  = torch.zeros(1, input_ids.shape[1], embed_dim, device=device)

        # Get actual input embeddings
        with torch.no_grad():
            input_embeds = self._embed_table(input_ids)   # (1, T, D)

        # Accumulate gradients along the interpolation path
        accumulated_grads = torch.zeros_like(input_embeds)
        for step in range(1, self.n_steps + 1):
            alpha      = step / self.n_steps
            interp     = baseline + alpha * (input_embeds - baseline)
            interp     = interp.detach().requires_grad_(True)

            # Forward pass with interpolated embeddings
            sim = self._cosine_sim_from_embeds(interp, enc, e_src, device)
            sim.backward()

            accumulated_grads += interp.grad.detach()

        # IG = (input − baseline) × average_gradient
        ig_attrs = (input_embeds.detach() - baseline) * (accumulated_grads / self.n_steps)
        # Reduce per-token: L2 norm of the per-token IG vector
        token_scores = ig_attrs.squeeze(0).norm(dim=-1).cpu().numpy()   # (T,)

        # L1 normalise to [0, 1]
        max_score = token_scores.max()
        if max_score > 0:
            token_scores = token_scores / max_score

        return list(zip(token_strs, token_scores.tolist()))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cosine_sim_from_embeds(self, input_embeds, enc, e_src, device):
        """Runs the transformer with provided embeddings and computes cosine sim."""
        attention_mask = enc.get("attention_mask")
        token_type_ids = enc.get("token_type_ids")

        outputs = self._hf_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state   # (1, T, D)

        # Mean pooling (matches SBERT's default Pooling module)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        # L2-normalise
        e_susp = torch.nn.functional.normalize(pooled, dim=-1)     # (1, D)
        e_src_ = torch.nn.functional.normalize(e_src,  dim=-1)     # (1, D)

        return (e_susp * e_src_).sum()   # scalar cosine similarity


# ── Word-level aggregation helper ─────────────────────────────────────────────

def aggregate_to_words(
    token_attributions: list[tuple[str, float]],
    original_text: str,
) -> list[tuple[str, float]]:
    """
    Merges sub-word token attributions into whole-word scores by summing.

    Most useful for display: the GWIE algorithm works at sub-word level
    internally but this gives a cleaner view for the UI.
    """
    words, word_scores = [], []
    current_word, current_score = "", 0.0

    for tok, score in token_attributions:
        # Skip special tokens
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"):
            continue
        # Sub-word tokens start with "##" (BERT) or "▁" (T5/SentencePiece)
        if tok.startswith("##"):
            current_word  += tok[2:]
            current_score += score
        elif tok.startswith("▁") or tok.startswith("Ġ"):
            if current_word:
                words.append(current_word)
                word_scores.append(current_score)
            current_word  = tok[1:]
            current_score = score
        else:
            if current_word:
                words.append(current_word)
                word_scores.append(current_score)
            current_word  = tok
            current_score = score

    if current_word:
        words.append(current_word)
        word_scores.append(current_score)

    # Re-normalise
    max_s = max(word_scores) if word_scores else 1.0
    if max_s > 0:
        word_scores = [s / max_s for s in word_scores]

    merged = []
    for word, score in zip(words, word_scores):
        clean_word = _clean_word(word)
        if clean_word:
            merged.append((clean_word, score))

    collapsed = {}
    for word, score in merged:
        collapsed[word] = max(collapsed.get(word, 0.0), score)

    max_collapsed = max(collapsed.values()) if collapsed else 1.0
    if max_collapsed > 0:
        return [(word, score / max_collapsed) for word, score in collapsed.items()]
    return list(collapsed.items())


def _clean_word(word: str) -> str:
    word = word.replace("##", "")
    word = word.strip()
    if not word:
        return ""
    if word.startswith("[") and word.endswith("]"):
        return ""
    word = "".join(ch for ch in word if ch.isalnum() or ch in {"-", "'"})
    return word.strip("-'")
