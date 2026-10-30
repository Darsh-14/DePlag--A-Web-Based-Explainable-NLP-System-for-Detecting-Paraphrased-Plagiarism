# DePlag — Deep Research Insights Report
**Analyst:** Advanced AI Researcher & Developer  
**Date:** 2026-04-22  
**Codebase Revision:** `5ba9fcb` (main branch)

---

## 1. Project Architecture Overview

DePlag (HEPDS — Hybrid Explainable Plagiarism Detection System) is a two-stage NLP pipeline for detecting paraphrased plagiarism in text pairs. The system combines fine-tuned sentence embeddings with a gradient-boosted classifier and a rule-based XAI layer.

```
Raw Text Pair
     │
     ▼
Fine-tuned MPNet SBERT (all-mpnet-base-v2 → deplag_brain)
     │  [768-dim embeddings per text]
     ▼
Feature Engineering: [|emb_a − emb_b|, emb_a ⊙ emb_b, cos(emb_a, emb_b)]
     │  [1537-dim feature vector]
     ▼
CalibratedGBM Classifier (GradientBoosting + isotonic calibration)
     │  [P(plagiarism)]
     ▼
Threshold Decision (Youden-optimized, but overridden in server.py)
     │
     ▼
Heuristic Subtype Classifier (Verbatim / Patchwriting / Structural / Paraphrase)
     │
     ▼
XAI Layer (SpaCy NER + shared-word highlighting + template advice)
     │
     ▼
Flask Web API → index.html
```

**Training Datasets:** MSRP + PAWS + STS-Benchmark  
**Total Training Pairs:** 59,226 (31,965 negative / 27,261 positive — slight class imbalance ~54:46)  
**Base Model:** `all-mpnet-base-v2` (MPNet, 12-layer, 768-dim, 514 max tokens)  
**Fine-tuning Loss:** CoSENT Loss  
**Classifier:** GradientBoostingClassifier (200 trees, depth=4, lr=0.05) + CalibratedClassifierCV (isotonic, 3-fold)  
**Validation Spearman ρ (Epoch 2):** 0.7677  

---

## 2. Novel Features and Genuine Strengths

These are legitimately well-thought-out design decisions.

### 2.1 Two-Stage Hybrid Architecture
The combination of a fine-tuned bi-encoder + a downstream GBM classifier is architecturally sound. The bi-encoder provides a semantically rich compressed representation; the GBM then learns a non-linear decision boundary over rich interaction features. This outperforms a threshold-only approach on cosine similarity.

### 2.2 Rich 1537-Dim Interaction Feature Vector
Rather than using a single cosine score, the system constructs three interaction signals:
- **Element-wise absolute difference** (`|emb_a − emb_b|`) — captures per-dimension disagreement
- **Element-wise product** (`emb_a ⊙ emb_b`) — captures per-dimension agreement
- **Global cosine similarity scalar** — captures aggregate angular alignment

This is a well-known NLI-style feature set (inspired by InferSent/ESIM) applied to plagiarism detection, which gives the classifier fine-grained information about where two embeddings agree and disagree.

### 2.3 Probability Calibration
Using `CalibratedClassifierCV` with isotonic regression on top of the GBM is a genuinely advanced step. Raw GBM probabilities are often miscalibrated (overconfident). Calibration produces reliable posterior probabilities, which is critical for threshold-based binary decisions.

### 2.4 Threshold Optimization via Youden's J Statistic
The Youden statistic (J = TPR − FPR, maximized on the ROC curve) is statistically principled and produces a threshold that balances sensitivity and specificity, unlike naive 0.5. This is correctly implemented in `train_semantic.py:find_optimal_threshold()`.

### 2.5 Multi-Dataset Training with SNLI Removal
The deliberate removal of SNLI from the training mix (`rebuild_data.py`) is an important decontamination step. SNLI contains entailment/contradiction pairs that introduce semantic relationships orthogonal to paraphrase; their presence caused false negatives on paraphrase plagiarism (noted in the file's docstring). This shows genuine domain awareness.

### 2.6 CoSENT Loss for Fine-Tuning
CoSENT (Cosine Sentence) loss trains the model to rank similarity scores in the correct order, producing better embedding geometry for similarity tasks compared to vanilla contrastive loss. This is a more sophisticated choice than standard triplet loss.

### 2.7 Plagiarism Sub-Type Taxonomy
The four-tier classification (Verbatim Copy → Patchwriting → Structural Plagiarism → Paraphrase Plagiarism) maps to academically recognized plagiarism categories. Even as a heuristic, it provides pedagogically meaningful output for students/educators.

### 2.8 NER-Enhanced XAI
Integrating SpaCy NER to identify shared named entities (people, places, organizations) as part of the explanation goes beyond simple word overlap. Named entities are semantically salient and their sharing is stronger evidence of copying than sharing common words.

### 2.9 Efficient Batched Inference (`detect_batch`)
The `detect_batch()` function in `app.py` interleaves all texts into a single flat list and encodes them in one batched SBERT call, then slices results. This is memory-efficient and avoids redundant model loads for batch workloads.

---

## 3. Critical Gaps and Weaknesses

### 3.1 THRESHOLD MISMATCH — Critical Bug
**File:** `models/threshold.txt` vs `server.py:load_resources()`

The Youden-optimal threshold stored after training is **0.05**. However, `server.py` line 32 silently overrides this:

```python
app._threshold = 0.35
```

This means:
- The carefully computed Youden threshold (0.05) is immediately discarded at inference time.
- The actual deployed threshold (0.35) was set arbitrarily, not through any principled method.
- A trained threshold of 0.05 is itself suspicious — it implies almost anything above 5% probability of plagiarism triggers a positive. This may indicate the calibration pipeline produced a degenerate probability distribution.
- The two values (0.05 and 0.35) represent a 7× inconsistency with no documentation explaining the override.

**Impact:** Unknown, unmeasured real-world precision/recall trade-off.

---

### 3.2 Benchmark Evaluation is Not Actual Accuracy Measurement
**Files:** `benchmark_results.csv`, `benchmark_results_16.csv`, `high_accuracy_16_bi_encoder.csv`, `benchmark_models.py`

All benchmark CSVs report a **raw cosine similarity score on ONE fixed text pair**, not actual classification metrics (accuracy, F1, AUC, precision, recall) on a held-out test set. The column headers say "Similarity Score" but the charts and paper language treat these as "accuracy."

```
text1 = "The study of deep learning architectures..."
text2 = "Research into neural network frameworks..."
```

A model scoring 0.96 on this single pair does not mean it achieves 96% accuracy on plagiarism detection. A model could produce high cosine similarity for any semantically similar pair without being able to classify plagiarism. **This is a fundamental evaluation design error.**

The confusion matrix in `graphs.py` is hardcoded:
```python
cm_data = np.array([[842, 115], [130, 895]])
```
These numbers do not come from any actual model run. They are fabricated for the visualization.

---

### 3.3 No Independent Test Set
**Files:** `train_semantic.py`, evaluation pipeline

The entire dataset (59,226 pairs) is split into train/val. There is no held-out test set that the model never sees. Evaluation metrics on the validation set are optimistic because:
- The threshold is optimized on the same val set used to report performance.
- No cross-dataset generalization is tested (e.g., train on MSRP+PAWS, test on a real plagiarism corpus like PAN).

---

### 3.4 Empty Module Files — Design Intent vs. Reality
The following files exist in the repo but are completely empty (1 blank line):

| File | Intended Purpose |
|------|-----------------|
| `nlp/embeddings.py` | Embedding utilities |
| `nlp/similarity.py` | Similarity computations |
| `nlp/preprocess.py` | Text preprocessing |
| `nlp/alignment.py` | Token alignment |
| `utils/helpers.py` | General utilities |
| `explainability/counterfactuals.py` | Counterfactual generation |
| `explainability/rationale.py` | Rationale extraction |
| `explainability/critical_spans.py` | Critical span detection |
| `evaluation/metrics.py` | Evaluation metrics |
| `evaluation/human_eval.py` | Human evaluation framework |

The architecture diagram (HEPDS) shows a Forensics Module with passive voice detection and synonym analysis. These are absent from the actual code.

---

### 3.5 Shallow XAI — Not Truly Explainable
**File:** `server.py:extract_explanation_data()`, `server.py:build_advice()`

The current XAI layer is:
1. **Shared word highlighting:** Any non-stopword appearing in both texts gets highlighted. This is trivial bag-of-words overlap — it does not explain WHY the classifier flagged the text as plagiarism. A high cosine-sim, low Jaccard pair (paraphrase plagiarism) will show zero highlighted words, which is misleading.
2. **Static advice strings:** Four hardcoded advice messages. They are not counterfactual — they don't show what the user would need to change to flip the prediction.
3. **No model attribution:** The GBM classifier's decision is not explained. Which of the 1537 features drove the prediction? SHAP values on the GBM would directly answer this.
4. **Stopword list is tiny (18 words):** Many common words like "are", "was", "were", "be", "have", "has" will appear as "shared words" and pollute the evidence panel.

---

### 3.6 Data Preprocessing Inconsistency
**Files:** `rebuild_data.py:clean()` vs `fine_tune_brain.py`

`rebuild_data.py` applies aggressive normalization: lowercase + remove all punctuation + collapse whitespace. But `fine_tune_brain.py` feeds raw text from the CSV directly to SBERT without applying any such cleaning. If the `final_train_data.csv` was built with this cleaning, SBERT trains on cleaned text but inference receives raw text. This creates a train/inference distribution mismatch.

---

### 3.7 Feature Engineering is Underspecified
**File:** `app.py:_build_features()`, `train_semantic.py:build_features()`

The 1537-dim vector uses only embedding-space interaction. Missing features that would improve paraphrase detection:

- **Lexical overlap signals:** BLEU, ROUGE-2, METEOR
- **N-gram Jaccard at multiple granularities** (unigram, bigram, trigram)
- **Edit distance / normalized Levenshtein**
- **Syntactic structure similarity** (dependency parse tree edit distance)
- **Length ratio** (plagiarists often shorten/expand texts)
- **Named entity overlap ratio** (separate from XAI)
- **TF-IDF weighted overlap** (common words carry less weight)

---

### 3.8 Sub-Type Thresholds are Arbitrary
**File:** `app.py:_classify_type()`

```python
if jaccard > 0.7:   return "Verbatim Copy"
if jaccard > 0.3:   return "Patchwriting"
if cosine_sim > 0.85: return "Structural Plagiarism"
return "Paraphrase Plagiarism"
```

These cutoffs (0.7, 0.3, 0.85) are not validated against any annotated sub-type dataset. The boundaries will produce misclassification — e.g., a sentence translated from another language (back-translation attack) would score low on Jaccard AND low cosine (< 0.85) and be classified as "Paraphrase Plagiarism" even though it's structurally closer to verbatim copying in meaning.

---

### 3.9 SBERT Fine-Tuning Evaluation Mismatch
**File:** `fine_tune_brain.py`, `models/deplag_brain/eval/`

Training evaluates on `EmbeddingSimilarityEvaluator` (Spearman ρ on continuous similarity). But the downstream task is **binary classification** (plagiarism / not). High Spearman ρ does not guarantee good binary classification AUC. The fine-tuned model might produce well-ranked similarities without a clean separation at any threshold. The training pipeline has no binary classification evaluator in the loop.

Only 2 epochs of evaluation were logged (Epoch 1: ρ=0.752, Epoch 2: ρ=0.768), though 3 epochs were configured. This suggests the model may have been saved at epoch 2 (best checkpoint) but it's unclear if epoch 3 was worse.

---

### 3.10 Production Readiness Issues
**File:** `server.py`

- `app.run(debug=True, port=5000)` — debug mode should never be on in production (exposes Werkzeug debugger shell)
- No input length validation — SBERT has a 514 token limit; texts longer than this will be silently truncated
- No rate limiting on `/predict` — trivially DoS-able
- No request size limit — malicious payloads can send megabytes of text
- No async request handling — each SBERT inference blocks the Flask thread
- CORS not configured

---

### 3.11 Model Selection Inconsistency
**File:** `benchmark_models.py`, `fine_tune_brain.py`

The benchmark shows `BAAI/bge-large-en-v1.5` achieving the highest score (0.9623) on the test pair, `bert-base-nli-mean-tokens` at 0.9176, while `all-mpnet-base-v2` scored only 0.6098 (ranked 10th out of 14). Yet `all-mpnet-base-v2` was chosen as the base model. The benchmark's single-pair evaluation makes this comparison invalid, but there's no documented rationale for why all-mpnet-base-v2 was selected over the benchmark leaders.

---

### 3.12 Jaccard Computation is Naïve
**File:** `app.py:_jaccard()`

```python
def _jaccard(text_a: str, text_b: str) -> float:
    a_words = set(text_a.lower().split())
    b_words = set(text_b.lower().split())
```

- No lemmatization: "running"/"run"/"runs" count as different words
- No stopword removal: "the"/"a"/"is" inflate the intersection
- No punctuation stripping: "dog." ≠ "dog"
- Set-based (not multiset): "the cat sat" vs "the the cat" → Jaccard treats as same

---

## 4. Quantitative Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Training pairs | 59,226 | MSRP + PAWS + STS |
| Class balance | 54% neg / 46% pos | Mild imbalance |
| SBERT fine-tune epochs | 3 | Best at epoch 2 |
| Val Spearman ρ (cosine) | 0.7677 | Not binary classification metric |
| Deployed threshold | 0.35 | Overrides Youden-optimal 0.05 |
| Youden-optimal threshold | 0.05 | Stored but not used |
| Real held-out test accuracy | Unknown | No test set exists |
| Benchmark evaluation pairs | 1 | Insufficient for claims |
| Confusion matrix | Hardcoded | Not derived from data |

---

## 5. Implementation Improvement Plan

Priority levels: **[P1]** = Critical fix, **[P2]** = High impact, **[P3]** = Medium impact, **[P4]** = Research-grade enhancement.

---

### P1 — Fix Threshold Mismatch and Re-Evaluate

**Problem:** The threshold pipeline is broken — trained threshold (0.05) is discarded; deployed threshold (0.35) is arbitrary.

**Implementation:**
1. Re-run `train_semantic.py` and inspect the raw probability histogram on the val set. If probabilities cluster near 0 and 1, the 0.05 Youden threshold is wrong — likely a calibration error.
2. Add a dedicated validation report step: plot the precision-recall curve and pick the threshold that maximizes F1 at the desired operating point.
3. Remove the `app._threshold = 0.35` override in `server.py`. Store only one threshold in `threshold.txt` and document how it was chosen.

```python
# train_semantic.py — add after find_optimal_threshold()
from sklearn.metrics import precision_recall_curve
precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1_scores)
f1_optimal_thresh = pr_thresholds[best_f1_idx]
print(f"F1-optimal threshold: {f1_optimal_thresh:.4f} | F1={f1_scores[best_f1_idx]:.4f}")
```

---

### P1 — Create a Proper Test Set and Real Evaluation

**Problem:** No held-out test set; confusion matrix and F1 scores are fabricated.

**Implementation:**
1. Before any training, split the full dataset 70/15/15 (train/val/test) and save test set indices. Never touch test during training or threshold selection.
2. After final model selection, run evaluation only once on the test set.
3. Replace hardcoded confusion matrix in `graphs.py` with actual values from test evaluation.

```python
# Add to train_semantic.py
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)
# ... train on X_trainval, then:
y_test_proba = cal.predict_proba(X_test)[:, 1]
y_test_pred  = (y_test_proba >= optimal_thresh).astype(int)
print("=== HELD-OUT TEST SET ===")
print(classification_report(y_test, y_test_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
```

---

### P1 — Implement Real Benchmark Evaluation

**Problem:** Benchmarks measure cosine similarity on one sentence pair, not task-level classification performance.

**Implementation:**
Create `evaluation/benchmark_eval.py` that loads a standard corpus (e.g., PAWS test set) and evaluates each model on:

```python
# evaluation/benchmark_eval.py
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

def evaluate_model_on_corpus(model, test_df, threshold=0.5):
    """Runs classification evaluation on a full labeled corpus."""
    emb_a = model.encode(test_df['text_a'].tolist(), normalize_embeddings=True)
    emb_b = model.encode(test_df['text_b'].tolist(), normalize_embeddings=True)
    cosine_sims = (emb_a * emb_b).sum(axis=1)
    preds = (cosine_sims >= threshold).astype(int)
    return {
        "f1":       f1_score(test_df['label'], preds),
        "auc":      roc_auc_score(test_df['label'], cosine_sims),
        "accuracy": (preds == test_df['label'].values).mean()
    }
```

---

### P2 — Implement SHAP-Based XAI for Classifier Decisions

**Problem:** The GBM's decision is a black box. True explainability requires attribution over the 1537-dim feature vector.

**Implementation:**

```python
# explainability/shap_explain.py
import shap
import numpy as np

class SHAPExplainer:
    def __init__(self, classifier, feature_names=None):
        self.explainer = shap.TreeExplainer(classifier)
        self.feature_names = feature_names
    
    def explain(self, features: np.ndarray) -> dict:
        shap_values = self.explainer.shap_values(features)
        # For binary: shap_values[1] = contribution to plagiarism class
        plag_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
        # Aggregate by feature group
        diff_contribution  = float(plag_shap[:768].sum())
        prod_contribution  = float(plag_shap[768:1536].sum())
        cosine_contribution = float(plag_shap[1536])
        
        # Top dimensions driving the decision
        top_dims = np.argsort(np.abs(plag_shap))[-10:][::-1]
        
        return {
            "diff_group_shap":   round(diff_contribution, 4),
            "prod_group_shap":   round(prod_contribution, 4),
            "cosine_shap":       round(cosine_contribution, 4),
            "top_dims":          top_dims.tolist(),
            "dim_contributions": plag_shap[top_dims].tolist()
        }
```

Integrate this into `server.py:/predict` and surface the group contributions in the UI as a breakdown bar ("Meaning Divergence contributed 60% of this decision").

---

### P2 — Add Lexical Feature Augmentation

**Problem:** Only embedding-space features. Lexical signals (n-gram overlap, edit distance) improve detection, especially for patchwriting.

**Implementation:**

```python
# nlp/lexical_features.py
import re
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def tokenize_lemmatized(text: str) -> list:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [lemmatizer.lemmatize(w) for w in words]

def ngram_jaccard(a_tokens, b_tokens, n=2) -> float:
    def ngrams(tokens, n):
        return Counter(zip(*[tokens[i:] for i in range(n)]))
    a_ng = ngrams(a_tokens, n)
    b_ng = ngrams(b_tokens, n)
    intersection = sum((a_ng & b_ng).values())
    union = sum((a_ng | b_ng).values())
    return intersection / union if union else 0.0

def length_ratio(text_a: str, text_b: str) -> float:
    la, lb = len(text_a.split()), len(text_b.split())
    return min(la, lb) / max(la, lb) if max(la, lb) > 0 else 1.0

def build_lexical_features(text_a: str, text_b: str) -> np.ndarray:
    ta = tokenize_lemmatized(text_a)
    tb = tokenize_lemmatized(text_b)
    return np.array([
        ngram_jaccard(ta, tb, 1),   # unigram Jaccard (lemmatized)
        ngram_jaccard(ta, tb, 2),   # bigram Jaccard
        ngram_jaccard(ta, tb, 3),   # trigram Jaccard
        length_ratio(text_a, text_b),
        sentence_bleu([ta], tb, weights=(0.5, 0.5, 0, 0)),  # BLEU-2
    ])
```

Then extend `build_features()` to concatenate these 5 lexical features, making the vector 1542-dim.

---

### P2 — Fix Text Preprocessing Inconsistency

**Problem:** `rebuild_data.py` cleans text; `fine_tune_brain.py` feeds raw text.

**Implementation:**
Create a single canonical preprocessing function used everywhere:

```python
# nlp/preprocess.py
import re

def normalize(text: str) -> str:
    """Canonical preprocessing: lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())
```

Import and apply `normalize()` in `fine_tune_brain.py` before creating `InputExample` objects, and in `app.py` before encoding.

---

### P2 — Validate and Learn Sub-Type Thresholds

**Problem:** Sub-type boundaries (Jaccard 0.7, 0.3, cosine 0.85) are arbitrary.

**Implementation:**
If a small labeled sub-type dataset is available, train a lightweight multiclass classifier:

```python
# Short-term: systematically validate thresholds on existing data
def calibrate_subtype_thresholds(df_plagiarism, sbert_model):
    """
    Given plagiarism pairs with sub-type annotations, find optimal thresholds
    via grid search over jaccard and cosine cutoffs.
    """
    from itertools import product
    best = {"f1": 0}
    for j1, j2, c1 in product([0.5,0.6,0.7,0.8], [0.2,0.3,0.4,0.5], [0.75,0.80,0.85,0.90]):
        preds = [_classify_type(True, row.cosine, row.jaccard, j1, j2, c1)
                 for row in df_plagiarism.itertuples()]
        f1 = f1_score(df_plagiarism['subtype'], preds, average='macro')
        if f1 > best["f1"]:
            best = {"f1": f1, "j1": j1, "j2": j2, "c1": c1}
    return best
```

---

### P3 — Implement Sentence-Pair Alignment for Evidence (Critical Spans)

**Problem:** Shared word highlighting is not span-level. For multi-sentence inputs, users can't tell which sentence is the offending one.

**Implementation:**

```python
# explainability/critical_spans.py
from sentence_transformers import SentenceTransformer, util
import nltk

def find_critical_sentence_pairs(text_a: str, text_b: str, model, top_k=3):
    """
    Finds the most semantically similar sentence pairs across two documents.
    Returns ranked list of (sentence_a, sentence_b, cosine_score).
    """
    sents_a = nltk.sent_tokenize(text_a)
    sents_b = nltk.sent_tokenize(text_b)
    
    embs_a = model.encode(sents_a, normalize_embeddings=True)
    embs_b = model.encode(sents_b, normalize_embeddings=True)
    
    sim_matrix = util.cos_sim(embs_a, embs_b).numpy()
    
    pairs = []
    for i, j in zip(*sim_matrix.shape):
        pairs.append((sents_a[i], sents_b[j], float(sim_matrix[i, j])))
    
    return sorted(pairs, key=lambda x: -x[2])[:top_k]
```

Surface the top-3 sentence pairs in the UI as clickable evidence cards.

---

### P3 — Add Input Validation and Production Hardening

**Problem:** No input validation, debug mode on, no rate limiting.

**Implementation:**

```python
# server.py additions
from functools import wraps
import time

MAX_CHARS = 5000
MAX_TOKENS_APPROX = 400  # ~500 tokens after whitespace

@app.route('/predict', methods=['POST'])
def predict():
    data  = request.json
    text1 = (data.get('text1') or '').strip()
    text2 = (data.get('text2') or '').strip()
    
    if not text1 or not text2:
        return jsonify({'error': 'Both texts are required.'}), 400
    
    if len(text1) > MAX_CHARS or len(text2) > MAX_CHARS:
        return jsonify({'error': f'Text too long. Limit: {MAX_CHARS} characters.'}), 413
    
    word_count_1 = len(text1.split())
    word_count_2 = len(text2.split())
    if word_count_1 > MAX_TOKENS_APPROX or word_count_2 > MAX_TOKENS_APPROX:
        return jsonify({'warning': 'Long text will be truncated at 514 tokens by SBERT.',
                        'truncated': True}), 200
    ...

# Change app.run:
if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)
```

---

### P3 — Improve Stopword List for XAI

**Problem:** 18 stopwords miss hundreds of common English words.

**Implementation:**

```python
# server.py
import spacy
STOPWORDS = nlp.Defaults.stop_words if nlp else {
    'the','a','an','is','was','are','were','be','been','being',
    'have','has','had','do','does','did','will','would','could',
    'should','may','might','shall','to','of','in','at','on','by',
    'for','from','with','about','into','through','during','this',
    'that','these','those','it','its','and','or','but','so','yet',
    'both','either','neither','not','no','nor','as','if','than',
    'then','when','where','while','after','before','since','until'
}
```

---

### P4 — Cross-Encoder Reranking for High-Stakes Cases

**Problem:** Bi-encoders are fast but less precise than cross-encoders. A cross-encoder can verify borderline cases.

**Implementation:**

```python
# app.py — add optional cross-encoder verification
from sentence_transformers import CrossEncoder

_cross_encoder = None
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BORDERLINE_ZONE = (0.30, 0.55)  # probabilities where bi-encoder is uncertain

def verify_with_cross_encoder(text_a: str, text_b: str) -> float:
    """Runs cross-encoder on borderline cases for higher precision."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    score = _cross_encoder.predict([[text_a, text_b]])[0]
    return float(score)

# In detect_plagiarism(), after getting proba:
if BORDERLINE_ZONE[0] <= proba <= BORDERLINE_ZONE[1]:
    cross_score = verify_with_cross_encoder(text_a, text_b)
    # Ensemble: weighted average
    proba = 0.6 * proba + 0.4 * (1 / (1 + np.exp(-cross_score)))
```

---

### P4 — Document-Level Detection (Multi-Sentence)

**Problem:** Current system only compares two text strings pairwise. Real plagiarism involves documents, not sentence pairs.

**Implementation plan:**
1. Accept document-level input (full paragraphs)
2. Segment into sentences with `nltk.sent_tokenize()`
3. Encode all sentences from both documents
4. Build a similarity matrix: N×M
5. Find optimal alignment using Hungarian algorithm or greedy threshold matching
6. Report: overall plagiarism score + per-sentence evidence heatmap

This is a significant architectural extension but would make DePlag genuinely useful for academic submission checking.

---

### P4 — Language-Agnostic Detection with Multilingual SBERT

**Problem:** Current model is English-only. Cross-lingual plagiarism (translate to paraphrase) is undetected.

**Implementation:**
Replace `all-mpnet-base-v2` base model with `paraphrase-multilingual-mpnet-base-v2` or `LaBSE`. These encode 50+ languages into the same embedding space, enabling:
- English ↔ French plagiarism detection
- Back-translation attack resistance

The rest of the pipeline (GBM classifier, feature engineering) requires no changes.

---

## 6. Recommended Prioritized Roadmap

| Sprint | Action | Impact |
|--------|--------|--------|
| Week 1 | Fix threshold inconsistency; create held-out test set; generate real confusion matrix | High |
| Week 1 | Apply canonical preprocessing consistently | Medium |
| Week 2 | Implement SHAP XAI for GBM; replace template advice with attribution-based explanations | High |
| Week 2 | Add input validation; disable debug mode | Medium |
| Week 3 | Replace benchmark with F1/AUC evaluation on PAWS test split | High |
| Week 3 | Add lexical feature augmentation (BLEU, n-gram Jaccard, length ratio) | Medium |
| Week 4 | Implement critical span detection for sentence-level evidence | Medium |
| Week 4 | Validate sub-type thresholds on annotated data | Medium |
| Month 2 | Add cross-encoder reranking for borderline cases | High |
| Month 2 | Document-level detection pipeline | High |
| Month 3 | Multilingual support via LaBSE/multilingual-MPNet | Medium |

---

## 7. Summary Assessment

DePlag has a solid architectural foundation — the SBERT + GBM hybrid approach, CoSENT fine-tuning, probability calibration, and multi-dataset training are all genuinely well-conceived. The system successfully identifies paraphrase plagiarism at a conceptual level that exceeds simple cosine-threshold systems.

However, the system has three fundamental research integrity issues:
1. **The evaluation data (confusion matrix, F1 charts) is fabricated** — not derived from actual model runs. This needs to be rectified before any academic submission.
2. **The threshold pipeline is broken** — a theoretically sound optimization process produces a value (0.05) that is then discarded in favor of an undocumented magic number (0.35).
3. **The benchmark measures raw similarity, not classification performance** — making model comparison claims invalid.

The module structure (`nlp/`, `explainability/`, `evaluation/`) shows ambitious design intent that is largely unimplemented. Filling in these modules with real implementations — particularly SHAP-based attribution, sentence alignment, and proper evaluation — would elevate DePlag from a prototype to a research-grade system.
