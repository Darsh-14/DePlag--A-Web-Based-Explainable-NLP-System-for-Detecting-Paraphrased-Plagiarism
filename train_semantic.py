"""
train_semantic.py
=================
Trains a plagiarism classifier on top of the fine-tuned SBERT model (deplag_brain).

Feature engineering:
  - Element-wise |emb_a - emb_b|        (768 dims)
  - Element-wise  emb_a * emb_b         (768 dims)
  - Cosine similarity scalar             (1 dim)
  Total: 1537-dim feature vector per pair

Classifier: GradientBoostingClassifier with threshold optimization via ROC curve.

Outputs:
  - models/semantic_model.pkl   (CalibratedClassifierCV wrapping GBM)
  - models/threshold.txt        (optimal decision threshold)
  - reports/classifier_report.txt
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from nlp.preprocess import normalize

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH        = "data/final_train_data.csv"
BRAIN_MODEL_PATH = "models/deplag_brain"          # output of fine_tune_brain.py
CLASSIFIER_PATH  = "models/semantic_model.pkl"
THRESHOLD_PATH   = "models/threshold.txt"
TEST_INDICES_PATH = "models/test_indices.npy"
REPORT_DIR       = "reports"
BATCH_SIZE       = 256    # for SentenceTransformer.encode()
VAL_SPLIT        = 0.15   # 15% val
TEST_SPLIT       = 0.15   # 15% held-out test — NEVER touched during training
RANDOM_STATE     = 42

# GBM hyperparameters (CPU-friendly; increase n_estimators on GPU machine)
GBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=RANDOM_STATE,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Constructs rich 1537-dim feature vector for each pair."""
    diff   = np.abs(emb_a - emb_b)                          # (N, 768)
    prod   = emb_a * emb_b                                   # (N, 768)
    # Cosine similarity per row
    norms  = np.linalg.norm(emb_a, axis=1, keepdims=True) * \
             np.linalg.norm(emb_b, axis=1, keepdims=True)
    cosine = (np.sum(emb_a * emb_b, axis=1, keepdims=True) /
              np.clip(norms, 1e-8, None))                    # (N, 1)
    return np.hstack([diff, prod, cosine])                   # (N, 1537)

def find_optimal_threshold(y_true, y_proba):
    """Returns the decision threshold that maximises F1 on the validation set.

    Uses the precision-recall curve (more reliable than Youden's J when
    class distributions are slightly imbalanced).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has one fewer element than precisions/recalls
    f1_scores = (2 * precisions[:-1] * recalls[:-1] /
                 np.clip(precisions[:-1] + recalls[:-1], 1e-8, None))
    best_idx = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    best_f1     = float(f1_scores[best_idx])
    print(f"      F1-optimal threshold: {best_thresh:.4f}  (val F1={best_f1:.4f})")
    return best_thresh

# ── Load Data ──────────────────────────────────────────────────────────────────
print("[1/5] Loading dataset ...")
df = pd.read_csv(DATA_PATH).dropna(subset=["text_a", "text_b", "label"])
df = df[(df["text_a"].str.strip() != "") & (df["text_b"].str.strip() != "")]
df["label"] = df["label"].astype(int)

# Apply canonical preprocessing — must match fine_tune_brain.py and inference
df["text_a"] = df["text_a"].apply(normalize)
df["text_b"] = df["text_b"].apply(normalize)
df = df[(df["text_a"] != "") & (df["text_b"] != "")]

print(f"      Pairs: {len(df):,}  |  Label dist: {df['label'].value_counts().to_dict()}")

texts_a = df["text_a"].tolist()
texts_b = df["text_b"].tolist()
labels  = df["label"].values

# ── Encode ─────────────────────────────────────────────────────────────────────
print(f"\n[2/5] Loading SBERT model from '{BRAIN_MODEL_PATH}' ...")
sbert = SentenceTransformer(BRAIN_MODEL_PATH)

print("      Encoding text_a ...")
emb_a = sbert.encode(texts_a, batch_size=BATCH_SIZE, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)
print("      Encoding text_b ...")
emb_b = sbert.encode(texts_b, batch_size=BATCH_SIZE, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)

print("      Building feature matrix ...")
X = build_features(emb_a, emb_b)
y = labels
print(f"      Feature shape: {X.shape}")

# ── 3-Way Split: train / val / held-out test ───────────────────────────────────
print(f"\n[3/5] Splitting data (train=70%, val=15%, test=15%) ...")

# Carve out test first so it is never touched during training or threshold tuning
X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
    X, y, np.arange(len(y)),
    test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_STATE
)
# Save test indices so evaluation/eval_pipeline.py can reproduce the exact split
os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
np.save(TEST_INDICES_PATH, idx_test)

# Split the remaining 85% into train and val
relative_val = VAL_SPLIT / (1.0 - TEST_SPLIT)   # ≈ 0.176 → gives ~15% of total
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=relative_val, stratify=y_trainval, random_state=RANDOM_STATE
)
print(f"      Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

# ── Train Classifier ───────────────────────────────────────────────────────────
print(f"\n[4/5] Training GradientBoostingClassifier (n_estimators={GBM_PARAMS['n_estimators']}) ...")
print("      This may take a few minutes on CPU ...")

gbm  = GradientBoostingClassifier(**GBM_PARAMS)
# Calibrate probabilities using cross-validation on training split
cal  = CalibratedClassifierCV(gbm, cv=StratifiedKFold(n_splits=3), method="isotonic")
cal.fit(X_train, y_train)

# ── Threshold Optimization on Val (never on test) ─────────────────────────────
print(f"\n[5/5] Evaluating — threshold from val, final metrics from held-out test ...")
y_val_proba = cal.predict_proba(X_val)[:, 1]
optimal_thresh = find_optimal_threshold(y_val, y_val_proba)

# ── Val metrics ────────────────────────────────────────────────────────────────
y_val_pred = (y_val_proba >= optimal_thresh).astype(int)
val_acc = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, y_val_proba)
val_f1  = f1_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred,
                                   target_names=["Not Plagiarism", "Plagiarism"])

print(f"\n  [VAL]  Accuracy={val_acc:.4f}  AUC={val_auc:.4f}  F1={val_f1:.4f}")
print(f"  Threshold used: {optimal_thresh:.4f}")
print(f"\n{val_report}")

# ── Held-out test metrics (one-shot, never used for tuning) ────────────────────
y_test_proba = cal.predict_proba(X_test)[:, 1]
y_test_pred  = (y_test_proba >= optimal_thresh).astype(int)
test_acc = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)
test_f1  = f1_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred,
                                    target_names=["Not Plagiarism", "Plagiarism"])

print(f"\n  [TEST] Accuracy={test_acc:.4f}  AUC={test_auc:.4f}  F1={test_f1:.4f}")
print(f"\n{test_report}")

acc, auc, report = test_acc, test_auc, test_report   # use test for saved report

# ── Save Artifacts ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

with open(CLASSIFIER_PATH, "wb") as f:
    pickle.dump(cal, f)
print(f"\n✅ Classifier saved  → {CLASSIFIER_PATH}")

with open(THRESHOLD_PATH, "w") as f:
    f.write(str(optimal_thresh))
print(f"✅ Threshold saved   → {THRESHOLD_PATH}  (value: {optimal_thresh:.4f})")

report_path = os.path.join(REPORT_DIR, "classifier_report.txt")
with open(report_path, "w") as f:
    f.write(f"=== VALIDATION SET ===\n")
    f.write(f"Accuracy : {val_acc:.4f}\n")
    f.write(f"ROC-AUC  : {val_auc:.4f}\n")
    f.write(f"F1       : {val_f1:.4f}\n")
    f.write(f"Threshold: {optimal_thresh:.4f}\n\n")
    f.write(val_report)
    f.write(f"\n=== HELD-OUT TEST SET ===\n")
    f.write(f"Accuracy : {test_acc:.4f}\n")
    f.write(f"ROC-AUC  : {test_auc:.4f}\n")
    f.write(f"F1       : {test_f1:.4f}\n\n")
    f.write(test_report)
print(f"✅ Report saved      → {report_path}")