"""
evaluation/eval_pipeline.py
============================
Proper held-out test-set evaluation for DePlag using the cross-encoder.

Loads the test predictions from models/test_predictions.csv (generated during
training) OR re-runs inference with the cross-encoder on the test data.

Run:
    python evaluation/eval_pipeline.py

Output:
    evaluation/test_results.txt     — full classification report
    evaluation/confusion_matrix.png — actual confusion matrix (use in paper)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nlp.preprocess import normalize

from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, matthews_corrcoef,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(BASE, "models", "results.json")
PREDS_PATH   = os.path.join(BASE, "models", "test_predictions.csv")
OUT_DIR      = os.path.dirname(__file__)

# ── Load threshold ─────────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    results = json.load(f)
threshold = results["threshold"]
print(f"[1/3] Loaded threshold: {threshold:.4f}")

# ── Load test predictions ──────────────────────────────────────────────────────
print("[2/3] Loading test predictions ...")

if os.path.exists(PREDS_PATH):
    df = pd.read_csv(PREDS_PATH)
    print(f"      Loaded {len(df):,} predictions from {PREDS_PATH}")
    y_test  = df["label"].values
    y_proba = df["proba"].values
    y_pred  = (y_proba >= threshold).astype(int)

    if "source" in df.columns:
        sources = df["source"].values
    else:
        sources = None
else:
    print(f"      ⚠️  {PREDS_PATH} not found.")
    print("         Re-run the training notebook to generate test predictions.")
    sys.exit(1)

# ── Metrics ────────────────────────────────────────────────────────────────────
print("[3/3] Computing metrics ...")
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
mcc  = matthews_corrcoef(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)
report_str = classification_report(
    y_test, y_pred, target_names=["Not Plagiarism", "Plagiarism"], digits=4
)

print(f"\n{'='*55}")
print(f"  HELD-OUT TEST SET RESULTS  (n={len(y_test):,})")
print(f"{'='*55}")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  MCC       : {mcc:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Threshold : {threshold:.4f}")
print(f"\n{report_str}")
print(f"  Confusion Matrix:")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

# ── Per-source breakdown ──────────────────────────────────────────────────────
if sources is not None:
    print(f"\n{'='*55}")
    print(f"  PER-SOURCE BREAKDOWN")
    print(f"{'='*55}")
    for src in sorted(set(sources)):
        mask = sources == src
        n_src = mask.sum()
        src_acc = accuracy_score(y_test[mask], y_pred[mask])
        src_f1  = f1_score(y_test[mask], y_pred[mask])
        src_auc = roc_auc_score(y_test[mask], y_proba[mask]) if len(set(y_test[mask])) > 1 else 0.0
        src_mcc = matthews_corrcoef(y_test[mask], y_pred[mask])
        print(f"  {src:6s}  n={n_src:5d}  Acc={src_acc:.4f}  F1={src_f1:.4f}  AUC={src_auc:.4f}  MCC={src_mcc:.4f}")

# ── Save text report ───────────────────────────────────────────────────────────
result_path = os.path.join(OUT_DIR, "test_results.txt")
with open(result_path, "w") as f:
    f.write(f"DePlag — Held-Out Test Set Evaluation (Cross-Encoder)\n")
    f.write(f"Model  : cross-encoder/quora-roberta-base (fine-tuned)\n")
    f.write(f"n_test : {len(y_test)}\n")
    f.write(f"Threshold: {threshold:.4f}\n\n")
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"Precision : {prec:.4f}\n")
    f.write(f"Recall    : {rec:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n")
    f.write(f"MCC       : {mcc:.4f}\n")
    f.write(f"ROC-AUC   : {auc:.4f}\n\n")
    f.write(report_str)
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"  TN={cm[0,0]}  FP={cm[0,1]}\n")
    f.write(f"  FN={cm[1,0]}  TP={cm[1,1]}\n")
print(f"\n[OK] Text report saved -> {result_path}")

# ── Save confusion matrix figure ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Not Plagiarism", "Plagiarism"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Cross-Encoder (Held-Out Test)", fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"[OK] Confusion matrix saved -> {cm_path}")
