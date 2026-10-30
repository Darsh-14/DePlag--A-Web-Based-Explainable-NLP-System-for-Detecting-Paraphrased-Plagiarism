# -*- coding: utf-8 -*-
# ---
# DePlag — DeBERTa-v3 Cross-Encoder Training (Google Colab)
# ==========================================================
# Brutal debug version: fixes 7 root causes to reach 93%+ accuracy.
#
# How to use in Colab:
#   1. Upload this file or the .ipynb version
#   2. Set Runtime → Change runtime type → T4 GPU
#   3. Run all cells sequentially
#   4. Download the saved model from Google Drive
# ---

# %% — Cell 1: Setup & GPU Check
import subprocess, sys, os, re, json, math, random
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40.0", "datasets", "accelerate", "scikit-learn",
    "pandas", "safetensors", "sentencepiece", "protobuf"])

import numpy as np
import pandas as pd
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% — Cell 2: Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")
SAVE_DIR = "/content/drive/MyDrive/DePlag_DeBERTa"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Artifacts -> {SAVE_DIR}")

# %% — Cell 3: Configuration
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH = 256
EPOCHS = 4
LR = 1e-5
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 16
GRAD_ACCUM = 2          # effective batch = 32
EVAL_BATCH_SIZE = 64
WARMUP_RATIO = 0.06
MAX_TRAIN_EXAMPLES = 120000
MAX_EVAL_EXAMPLES = 10000

print(json.dumps({
    "model": MODEL_NAME, "max_length": MAX_LENGTH,
    "epochs": EPOCHS, "lr": LR,
    "effective_batch": TRAIN_BATCH_SIZE * GRAD_ACCUM,
}, indent=2))

# %% — Cell 4: Download and Prepare Datasets
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# FIX #1: Whitespace-only normalization — NO punctuation stripping.
# DeBERTa's tokenizer was pre-trained on text WITH punctuation.
def clean_text(text):
    """Minimal cleaning: collapse whitespace only. Preserves punctuation."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())

def norm_key(text):
    """Normalized key for deduplication (lowercase, no punct, collapsed)."""
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())

def as_pair_df(rows, source):
    df = pd.DataFrame(rows)
    df["text_a"] = df["text_a"].map(clean_text)
    df["text_b"] = df["text_b"].map(clean_text)
    df = df[(df.text_a != "") & (df.text_b != "")].copy()
    df["label"] = df["label"].astype(int)
    df = df[df.label.isin([0, 1])].copy()
    df["source"] = source
    return df[["text_a", "text_b", "label", "source"]].reset_index(drop=True)

def split_val_test(df, test_size=0.5):
    val, test = train_test_split(df, test_size=test_size, stratify=df.label, random_state=SEED)
    return val.reset_index(drop=True), test.reset_index(drop=True)

train_parts, val_parts, test_parts = [], [], []

# FIX #3: Load ALL official splits, not just 'train'

# -- MRPC --
print("Loading MRPC ...")
mrpc = load_dataset("glue", "mrpc")
def mrpc_rows(split):
    return [{"text_a": x["sentence1"], "text_b": x["sentence2"], "label": x["label"]} for x in mrpc[split]]
train_parts.append(as_pair_df(mrpc_rows("train"), "mrpc"))
mrpc_val, mrpc_test = split_val_test(as_pair_df(mrpc_rows("validation"), "mrpc"))
val_parts.append(mrpc_val)
test_parts.append(mrpc_test)
print(f"  MRPC: train={len(train_parts[-1])}, val={len(val_parts[-1])}, test={len(test_parts[-1])}")

# -- PAWS --
print("Loading PAWS ...")
paws = load_dataset("paws", "labeled_final")
def paws_rows(split):
    return [{"text_a": x["sentence1"], "text_b": x["sentence2"], "label": x["label"]} for x in paws[split]]
train_parts.append(as_pair_df(paws_rows("train"), "paws"))
val_parts.append(as_pair_df(paws_rows("validation"), "paws"))
test_parts.append(as_pair_df(paws_rows("test"), "paws"))
print(f"  PAWS: train={len(train_parts[-1])}, val={len(val_parts[-1])}, test={len(test_parts[-1])}")

# -- QQP --
print("Loading QQP ...")
qqp = load_dataset("glue", "qqp")
def qqp_rows(split):
    rows = []
    for x in qqp[split]:
        if x["question1"] and x["question2"]:
            rows.append({"text_a": x["question1"], "text_b": x["question2"], "label": x["label"]})
    return rows
train_parts.append(as_pair_df(qqp_rows("train"), "qqp"))
qqp_val, qqp_test = split_val_test(as_pair_df(qqp_rows("validation"), "qqp"))
val_parts.append(qqp_val)
test_parts.append(qqp_test)
print(f"  QQP: train={len(train_parts[-1])}, val={len(val_parts[-1])}, test={len(test_parts[-1])}")

# -- STS-B (FIX #5: drop ambiguous 2.5-3.5 scores) --
print("Loading STS-B ...")
try:
    stsb = load_dataset("mteb/stsbenchmark-sts")
    score_key, s1_key, s2_key = "score", "sentence1", "sentence2"
except Exception:
    stsb = load_dataset("stsb_multi_mt", "en")
    score_key, s1_key, s2_key = "similarity_score", "sentence1", "sentence2"

def stsb_rows_clean(split):
    rows = []
    for x in stsb[split]:
        score = float(x[score_key])
        if score >= 3.5:
            label = 1
        elif score <= 2.5:
            label = 0
        else:
            continue  # FIX #5: DROP ambiguous scores
        rows.append({"text_a": x[s1_key], "text_b": x[s2_key], "label": label})
    return rows

train_parts.append(as_pair_df(stsb_rows_clean("train"), "stsb"))
val_parts.append(as_pair_df(stsb_rows_clean("validation"), "stsb"))
test_parts.append(as_pair_df(stsb_rows_clean("test"), "stsb"))
print(f"  STS-B: train={len(train_parts[-1])}, val={len(val_parts[-1])}, test={len(test_parts[-1])}")

# Combine
train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"\nRaw sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# %% — Cell 5: Dedupe, Leakage Prevention, Balance
# FIX #7: Strict cross-split leakage prevention
def add_pair_key(df):
    a = df.text_a.map(norm_key)
    b = df.text_b.map(norm_key)
    df = df.copy()
    df["pair_key"] = np.minimum(a, b) + " ||| " + np.maximum(a, b)
    return df.drop_duplicates("pair_key").reset_index(drop=True)

def balance_binary(df, max_examples=None, name="split"):
    counts = df.label.value_counts().to_dict()
    if 0 not in counts or 1 not in counts:
        return df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    per_class = min(counts[0], counts[1])
    if max_examples is not None:
        per_class = min(per_class, max_examples // 2)
    out = pd.concat([
        df[df.label == 0].sample(per_class, random_state=SEED),
        df[df.label == 1].sample(per_class, random_state=SEED),
    ], ignore_index=True)
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Deduplicate within each split
train_df = add_pair_key(train_df)
val_df = add_pair_key(val_df)
test_df = add_pair_key(test_df)

# FIX #2 & #7: Remove val/test pairs that appear in train (leakage prevention)
train_keys = set(train_df.pair_key)
val_df = val_df[~val_df.pair_key.isin(train_keys)].reset_index(drop=True)
val_keys = set(val_df.pair_key)
test_df = test_df[~test_df.pair_key.isin(train_keys | val_keys)].reset_index(drop=True)

# FIX #4: Balance classes (caps QQP domination)
train_df = balance_binary(train_df, MAX_TRAIN_EXAMPLES, "train")
val_df = balance_binary(val_df, MAX_EVAL_EXAMPLES, "validation")
test_df = balance_binary(test_df, MAX_EVAL_EXAMPLES, "test")

for name, d in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(f"\n{name}: {len(d)} pairs")
    print(d.source.value_counts())
    print(d.label.value_counts(normalize=True))

# %% — Cell 6: Tokenize
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def to_dataset(df):
    return Dataset.from_pandas(
        df[["text_a", "text_b", "label", "source"]].reset_index(drop=True),
        preserve_index=False,
    )

raw_ds = DatasetDict({
    "train": to_dataset(train_df),
    "validation": to_dataset(val_df),
    "test": to_dataset(test_df),
})

def tokenize_batch(batch):
    return tokenizer(batch["text_a"], batch["text_b"], truncation=True, max_length=MAX_LENGTH)

encoded = raw_ds.map(tokenize_batch, batched=True, remove_columns=["text_a", "text_b", "source"])
encoded = encoded.rename_column("label", "labels")
print(encoded)

# %% — Cell 7: Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix,
)

def logits_to_proba(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    if logits.ndim == 1 or logits.shape[1] == 1:
        z = np.clip(logits.reshape(-1), -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -50, 50))
    return exp_logits[:, 1] / np.clip(exp_logits.sum(axis=1), 1e-12, None)

def metric_dict(y_true, proba, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else 0.0,
        "threshold": float(threshold),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    proba = logits_to_proba(logits)
    if np.isnan(proba).any():
        proba = np.nan_to_num(proba, nan=0.5)
    return metric_dict(labels, proba, 0.5)

def threshold_sweep(y_true, proba):
    rows = []
    for t in np.linspace(0.05, 0.95, 181):
        rows.append(metric_dict(y_true, proba, t))
    return pd.DataFrame(rows).sort_values(["accuracy", "mcc"], ascending=False)

# %% — Cell 8: Train Cross-Encoder
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

# FIX #6: Proper model initialization
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True,
)
model.config.use_cache = False  # FIX #6: Required for gradient checkpointing

collator = DataCollatorWithPadding(tokenizer=tokenizer)  # FIX #6: Proper padding

# Compute warmup steps
steps_per_epoch = max(1, math.ceil(len(encoded["train"]) / TRAIN_BATCH_SIZE))
warmup_steps = int(steps_per_epoch * EPOCHS * WARMUP_RATIO)
major_cc = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
USE_BF16 = bool(torch.cuda.is_available() and major_cc >= 8)
USE_FP16 = bool(torch.cuda.is_available() and not USE_BF16)

print(json.dumps({
    "steps_per_epoch": steps_per_epoch,
    "warmup_steps": warmup_steps,
    "fp16": USE_FP16,
    "bf16": USE_BF16,
}, indent=2))

# FIX #6: Corrected hyperparameters
args = TrainingArguments(
    output_dir="/content/deberta_checkpoints",
    learning_rate=LR,                        # 1e-5 (was 2e-5)
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,  # effective batch = 32 (was 64)
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,               # 0.06 (was 0.1)
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    label_smoothing_factor=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    logging_nan_inf_filter=True,
    save_total_limit=3,
    fp16=USE_FP16,
    bf16=USE_BF16,
    gradient_checkpointing=True,
    report_to="none",
    seed=SEED,
    dataloader_num_workers=2,
)

# NaN guard callback
class NaNGuard(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            if math.isnan(logs["loss"]) or math.isinf(logs["loss"]):
                print(f"FATAL: NaN/Inf loss at step {state.global_step}")
                control.should_training_stop = True

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2), NaNGuard()],
)

print("Starting training ...")
trainer.train()
print("Training complete!")

# %% — Cell 9: Threshold Tuning & Test Evaluation
def predict_proba(split_name):
    pred = trainer.predict(encoded[split_name])
    labels = np.asarray(pred.label_ids).astype(int)
    proba = logits_to_proba(pred.predictions)
    print(split_name, {
        "n": len(labels), "finite": bool(np.isfinite(proba).all()),
        "min": float(proba.min()), "max": float(proba.max()),
    })
    return labels, proba

y_val, val_proba = predict_proba("validation")
y_test, test_proba = predict_proba("test")

val_sweep = threshold_sweep(y_val, val_proba)
print(val_sweep.head(15).to_string())
best_threshold = float(val_sweep.iloc[0].threshold)
test_pred = (test_proba >= best_threshold).astype(int)

val_metrics = metric_dict(y_val, val_proba, best_threshold)
test_metrics = metric_dict(y_test, test_proba, best_threshold)

print(f"\nBest threshold: {best_threshold}")
print(f"Validation: {json.dumps(val_metrics, indent=2)}")
print(f"Test: {json.dumps(test_metrics, indent=2)}")
print(classification_report(y_test, test_pred,
    target_names=["Not Plagiarism", "Plagiarism"], digits=4))

# Per-source breakdown
print("\nPER-SOURCE TEST BREAKDOWN:")
source_rows = []
for src in sorted(test_df.source.unique()):
    idx = test_df.source.values == src
    m = metric_dict(y_test[idx], test_proba[idx], best_threshold)
    m["source"] = src
    m["n"] = int(idx.sum())
    source_rows.append(m)
    print(f"  {src:6s}  n={m['n']:5d}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")

# %% — Cell 10: Save Model & Artifacts
model_save_path = os.path.join(SAVE_DIR, "model")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved -> {model_save_path}")

results = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "threshold": best_threshold,
    "train_examples": len(train_df),
    "validation_examples": len(val_df),
    "test_examples": len(test_df),
    "validation": val_metrics,
    "test": test_metrics,
    "per_source": source_rows,
    "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
}
with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved -> {SAVE_DIR}/results.json")

preds_df = pd.DataFrame({
    "label": y_test, "proba": test_proba, "pred": test_pred,
    "source": test_df["source"].values if "source" in test_df.columns else "unknown",
})
preds_df.to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)
print(f"Predictions saved -> {SAVE_DIR}/test_predictions.csv")

# %% — Cell 11: Deployment Instructions
print("""
========================================================
  DEPLOYMENT INSTRUCTIONS
========================================================

  1. Download from Google Drive:
     DePlag_DeBERTa/model/  (all files)
     DePlag_DeBERTa/results.json
     DePlag_DeBERTa/test_predictions.csv

  2. Copy to local project:
     cp DePlag_DeBERTa/model/* models/
     cp DePlag_DeBERTa/results.json models/
     cp DePlag_DeBERTa/test_predictions.csv models/

  3. Run: python server.py

  IMPORTANT: This model was trained WITHOUT punctuation
  stripping. Make sure nlp/preprocess.py normalize()
  uses whitespace-only cleaning (no re.sub punctuation).
========================================================
""")
