"""
fine_tune_brain.py
==================
Fine-tunes an SBERT model on DePlag's combined dataset (MSRP + PAWS + STS + SNLI).
Uses the full dataset, GPU if available, and saves a model ready for app.py inference.

Output: models/deplag_brain/
"""

import os
import sys
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from nlp.preprocess import normalize

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH    = "data/final_train_data.csv"
OUTPUT_DIR   = "models/deplag_brain"
BASE_MODEL   = "all-mpnet-base-v2"      # 768-dim; swap to 'all-MiniLM-L6-v2' if RAM < 4 GB
EPOCHS       = 3
BATCH_SIZE   = 32 if torch.cuda.is_available() else 8
WARMUP_RATIO = 0.1
LR           = 2e-5
VAL_SPLIT    = 0.1
RANDOM_STATE = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device.upper()}  |  Base model: {BASE_MODEL}")
print(f"[INFO] Batch size: {BATCH_SIZE}  |  Epochs: {EPOCHS}")

# ── Load Data ──────────────────────────────────────────────────────────────────
print(f"\n[1/4] Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text_a", "text_b", "label"])
df["label"] = df["label"].astype(float)

# Filter out any rows with empty strings after cleaning
df = df[(df["text_a"].str.strip() != "") & (df["text_b"].str.strip() != "")]
print(f"      Total usable pairs: {len(df):,}")
print(f"      Label distribution:\n{df['label'].value_counts().to_string()}")

# ── Train / Val Split ──────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=df["label"].astype(int)
)
print(f"\n[2/4] Train: {len(train_df):,}  |  Val: {len(val_df):,}")

# Convert to InputExample (CoSENTLoss expects float labels in [0, 1])
# normalize() applied here must match what rebuild_data.py and inference use.
train_examples = [
    InputExample(texts=[normalize(row.text_a), normalize(row.text_b)], label=float(row.label))
    for row in train_df.itertuples(index=False)
]
val_examples = [
    InputExample(texts=[normalize(row.text_a), normalize(row.text_b)], label=float(row.label))
    for row in val_df.itertuples(index=False)
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# Evaluator: measures Spearman correlation on validation set
val_sentences1 = [ex.texts[0] for ex in val_examples]
val_sentences2 = [ex.texts[1] for ex in val_examples]
val_scores     = [ex.label     for ex in val_examples]
evaluator = EmbeddingSimilarityEvaluator(
    val_sentences1, val_sentences2, val_scores,
    name="deplag_val"
)

# ── Model & Loss ───────────────────────────────────────────────────────────────
print(f"\n[3/4] Loading base model '{BASE_MODEL}' ...")
model = SentenceTransformer(BASE_MODEL, device=device)
train_loss = losses.CoSENTLoss(model)

warmup_steps = int(len(train_dataloader) * EPOCHS * WARMUP_RATIO)
print(f"      Warmup steps: {warmup_steps}")

# ── Training ───────────────────────────────────────────────────────────────────
print(f"\n[4/4] Training for {EPOCHS} epoch(s) ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LR},
    output_path=OUTPUT_DIR,
    save_best_model=True,
    show_progress_bar=True,
)

print(f"\n✅ Fine-tuned model saved to: {OUTPUT_DIR}")
print("   Next step: run train_semantic.py to train the classifier on top of this model.")