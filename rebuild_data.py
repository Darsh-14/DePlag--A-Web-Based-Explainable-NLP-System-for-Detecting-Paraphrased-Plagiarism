"""
rebuild_data.py
===============
Rebuilds final_train_data.csv using MSRP + PAWS + STS only (no SNLI).
Removes the SNLI contamination that was causing false negatives on paraphrase plagiarism.
"""

import pandas as pd
import json
import re
import os

def clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

# ── MSRP ──────────────────────────────────────────────────────────────────────
print("Loading MSRP...")
msrp_path = 'data/raw/msr_paraphrase_train.txt'
if os.path.exists(msrp_path):
    msrp_raw = pd.read_csv(msrp_path, sep='\t', quoting=3, on_bad_lines='skip')
    print(f"  MSRP columns: {msrp_raw.columns.tolist()}")

    # Auto-detect columns
    cols = msrp_raw.columns.tolist()
    label_col = cols[0]   # always first column
    text_a_col = cols[3] if len(cols) > 3 else cols[1]
    text_b_col = cols[4] if len(cols) > 4 else cols[2]

    msrp = msrp_raw[[label_col, text_a_col, text_b_col]].copy()
    msrp.columns = ['label', 'text_a', 'text_b']
    print(f"  Loaded {len(msrp):,} MSRP rows")
else:
    print(f"  ❌ Missing {msrp_path}")
    msrp = pd.DataFrame()

# ── PAWS ──────────────────────────────────────────────────────────────────────
print("Loading PAWS...")
paws_path = 'data/raw/paws_train.csv'
if os.path.exists(paws_path):
    try:
        paws_raw = pd.read_csv(paws_path, on_bad_lines='skip')
        if 'sentence1' not in paws_raw.columns:
            paws_raw = pd.read_csv(paws_path, sep='\t', on_bad_lines='skip')
        paws = paws_raw[['label', 'sentence1', 'sentence2']].copy()
        paws.columns = ['label', 'text_a', 'text_b']
        print(f"  Loaded {len(paws):,} PAWS rows")
    except Exception as e:
        print(f"  ❌ PAWS error: {e}")
        paws = pd.DataFrame()
else:
    print(f"  ❌ Missing {paws_path}")
    paws = pd.DataFrame()

# ── STS ───────────────────────────────────────────────────────────────────────
print("Loading STS...")
sts_path = 'data/raw/sts-train.jsonl'
if os.path.exists(sts_path):
    try:
        sts_rows = [json.loads(l) for l in open(sts_path, encoding='utf-8', errors='ignore')]
        sts_raw  = pd.DataFrame(sts_rows)
        print(f"  STS columns: {sts_raw.columns.tolist()}")

        # Find score and sentence columns
        score_col = 'score' if 'score' in sts_raw.columns else sts_raw.columns[2]
        s1_col    = 'sentence1' if 'sentence1' in sts_raw.columns else sts_raw.columns[0]
        s2_col    = 'sentence2' if 'sentence2' in sts_raw.columns else sts_raw.columns[1]

        sts = sts_raw[[s1_col, s2_col, score_col]].copy()
        sts.columns = ['text_a', 'text_b', 'score']
        sts['label'] = pd.to_numeric(sts['score'], errors='coerce').apply(
            lambda x: 1 if x > 3.0 else 0
        )
        sts = sts[['label', 'text_a', 'text_b']]
        print(f"  Loaded {len(sts):,} STS rows")
    except Exception as e:
        print(f"  ❌ STS error: {e}")
        sts = pd.DataFrame()
else:
    print(f"  ❌ Missing {sts_path}")
    sts = pd.DataFrame()

# ── Merge ─────────────────────────────────────────────────────────────────────
print("\nMerging datasets (NO SNLI)...")
df = pd.concat([msrp, paws, sts], ignore_index=True)
df = df.dropna(subset=['text_a', 'text_b', 'label'])
df = df[(df['text_a'].str.strip() != '') & (df['text_b'].str.strip() != '')]
df['label'] = df['label'].astype(int)
df['text_a'] = df['text_a'].apply(clean)
df['text_b'] = df['text_b'].apply(clean)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Final dataset: {len(df):,} rows")
print(f"   Label distribution:\n{df['label'].value_counts().to_string()}")

df.to_csv('data/final_train_data.csv', index=False)
print("\n✅ Saved to data/final_train_data.csv")
print("   Now run: python train_semantic.py")
