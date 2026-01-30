import pandas as pd
import re
import os

FILES = {
    "MSRP": "data/raw/msr_paraphrase_train.txt",   
    "PAWS": "data/raw/paws_train.csv",             
    "STS":  "data/raw/sts-train.jsonl",            
    "SNLI": "data/raw/snli_1.0_train.csv"          
}
OUTPUT_FILE = "data/final_train_data.csv"

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def load_msrp():
    print("   -> Loading MSRP (Plagiarism Base)...")
    if not os.path.exists(FILES["MSRP"]): 
        print(f"      ❌ Missing {FILES['MSRP']}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(FILES["MSRP"], sep='\t', quoting=3, on_bad_lines='skip')
        df = df[['Quality', 'String1', 'String2']]
        df.columns = ['label', 'text_a', 'text_b']
        return df
    except Exception as e:
        print(f"      ⚠️ Error MSRP: {e}")
        return pd.DataFrame()

def load_paws():
    print("   -> Loading PAWS (Tricky Cases)...")
    if not os.path.exists(FILES["PAWS"]): 
        print(f"      ❌ Missing {FILES['PAWS']}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(FILES["PAWS"], on_bad_lines='skip')
        if 'sentence1' not in df.columns:
            df = pd.read_csv(FILES["PAWS"], sep='\t', on_bad_lines='skip')
        
        df = df[['label', 'sentence1', 'sentence2']]
        df.columns = ['label', 'text_a', 'text_b']
        return df
    except Exception as e:
        print(f"      ⚠️ Error PAWS: {e}")
        return pd.DataFrame()

def load_sts():
    print("   -> Loading STS (Semantic Score)...")
    path = FILES["STS"]
    if not os.path.exists(path): 
        print(f"      ❌ Missing {path}")
        return pd.DataFrame()
    try:
        df = pd.read_json(path, lines=True)
        df = df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'score'})
        df['label'] = pd.to_numeric(df['score'], errors='coerce').apply(lambda x: 1 if x > 3.0 else 0)
        
        return df[['label', 'text_a', 'text_b']]
    except Exception as e:
        print(f"      ⚠️ Error STS: {e}")
        return pd.DataFrame()

def load_snli():
    print("   -> Loading SNLI (Logic/Contradiction)...")
    path = FILES["SNLI"]
    if not os.path.exists(path): 
        print(f"      ❌ Missing {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        df = df[['gold_label', 'sentence1', 'sentence2']]
        df = df[df['gold_label'].isin(['entailment', 'contradiction'])]
        
        label_map = {'entailment': 1, 'contradiction': 0}
        df['label'] = df['gold_label'].map(label_map)
        df = df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b'})
        
        return df[['label', 'text_a', 'text_b']].sample(n=50000, random_state=42)
    except Exception as e:
        print(f"      ⚠️ Error SNLI: {e}")
        return pd.DataFrame()

def merge_all():
    print("Starting merge.")
    
    df1 = load_msrp()
    df2 = load_paws()
    df3 = load_sts()
    df4 = load_snli()
    
    # Check sizes
    print(f"\n📊 Dataset Sizes:")
    print(f"   - MSRP: {len(df1)}")
    print(f"   - PAWS: {len(df2)}")
    print(f"   - STS:  {len(df3)}")
    print(f"   - SNLI: {len(df4)}")
    
    # Merge
    full_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    # Clean
    print("\n Cleaning Text...")
    full_df.dropna(subset=['text_a', 'text_b'], inplace=True)
    full_df['text_a'] = full_df['text_a'].apply(clean_text)
    full_df['text_b'] = full_df['text_b'].apply(clean_text)
    
    # Shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n SUCCESS. Combined dataset saved to: {OUTPUT_FILE}")
    print(f" Total Training Pairs: {len(full_df)}")

if __name__ == "__main__":
    merge_all()