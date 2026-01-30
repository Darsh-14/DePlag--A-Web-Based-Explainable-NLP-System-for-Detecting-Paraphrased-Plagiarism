import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/final_train_data.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "semantic_model.pkl")
SBERT_MODEL_NAME = 'sentence-t5-base'

def train_semantic():
    print(f"🚀 Starting Training with {SBERT_MODEL_NAME}...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: {DATA_PATH} not found. Run merge_all_final.py first.")
        return

    print("   -> Loading dataset...")
    df = pd.read_csv(DATA_PATH).dropna().sample(15000, random_state=42)
    print(f"   -> Loading SBERT Brain ({SBERT_MODEL_NAME})...")
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    print("   -> Generating semantic features (This might take a moment)...")
    embeddings_a = sbert.encode(df['text_a'].tolist(), show_progress_bar=True)
    embeddings_b = sbert.encode(df['text_b'].tolist(), show_progress_bar=True)
    
    # Cosine Sim
    norm_a = np.linalg.norm(embeddings_a, axis=1)
    norm_b = np.linalg.norm(embeddings_b, axis=1)
    norm_a[norm_a == 0] = 1e-10
    norm_b[norm_b == 0] = 1e-10
    
    semantic_sim = np.sum(embeddings_a * embeddings_b, axis=1) / (norm_a * norm_b)
    
    # Word Overlap
    def get_overlap(s1, s2):
        set1 = set(str(s1).lower().split())
        set2 = set(str(s2).lower().split())
        if not set1 or not set2: return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))
        
    overlap_scores = [get_overlap(r['text_a'], r['text_b']) for _, r in df.iterrows()]
    
    # Features and Labels
    X = pd.DataFrame({
        'semantic_sim': semantic_sim,
        'word_overlap': overlap_scores
    })
    y = df['label']

    # Train Classifier
    print(" Training Classifier.")
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    
    #  Save
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"\n SUCCESS. Retrained classifier for {SBERT_MODEL_NAME}.")

if __name__ == "__main__":
    train_semantic()