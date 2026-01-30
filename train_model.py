import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
DATA_PATH = "data/final_train_data.csv"
MODEL_DIR = "models"

def get_handcrafted_features(df):
    """
    Creates 'Explainable' features:
    1. Length Difference
    2. Word Overlap Ratio
    """
    print("Generating manual features")
    
    # Length Difference
    len_a = df['text_a'].str.split().str.len()
    len_b = df['text_b'].str.split().str.len()
    len_diff = abs(len_a - len_b) / (len_a + 1) # +1 to avoid div by zero
    
    # Jaccard Similarity 
    def jaccard(row):
        set_a = set(str(row['text_a']).split())
        set_b = set(str(row['text_b']).split())
        if len(set_a) == 0 or len(set_b) == 0: return 0.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    
    overlap = df.apply(jaccard, axis=1)
    
    return pd.DataFrame({'len_diff': len_diff, 'word_overlap': overlap})

def train():
    print("🚀 Loading Data...")
    if not os.path.exists(DATA_PATH):
        print("❌ Error: final_train_data.csv not found!")
        return

    df = pd.read_csv(DATA_PATH).dropna() 
    print(f"   Training on {len(df)} pairs...")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[['text_a', 'text_b']], df['label'], test_size=0.2, random_state=42
    )

    print("🧠 Vectorizing Text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=3000) 
    
    all_text = pd.concat([X_train_raw['text_a'], X_train_raw['text_b']])
    vectorizer.fit(all_text)
    
    # Transform Train Data
    tfidf_a = vectorizer.transform(X_train_raw['text_a'])
    tfidf_b = vectorizer.transform(X_train_raw['text_b'])
    
    # 3. Compute Cosine Similarity 
    print("📐 Calculating Cosine Similarity...")
    dot_product = np.multiply(tfidf_a.toarray(), tfidf_b.toarray()).sum(axis=1)
    
    manual_features = get_handcrafted_features(X_train_raw)
    
    X_train_final = pd.DataFrame({
        'cosine_sim': dot_product,
        'word_overlap': manual_features['word_overlap'].values,
        'len_diff': manual_features['len_diff'].values
    })
    
    print(" Training Classifier using Logistic Regression")
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_final, y_train)
    
    # --- EVALUATION ---
    print("\n📝 Evaluating...")
    test_tfidf_a = vectorizer.transform(X_test_raw['text_a'])
    test_tfidf_b = vectorizer.transform(X_test_raw['text_b'])
    test_dot = np.multiply(test_tfidf_a.toarray(), test_tfidf_b.toarray()).sum(axis=1)
    test_manual = get_handcrafted_features(X_test_raw)
    
    X_test_final = pd.DataFrame({
        'cosine_sim': test_dot,
        'word_overlap': test_manual['word_overlap'].values,
        'len_diff': test_manual['len_diff'].values
    })
    
    preds = model.predict(X_test_final)
    print(f"  Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    # --- SAVE ARTIFACTS ---
    print("Saving Model & Vectorizer")
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    with open(f"{MODEL_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Done. Model saved to 'models/'")

if __name__ == "__main__":
    train()