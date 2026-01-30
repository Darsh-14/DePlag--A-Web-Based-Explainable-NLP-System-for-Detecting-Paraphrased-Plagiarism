from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
import spacy

app = Flask(__name__)

# --- CONFIG ---
MODEL_PATH = "models/semantic_model.pkl"
SBERT_MODEL_NAME = 'sentence-t5-base'

# --- GLOBAL VARIABLES ---
classifier = None
sbert = None
nlp = None 

def load_resources():
    global classifier, sbert, nlp
    print("⏳ Initializing Ethical Engine...")
    
    # Load Spacy (Entity Recognition)
    try:
        print("   -> Loading Spacy (NER)...")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Spacy model missing. Run: python -m spacy download en_core_web_sm")
        return

    # Load SBERT for Semantic Embeddings
    try:
        print(f"   -> Loading SBERT ({SBERT_MODEL_NAME})...")
        sbert = SentenceTransformer(SBERT_MODEL_NAME)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load SBERT. Details: {e}")
        return

    # Load Classifier Logistic Regression
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                classifier = pickle.load(f)
            print("✅ System Ready!")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Model corrupt. Details: {e}")
    else:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}. Run train_semantic.py!")

# We will Initialize everything on startup
load_resources()

#  WE CAN FIND CAUSAL WORDS & ENTITIES 
def extract_explanation_data(text1, text2):
    # 1. Standard Token Overlap
    tokens1 = set(re.findall(r'\w+', text1.lower()))
    tokens2 = set(re.findall(r'\w+', text2.lower()))
    
    stop_words = {'the', 'is', 'in', 'at', 'of', 'on', 'and', 'a', 'an', 
                  'to', 'for', 'it', 'with', 'that', 'this', 'by', 'from', 'as'}
    
    shared_tokens = list((tokens1 - stop_words).intersection(tokens2 - stop_words))
    
    # Named Entity Overlap using Spacy
    if nlp:
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        ents1 = {e.text.lower() for e in doc1.ents}
        ents2 = {e.text.lower() for e in doc2.ents}
        shared_phrases = list(ents1.intersection(ents2))
        
        entity_tokens = []
        for phrase in shared_phrases:
            entity_tokens.extend(phrase.split())
            
        combined_evidence = list(set(shared_tokens + entity_tokens))
        return combined_evidence
        
    return shared_tokens


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not classifier or not sbert:
        return jsonify({'error': 'Server Error: Models are not loaded.'}), 500

    try:
        data = request.json
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        
        if not text1 or not text2:
            return jsonify({'error': 'Both sentences are required'}), 400

        # FEATURES
        emb1 = sbert.encode([text1])[0]
        emb2 = sbert.encode([text2])[0]
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        semantic_sim = 0.0
        if norm1 > 0 and norm2 > 0:
            semantic_sim = np.dot(emb1, emb2) / (norm1 * norm2)
            
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        overlap = 0.0
        if set1 or set2:
            overlap = len(set1.intersection(set2)) / len(set1.union(set2))

        len1 = len(text1.split())
        len2 = len(text2.split())
        len_diff = abs(len1 - len2) / len1 if len1 > 0 else 0.0

        # PREDICTION
        features = [[semantic_sim, overlap]] 
        prob = classifier.predict_proba(features)[0][1]
        is_plagiarism = prob > 0.5
        
        # EXPLAINABILITY ---
        shared_words = extract_explanation_data(text1, text2)
        
        # Reasoning
        advice = ""
        if is_plagiarism:
            if overlap > 0.3:
                advice = "⚠️ Patchwriting Detected. You are keeping too many original words. Challenge: Read the original, look away, and write the idea entirely from memory."
            # UPDATED THRESHOLD: 0.75 -> 0.85 because T5 scores are consistently higher
            elif semantic_sim > 0.85:
                advice = "⚠️ Structural Plagiarism. You changed the words, but the sentence logic is identical. Try changing the sentence structure (e.g., Active ↔ Passive voice) or combining this with your own analysis."
            else:
                advice = "⚠️ Conceptual Similarity. The ideas are very close. Ensure you cite the source or add your own unique perspective."
        else:
            if semantic_sim > 0.6:
                advice = "✅ Pass, but Borderline. Be careful: You are still very close to the original meaning. Ensure you aren't just swapping synonyms."
            else:
                advice = "✅ Content appears original."

        return jsonify({
            "is_plagiarism": bool(is_plagiarism),
            "confidence": round(prob * 100, 1),
            "explanation": {
                "cosine_pct": round(float(semantic_sim) * 100, 1),
                "overlap_pct": round(float(overlap) * 100, 1),
                "len_diff": round(float(len_diff), 2),
                "shared_words": shared_words, 
                "advice": advice
            }
        })

    except Exception as e:
        print(f"❌ PREDICTION ERROR: {e}")
        return jsonify({'error': f'Internal Logic Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)