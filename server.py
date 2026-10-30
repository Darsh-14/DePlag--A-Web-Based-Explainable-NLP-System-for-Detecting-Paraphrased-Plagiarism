"""
server.py
=========
DePlag — Flask Web Server

Imports inference from app.py (cross-encoder RoBERTa, 90.84% accuracy) and
exposes a /predict endpoint that returns:
  - detection verdict + confidence
  - shared-word visual evidence (NER-enhanced)
  - GWIE counterfactual explanation (pivotal words to change)
  - Integrated Gradients token attribution heatmap data
  - formative advice

Run:
    python server.py
Then open: http://localhost:5000
"""

import sys
import os
import re
import time
from flask import Flask, render_template, request, jsonify

import app as deplag_app
from app import detect_plagiarism, _load_models

sys.path.insert(0, os.path.dirname(__file__))
from explainability.gwie import GWIE
from explainability.integrated_gradients import IntegratedGradients, aggregate_to_words

flask_app = Flask(__name__)

MAX_CHARS = 4000   # ~500 tokens; cross-encoder truncates at 192 tokens
GWIE_COSINE_THRESHOLD = 0.75
GWIE_MAX_WORDS = 80
GWIE_MAX_ITER = 6
IG_STEPS = 8

# ── Resource loading ──────────────────────────────────────────────────────────
nlp      = None   # spaCy pipeline
_gwie    = None   # GWIE explainer
_ig      = None   # Integrated Gradients explainer


def load_resources():
    global nlp, _gwie, _ig

    print("⏳ Loading DePlag models...")
    _load_models()

    sbert = deplag_app._sbert   # already loaded above

    print("   -> Initialising GWIE explainer...")
    _gwie = GWIE(
        sbert,
        alpha=GWIE_COSINE_THRESHOLD,
        max_iter=GWIE_MAX_ITER,
        ig_steps=IG_STEPS,
    )

    print("   -> Initialising Integrated Gradients...")
    _ig = IntegratedGradients(sbert, n_steps=IG_STEPS)

    try:
        import spacy
        print("   -> Loading SpaCy (NER)...")
        nlp = spacy.load("en_core_web_sm")
        print("✅ System Ready!")
    except Exception:
        print(f"⚠️  SpaCy not found — NER disabled. Run: python -m spacy download en_core_web_sm")


load_resources()


# ── Explainability helpers ────────────────────────────────────────────────────

def extract_shared_words(text1: str, text2: str) -> list[str]:
    """Returns shared meaningful tokens + shared NER entities."""
    STOPWORDS = {
        'the','is','in','at','of','on','and','a','an','to','for',
        'it','with','that','this','by','from','as','are','was','were',
        'be','been','being','have','has','had','do','does','did',
        'will','would','could','should','may','might','not','no',
        'or','but','so','if','then','when','where','while','after',
        'before','since','its','their','they','we','he','she','you','i'
    }
    tokens1 = set(re.findall(r'\w+', text1.lower())) - STOPWORDS
    tokens2 = set(re.findall(r'\w+', text2.lower())) - STOPWORDS
    shared  = list(tokens1 & tokens2)

    if nlp:
        ents1 = {e.text.lower() for e in nlp(text1).ents}
        ents2 = {e.text.lower() for e in nlp(text2).ents}
        for phrase in ents1 & ents2:
            shared.extend(phrase.split())
    return list(set(shared))


def build_advice(is_plagiarism: bool, cosine_sim: float,
                 overlap: float, plag_type: str,
                 gwie_result: dict | None) -> str:
    """Combines static rule-based advice with GWIE counterfactual output."""
    if is_plagiarism:
        base = {
            "Verbatim Copy":        "🚨 Verbatim Copy. Rewrite entirely in your own words and cite the source.",
            "Patchwriting":         "⚠️ Patchwriting. Too many original words remain. Read the source, look away, then write from memory.",
            "Structural Plagiarism":"⚠️ Structural Plagiarism. Words changed but sentence logic is identical. Try changing structure (active ↔ passive voice) or add your own analysis.",
            "Paraphrase Plagiarism":"⚠️ Paraphrase Plagiarism. Meaning preserved too closely. Cite the source or add your own unique perspective.",
        }.get(plag_type, "⚠️ Plagiarism detected.")

        if gwie_result and gwie_result.get("counterfactual_set"):
            pivots = ", ".join(f'"{w}"' for w in gwie_result["counterfactual_set"][:5])
            if gwie_result.get("converged"):
                base += f" Key words driving detection: {pivots}. Changing these would likely flip the verdict."
            else:
                base += f" Key words driving detection: {pivots}. These are the strongest content words contributing to the verdict."
        return base
    else:
        if cosine_sim > 0.6:
            return "✅ Pass, but borderline. You are still close to the original meaning. Ensure you aren't just swapping synonyms."
        return "✅ Content appears original."


# ── Routes ────────────────────────────────────────────────────────────────────

@flask_app.route('/')
def home():
    return render_template('index.html')


@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        started = time.perf_counter()
        data  = request.json
        text1 = (data.get('text1') or '').strip()
        text2 = (data.get('text2') or '').strip()

        if not text1 or not text2:
            return jsonify({'error': 'Both texts are required.'}), 400

        if len(text1) > MAX_CHARS or len(text2) > MAX_CHARS:
            return jsonify({'error': f'Text too long. Maximum {MAX_CHARS} characters.'}), 413

        # ── Core inference ─────────────────────────────────────────────────────
        result     = detect_plagiarism(text1, text2)
        cosine_sim = result['cosine_similarity']
        overlap    = result['word_overlap']
        is_plag    = result['is_plagiarism']
        plag_type  = result['plagiarism_type']
        confidence = result['confidence']
        normalized_text2 = None
        e_src = None

        if is_plag and (_ig or _gwie):
            from nlp.preprocess import normalize
            normalized_text2 = normalize(text2)
            e_src = deplag_app._sbert.encode(
                normalize(text1),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        # ── Integrated Gradients attribution ───────────────────────────────────
        ig_data = []
        if _ig and is_plag:
            try:
                token_attrs = _ig.attribute(normalized_text2, e_src)
                word_attrs  = aggregate_to_words(token_attrs, normalized_text2)
                ig_data     = [{"word": w, "score": round(s, 3)} for w, s in word_attrs]
            except Exception as ig_err:
                print(f"⚠️  IG attribution failed: {ig_err}")

        # ── GWIE counterfactual ────────────────────────────────────────────────
        gwie_result = None
        if _gwie and is_plag:
            if len(normalized_text2.split()) <= GWIE_MAX_WORDS:
                try:
                    gwie_result = _gwie.explain(normalized_text2, e_src)
                except Exception as gwie_err:
                    print(f"⚠️  GWIE failed: {gwie_err}")
            else:
                print("⚠️  GWIE skipped: suspicious text too long for interactive explanation.")

        # ── Shared words & advice ─────────────────────────────────────────────
        shared_words = extract_shared_words(text1, text2)
        advice       = build_advice(is_plag, cosine_sim, overlap, plag_type, gwie_result)

        response = {
            "is_plagiarism": is_plag,
            "confidence":    round(confidence * 100, 1),
            "explanation": {
                "cosine_pct":   round(cosine_sim * 100, 1),
                "overlap_pct":  round(overlap * 100, 1),
                "shared_words": shared_words,
                "advice":       advice,
                "plag_type":    plag_type,
            },
            "attribution": ig_data,
            "counterfactual": {
                "pivot_words":         gwie_result["counterfactual_set"]   if gwie_result else [],
                "counterfactual_text": gwie_result["counterfactual_text"]  if gwie_result else "",
                "initial_similarity":  gwie_result["initial_similarity"]   if gwie_result else None,
                "final_similarity":    gwie_result["final_similarity"]     if gwie_result else None,
                "converged":           gwie_result["converged"]            if gwie_result else None,
            }
        }
        print(f"✅ /predict completed in {time.perf_counter() - started:.2f}s")
        return jsonify(response)

    except Exception as e:
        print(f"❌ PREDICTION ERROR: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Internal Error: {str(e)}'}), 500


if __name__ == '__main__':
    flask_app.run(debug=False, port=5000, threaded=True)
