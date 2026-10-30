# DePlag: Hybrid, Explainable Paraphrase-Plagiarism Detection

DePlag decides whether a suspicious text is a paraphrase of a source text, and shows which words
drive that decision. A fine-tuned RoBERTa cross-encoder makes the verdict. An SBERT-based
explanation layer (Integrated Gradients and an iterative-erasure counterfactual called GWIE)
shows the user which words to change.

The system works on **sentence pairs**. Document-level detection is future work.

## How it works

```
source text A, suspicious text B
        |
   normalize()
        |
 RoBERTa cross-encoder -> P(plagiarism)  --> verdict = P >= 0.435
        |
 SBERT (deplag_brain) -> cosine(A, B);  word-set Jaccard(A, B)  --> sub-type label
        |
 if flagged:
     Integrated Gradients on B (w.r.t. cosine to SBERT(A))  --> token attribution
     GWIE iterative erasure on B                            --> counterfactual word set
     spaCy NER                                              --> shared entities
        |
 JSON response + web UI
```

- **Verdict:** `cross-encoder/quora-roberta-base` (125M parameters), fine-tuned on 120,000 pairs
  from MRPC, PAWS, QQP and STS-B. 3 epochs, learning rate 5e-6, max length 192. The decision
  threshold (0.435) maximises F1 on the validation set.
- **Sub-type label** (heuristic, not validated against annotated data): Jaccard > 0.7 is
  "Verbatim Copy", > 0.3 is "Patchwriting", otherwise cosine > 0.85 is "Structural Plagiarism",
  else "Paraphrase Plagiarism". Jaccard does not affect the verdict.
- **GWIE (Gradient-Weighted Iterative Erasure):** repeatedly takes the non-stopword with the
  highest Integrated Gradients score, replaces it with `[ERASED]`, and re-encodes. It stops when
  SBERT cosine to the source falls below 0.75 or the iteration cap is reached, then prunes erased
  words that turn out not to be needed. The server runs it with 6 iterations, 8 IG steps, and
  only for texts of at most 80 words.

## Results (held-out test set, n = 10,000)

| Metric | Value |
|---|---|
| Accuracy | 90.84% |
| Precision / Recall | 0.8928 / 0.9282 |
| F1 | 0.9102 |
| MCC | 0.8174 |
| ROC-AUC | 0.9640 |
| Confusion matrix | TN 4443, FP 557, FN 359, TP 4641 |

Validation F1 was 0.9151 at the same threshold.

| Source | n | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| QQP | 6,825 | 0.919 | 0.917 | 0.973 |
| PAWS | 2,737 | 0.889 | 0.901 | 0.946 |
| STS-B | 355 | 0.896 | 0.897 | 0.968 |
| MRPC | 83 | 0.723 | 0.830 | 0.624 |

Reading these numbers:

- The base checkpoint was already trained on Quora question pairs, and QQP is 68% of the test
  set. On the 3,175 non-QQP pairs, accuracy is about 88.5% (computed from the per-source rows).
- MRPC has only 83 test pairs, and its MCC is 0.09, so treat it as indicative at best.
- The full report is in `evaluation/test_results.txt`, `models/results.json`,
  `models/per_source_metrics.csv` and `evaluation/confusion_matrix.png`. Metrics can be
  recomputed from `models/test_predictions.csv` with `python evaluation/eval_pipeline.py`.

## Explainability: scope and limits

- The explanation layer explains the **SBERT cosine similarity**, not the cross-encoder's
  decision. A GWIE counterfactual shows which words keep two texts close in SBERT space.
- The 0.75 stopping threshold is a fixed setting and is not derived from the 0.435 verdict
  threshold.
- Per-token IG scores are the L2 norm of each token's IG vector, scaled by the maximum, so they
  rank words but do not sum to the change in similarity.
- Faithfulness has been checked qualitatively. There are no deletion/insertion curves or human
  evaluation yet.

## Run it

```bash
git clone https://github.com/Darsh-14/DePlag--A-Web-Based-Explainable-NLP-System-for-Detecting-Paraphrased-Plagiarism.git
cd DePlag--A-Web-Based-Explainable-NLP-System-for-Detecting-Paraphrased-Plagiarism
git lfs pull                       # models/model.safetensors (cross-encoder weights) is stored in Git LFS
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python server.py                   # then open http://localhost:5000
```

`app.py` also has a command-line demo (`python app.py`).

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Cross-encoder inference, SBERT similarity, sub-type heuristic |
| `server.py`, `templates/`, `static/` | Flask app and UI (`/predict`) |
| `explainability/gwie.py`, `integrated_gradients.py` | GWIE and Integrated Gradients |
| `evaluation/` | Held-out evaluation script, report, confusion matrix |
| `models/` | Cross-encoder weights and tokenizer, SBERT (`deplag_brain/`), `results.json`, test predictions |
| `DePlag_*.ipynb`, `DePlag_Colab.py` | Colab/Kaggle training and experiment notebooks |
| `Paper_pdf/` | Paper drafts. These are **not yet updated** to the cross-encoder system |
| `archive/legacy/` | Material from the earlier SBERT + gradient-boosting pipeline |

## Legacy and known gaps

- `train_model.py` (TF-IDF + logistic regression) and `train_semantic.py` (SBERT + gradient
  boosting) belong to the earlier pipeline. They did not produce the reported results.
- The 140,000-pair training/validation/test set (which includes QQP) is not bundled in this
  repository. Training was run in the notebooks above.
- Several placeholder modules are empty: `explainability/counterfactuals.py`, `rationale.py`,
  `critical_spans.py`, `evaluation/metrics.py`, `human_eval.py`, `utils/helpers.py`, and three
  files under `nlp/`.
- Paper manuscript: in preparation.

## Intended use

DePlag is a writing aid for learning attribution, not a punitive tool. A flag needs human
review. False positives are possible (shared topic or boilerplate phrasing), and the system has
not been evaluated on real student submissions.