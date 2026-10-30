# DePlag: A Web-Based Explainable NLP System for Detecting Paraphrased Plagiarism

**Bipin Kumar**<sup>1*</sup>

<sup>1</sup>Electronics and Communication Engineering with specialization in Data Science, SRM University, Chennai, India.

*Corresponding author. E-mail: bipinkumar620013@gmail.com

---

## Abstract

Paraphrased plagiarism — where content is restructured through semantically equivalent substitutions while preserving the original idea — poses a significant challenge to academic integrity. Traditional systems relying on n-gram matching or digital fingerprinting fail to capture such complex transformations. While recent transformer-based models have improved semantic similarity detection, most operate as opaque "black-box" systems providing similarity scores without human-interpretable reasoning. This paper introduces **DePlag**, a Hybrid Explainable Plagiarism Detection System (HEPDS) that combines a domain fine-tuned Sentence-BERT encoder with a calibrated Gradient Boosting classifier, enhanced by a two-layer Explainable AI module. The system classifies specific obfuscation strategies (synonym substitution, structural inversion, patchwriting) through linguistic forensics. A novel **Gradient-Weighted Iterative Erasure (GWIE)** algorithm generates minimal counterfactual explanations, identifying the exact tokens driving a plagiarism decision. Evaluated on 59,226 sentence pairs from MSRP, PAWS, and STS-Benchmark, the system achieves an F1-score of 0.89 on a held-out test set, outperforming all baseline models. DePlag is deployed as a web-based prototype, shifting plagiarism detection from a punitive model toward a transparent, formative tool for academic communities.

**Keywords:** Explainable Plagiarism Detection; Counterfactual Reasoning; NLP; Sentence-BERT; Academic Integrity; Paraphrase Detection; Integrated Gradients; GWIE

---

## 1. Introduction

The rapid commercialization of Large Language Models (LLMs) and Generative AI has fundamentally transformed academic integrity. Plagiarism has evolved beyond simple word-for-word copying into a complex domain dominated by semantic obfuscation and automated text generation [1]. Students can bypass leading detectors including Turnitin through recursive paraphrasing strategies such as synonym replacement and back-translation [4]. This adversarial paraphrasing effectively reduces the variation and predictability patterns that previous detectors rely on [5].

The inadequacy of current tools is well-documented. Weber-Wulff et al. [6] found that detector systems exhibit high false-negative rates on obfuscated text and high false-positive rates on non-native English writing. Without transparent evidence, plagiarism accusations often backfire, creating conflict between institutional policy and student practice [7]. This trust gap is exacerbated when detection tools produce vague percentage scores rather than actionable explanations [8, 9].

To address these limitations, this paper proposes DePlag — a Hybrid Explainable Plagiarism Detection System (HEPDS). Our system combines SBERT-based semantic modelling with a calibrated gradient boosting classifier and introduces: (1) domain-specific fine-tuning of an SBERT encoder on a 59,226-pair combined corpus; (2) a high-dimensional interaction feature vector for a downstream classifier; (3) a dual-stage XAI layer comprising Integrated Gradients token attribution and the novel GWIE algorithm for counterfactual explanation; and (4) a web-based prototype delivering transparent, reasoning-based detection.

---

## 2. Literature Review

**Lexical and Statistical Approaches.** Early plagiarism detection relied on surface-level similarity. Foltýnek et al. [18] categorized methods into character-based (n-gram) and vector-based approaches. Samen [24] refined n-gram analysis for scholarly documents, finding that 3-gram overlaps detect verbatim copying but fail against paraphrased plagiarism. El-Rashidy et al. [19] improved results using Support Vector Machines, though these methods lack the semantic understanding required for GenAI-paraphrased content.

**Hybrid Architectures.** Jieren et al. [23] proposed a two-stage system that filters candidates using lightweight metrics before applying transformer classification — a strategy we adopt to ensure scalability. Tang et al. [22] combined FAISS with sentence embeddings for large-scale retrieval. Setu et al. [24] and Hassan et al. [25] further enhanced Semantic Textual Similarity using web scraping and chaotic perturbation optimization.

**Explainability in Detection.** Luo et al. [26] and Najjar et al. [27] emphasize the necessity of local interpretations for NLP systems. Techniques such as LIME [32] and SHAP [33] have been adapted for feature-level importance scores [28, 29]. Nasim et al. [30] proposed a Trustworthy XAI framework showing that visually highlighting suspicious spans improves user acceptance, supported by Vilone and Longo [31]. Integrated Gradients (Sundararajan et al., 2017) provides the theoretical foundation for our token attribution module.

**Adversarial Robustness.** Macko et al. [34] and Maiti et al. [35] demonstrated that automated paraphrasing tools strongly degrade detection performance. Li et al. [36] and Stefanović et al. [37] identify challenges posed by GenAI data augmentation. Alharthi et al. [38] proposed mitigation methods using ensemble learning [39] to address these issues.

---

## 3. Methodology

Let D_susp denote a suspicious document and D_src a source document. Each document is treated as an ordered sequence of sentences S = {s₁, s₂, ..., sₙ} [18].

The detection task requires a mapping function f: S_susp → S_src that aligns sentences expressing the same meaning despite different wording. We define two complementary similarity measures:

- **Lexical Similarity (L):** Computed via the Jaccard coefficient over word sets, measuring surface-level word overlap.
- **Semantic Similarity (S):** Computed via cosine similarity between dense SBERT embeddings e_i and e_j.

A sentence pair (s_i, s_j) is classified as **paraphrased plagiarism** if:

$$P(s_i, s_j) \iff (L(s_i, s_j) < \tau_{lex}) \wedge (S(s_i, s_j) \geq \tau_{sem})$$
**(Eq. 1)**

where τ_lex = 0.3 (below which n-gram detectors fail) and τ_sem is the F1-optimal decision threshold selected from the validation set ROC curve (Section 3.2). The detection function aggregates sub-type judgments:

$$F(s_i, s_j) = \begin{cases} 1 \text{ (Verbatim)}, & \text{if } L(s_i, s_j) \geq 0.7 \\ 1 \text{ (Patchwriting)}, & \text{if } 0.3 \leq L(s_i, s_j) < 0.7 \\ 1 \text{ (Structural)}, & \text{if } L < 0.3 \text{ and } S \geq 0.85 \\ 1 \text{ (Paraphrase)}, & \text{if } P(s_i, s_j) \text{ else holds} \\ 0 \text{ (Original)}, & \text{otherwise} \end{cases}$$
**(Eq. 2)**

### 3.1. Explainability Requirement

A binary label F ∈ {0,1} is insufficient for academic contexts. We adopt the taxonomy of local interpretations [26] and define a token-level attribution map. For a suspicious sentence s_susp = {w₁, w₂, ..., w_m}, the explainability function maps:

$$E(s_{susp}, s_{src}) \rightarrow \Phi = \{\phi_1, \phi_2, ..., \phi_m\}$$
**(Eq. 3)**

where φ_k is the attribution score of token w_k. A visual explanation mask applies a relevance threshold δ to classify tokens as High Risk (φ_k ≥ δ) or Low Risk (φ_k < δ).

### 3.2. Domain Fine-Tuning of SBERT

We selected **all-mpnet-base-v2** as the base encoder — a 12-layer MPNet model producing 768-dimensional embeddings with a 514-token context window — and fine-tuned it on our combined plagiarism corpus using **CoSENT Loss** [20]. CoSENT directly optimises the cosine similarity ranking of sentence pairs, producing superior embedding geometry for similarity-based classification compared to standard contrastive loss.

**Training configuration:** 3 epochs; batch size 32 (GPU) / 8 (CPU); learning rate 2×10⁻⁵; 10% warm-up ratio; validated with EmbeddingSimilarityEvaluator (Spearman's ρ). The fine-tuned model, `deplag_brain`, achieved a validation Spearman ρ of **0.768** (vs. 0.573 for the base model pre-fine-tuning on this corpus), demonstrating effective domain adaptation.

The base model selection was guided by benchmarking 16 pre-trained transformer architectures on semantic fidelity and inference speed (detailed in Section 5.1). all-mpnet-base-v2 was selected for its favourable accuracy-latency trade-off at the Pareto frontier.

For aligned sentence pairs, the system identifies obfuscation strategies using dependency parsing (via spaCy):

- **Synonym Substitution:** Syntactic structure constant while leaf nodes differ but share WordNet synsets.
- **Syntactic Obfuscation:** Detected via dependency shifts — voice changes where object (dobj) in T_src becomes subject (nsubjpass) in T_susp.
- **Structural Restructuring:** Detected when tree depth changes significantly while the semantic embedding remains stable.

### 3.3. Calibrated Gradient Boosting Classifier

Rather than applying a fixed cosine threshold, we train a downstream **GradientBoostingClassifier** on top of SBERT embeddings. For each text pair (s_i, s_j), we construct a 1537-dimensional interaction feature vector:

$$\mathbf{x} = \left[ |\mathbf{e}_i - \mathbf{e}_j|, \; \mathbf{e}_i \odot \mathbf{e}_j, \; \cos(\mathbf{e}_i, \mathbf{e}_j) \right] \in \mathbb{R}^{1537}$$
**(Eq. 4)**

where |·| is element-wise absolute difference (768 dims), ⊙ is element-wise product (768 dims), and cosine similarity is a scalar (1 dim). This design follows the InferSent interaction feature strategy and gives the classifier fine-grained information about per-dimension agreement and disagreement between embeddings.

Raw GBM probability estimates are often miscalibrated. We therefore wrap the classifier in a **CalibratedClassifierCV** with isotonic regression over 3 stratified folds, producing reliable posterior probabilities P(plagiarism | x). The decision threshold τ is selected from the validation set precision-recall curve by maximising F1:

$$\tau^* = \underset{\tau}{\operatorname{argmax}} \; F_1(\tau) = \underset{\tau}{\operatorname{argmax}} \; \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$
**(Eq. 5)**

### 3.4. Integrated Gradients Token Attribution

To generate the visual mask M (Eq. 3), we compute Integrated Gradients (IG) of the cosine similarity score with respect to input token embeddings. Let S_sim = cos(e_susp, e_src) be the scalar output. The attribution score φ_k for token w_k is:

$$\phi_k = (x_k - \tilde{x}_k) \times \int_{\alpha=0}^{1} \frac{\partial S_{sim}(\tilde{x} + \alpha(x - \tilde{x}))}{\partial x_k} \, d\alpha$$
**(Eq. 6)**

where x̃ is a zero-vector baseline. The integral is approximated by 50 Riemann steps. Token scores are L2-normed per-token (capturing magnitude over the embedding dimension) and L1-normalised to [0, 1] for display. Sub-word tokens are aggregated to whole-word scores by summation before rendering the heatmap.

### 3.5. Counterfactual Generation — GWIE Algorithm

While token attribution identifies *where* the model focuses, counterfactual generation answers: *"What is the minimal set of edits required to revert the plagiarism decision?"* We formulate this as a constrained optimisation:

$$\min_{C \subset T} |C| \quad \text{subject to} \quad S(\text{SBERT}(s'_{susp}), \mathbf{e}_{src}) < \tau^*$$
**(Eq. 7)**

Since exhaustive search over 2^m token subsets is intractable, we introduce **Gradient-Weighted Iterative Erasure (GWIE)**, which leverages IG attribution scores to guide the search greedily.

**Algorithm 1: Gradient-Weighted Iterative Erasure (GWIE)**

```
Require: Suspicious sentence s_susp, Source embedding e_src, Threshold τ*
Ensure:  Counterfactual Set C (minimal pivotal words)

 1: C ← ∅
 2: s_curr ← s_susp
 3: Sim_curr ← cos(SBERT(s_curr), e_src)
 4: while Sim_curr ≥ τ* do
 5:     Φ ← IntegratedGradients(s_curr, e_src)        // 50-step IG (Eq. 6)
 6:     w_pivot ← argmax_{w ∈ s_curr \ C}(Φ)          // highest-attribution token
 7:     s_next ← MaskToken(s_curr, w_pivot)            // ablate pivot
 8:     C ← C ∪ {w_pivot}
 9:     Sim_curr ← cos(SBERT(s_next), e_src)           // re-evaluate
10:     s_curr ← s_next
11: end while
12: return C
```

GWIE exploits the differentiable embedding space: by iteratively removing the token with the highest IG magnitude, it identifies the *pivotal synonyms or restructured verbs* used to evade detection. The resulting set C serves as evidence of the paraphrase strategy and enables formative feedback by showing students the exact words driving the decision. Convergence is guaranteed because token removal is monotonically non-increasing in cosine similarity; a safety cap of 20 iterations prevents degenerate loops.

**Complexity:** O(|C| × m × forward_passes), where m = 50 IG steps and |C| empirically averages 3–8 words on sentence-length inputs.

### 3.6. System Architecture and Implementation

The system follows a decoupled architecture separating offline training from real-time inference.

**Preprocessing (canonical, applied uniformly at all stages):**
```python
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
    return " ".join(text.split())          # collapse whitespace
```
Applying the *same* preprocessing in `rebuild_data.py`, `fine_tune_brain.py`, and inference prevents train/inference distribution mismatch.

**Training pipeline:**
1. `rebuild_data.py` — Merges MSRP + PAWS + STS-Benchmark; removes SNLI (causes false negatives on paraphrase pairs); applies `normalize()`; outputs `data/final_train_data.csv` (59,226 pairs).
2. `fine_tune_brain.py` — Fine-tunes all-mpnet-base-v2 with CoSENT Loss; saves best checkpoint as `models/deplag_brain`.
3. `train_semantic.py` — Loads `deplag_brain`, encodes all pairs, builds 1537-dim features, trains calibrated GBM, selects τ* on validation set; saves `models/semantic_model.pkl` and `models/threshold.txt`.

**Data split:** 70% train / 15% validation / 15% held-out test. Test indices are persisted to `models/test_indices.npy` and never used during training or threshold selection.

**Inference pipeline:** `app.py:detect_plagiarism()` — encodes both texts in a single batched SBERT call, builds features, queries calibrated GBM, applies stored τ*, classifies sub-type via Eq. 2. The server pre-loads all models at startup to minimise per-request latency.

**Web deployment:** Flask microservice; spaCy `en_core_web_sm` for NER-enhanced evidence; GWIE and IG run only when plagiarism is detected (latency trade-off justified by academic use case).

---

## 4. Experimental Setup

### 4.1. Datasets

**Table 1: Dataset Summary**

| Dataset | Pairs | Purpose | Key Challenge |
|---------|-------|---------|---------------|
| MSRP [42] | 5,801 | Semantic training baseline | Natural paraphrase variance; few shared n-grams |
| PAWS [41] | ~49,400 | Adversarial robustness | High lexical overlap with opposite meanings |
| STS-Benchmark [20] | ~4,000 | Continuous similarity grounding | Mapping continuous scores to binary labels |
| **Combined** | **59,226** | **Final training corpus** | **Class balance: 54% negative / 46% positive** |

The STS-Benchmark continuous similarity scores are binarised at threshold 3.0 (out of 5.0) — pairs with score > 3.0 are labelled as paraphrase. SNLI was deliberately excluded after ablation showed it introduced false negatives on paraphrase plagiarism (entailment/contradiction pairs contaminate the paraphrase signal).

**Preprocessing:** `normalize()` applied identically during data construction, fine-tuning, classifier training, and inference.

### 4.2. Evaluation Metrics

Detection performance is evaluated on the held-out test set (8,884 pairs, never seen during training or threshold selection). Metrics: Accuracy, Precision, Recall, F1-score (primary, due to mild class imbalance), and ROC-AUC.

Explainability is evaluated qualitatively: GWIE counterfactual coherence (does removing the pivot words actually reduce similarity below threshold?), and IG attribution alignment with linguistically salient tokens.

---

## 5. Results and Discussion

### 5.1. Model Selection — Benchmarking 16 Transformer Architectures

We benchmarked 16 pre-trained transformer models on semantic fidelity (cosine similarity on paired paraphrase examples) and inference speed. Table 2 shows representative results.

**Table 2: Benchmarking of Transformer Architectures (Bi-Encoder, CPU)**

| Model | Sim. Score | Speed (ms/sent) | Dim |
|-------|-----------|-----------------|-----|
| BAAI/bge-large-en-v1.5 | 0.962 | 1871 | 1024 |
| bert-base-nli-mean-tokens | 0.918 | 332 | 768 |
| sentence-t5-base | 0.888 | 80 | 768 |
| sup-simcse-bert-base | 0.838 | 319 | 768 |
| BAAI/bge-small-en-v1.5 | 0.816 | 94 | 384 |
| **all-mpnet-base-v2 (fine-tuned, Ours)** | **—** | **~320** | **768** |
| gtr-t5-base | 0.765 | 63 | 768 |
| all-MiniLM-L6-v2 | 0.487 | 45 | 384 |
| paraphrase-MiniLM-L3-v2 | 0.351 | 27 | 384 |

Note: The similarity score for our fine-tuned model is not directly comparable via this single-pair benchmark (which measures pre-trained models only). The fine-tuned `deplag_brain` is evaluated on the full held-out test set (Table 3). We selected all-mpnet-base-v2 as the base because: (1) it achieves the best average performance across the STS Benchmark suite among 768-dim models; (2) its 12-layer MPNet architecture with relative attention provides stronger cross-sentence reasoning than distilled models; (3) the fine-tuning improvement (Spearman ρ: 0.573 → 0.768) demonstrates strong domain adaptation capacity.

*[Fig. 1: Model Selection Scatter Plot — latency vs. semantic fidelity. all-mpnet-base-v2 occupies a strong position before fine-tuning; post fine-tuning it achieves superior task-specific performance.]*

### 5.2. Detection Performance — Held-Out Test Set

The calibrated GBM classifier was evaluated on the 8,884-pair held-out test set with the F1-optimal threshold τ* selected from the validation set. Results are presented in Table 3.

**Table 3: Detection Performance on Held-Out Test Set**

| Metric | Value |
|--------|-------|
| Accuracy | 0.89 |
| Precision | 0.89 |
| Recall | 0.88 |
| **F1-Score** | **0.89** |
| ROC-AUC | 0.95 |

*[Fig. 2: Confusion Matrix — held-out test set. Results show balanced precision-recall, confirming the F1-optimal threshold selection produces minimal bias toward either false positive or false negative errors.]*

*[Fig. 3: Performance Comparison — HEPDS (red) vs. baseline TF-IDF and standalone cosine-threshold BERT (gray) across MSRP and PAWS subsets.]*

### 5.3. Ablation Study

A component-wise ablation was conducted on the PAWS adversarial subset (high lexical overlap, different meanings — hardest split). Table 4 reports F1-score when components are systematically removed.

**Table 4: Ablation Results — F1-Score on PAWS Adversarial Subset**

| Configuration | F1-Score | Notes |
|--------------|----------|-------|
| Baseline (TF-IDF cosine) | 0.43 | N-grams blind to synonym substitution |
| Pre-trained SBERT (no fine-tune) | 0.72 | General-purpose embeddings |
| Fine-tuned SBERT (deplag_brain) | 0.81 | Domain adaptation on paraphrase corpus |
| + 1537-dim features + GBM | 0.86 | Interaction features improve boundary |
| **+ Calibration + F1-optimal τ*** | **0.89** | **Full HEPDS system** |

Key observations: (1) N-gram TF-IDF fails on adversarial PAWS (F1=0.43) because surface text changes while meaning is preserved. (2) Domain fine-tuning delivers a 9-point F1 improvement over the pre-trained base model, validating the CoSENT loss approach. (3) The interaction feature vector + GBM adds 5 further points over cosine thresholding alone, confirming that per-dimension agreement/disagreement signals carry classification-relevant information. (4) Probability calibration + principled threshold selection contributes the final 3-point gain.

### 5.4. Hyperparameter Sensitivity

A grid search over the semantic threshold τ_sem ∈ [0.60, 0.95] and lexical threshold τ_lex ∈ [0.20, 0.60] was performed on the validation set to optimise F1-score for the sub-type classifier (Eq. 2). The sensitivity surface (Fig. 4) reveals three zones: (1) **Under-fitted** (τ_sem < 0.70): excessive false positives; (2) **Over-fitted** (τ_lex > 0.50): inability to distinguish paraphrased from verbatim plagiarism; (3) **Optimal**: peak F1 at τ_sem = 0.80, τ_lex = 0.40, requiring deep semantic overlap while tolerating significant surface rewording.

*[Fig. 4: F1-Score Sensitivity Heatmap — performance is highly sensitive to τ_sem. Optimal configuration at τ_sem = 0.80, τ_lex = 0.40.]*

### 5.5. Explainability — GWIE Qualitative Evaluation

**Example:** Source: *"The study of deep learning architectures reveals significant improvements in NLP tasks."*
Suspected: *"Research into neural network frameworks demonstrates a major leap in language processing performance."*

GWIE Counterfactual Set C = {"leap", "demonstrates", "frameworks"}

Removing these three words reduces cosine similarity from 0.83 → 0.61 (below τ*). The set correctly identifies the synonym cluster driving the paraphrase — "leap" ↔ "improvements", "frameworks" ↔ "architectures", "demonstrates" ↔ "reveals" — providing actionable feedback to the student.

*[Fig. 5: Robustness Radar — HEPDS vs. standard BERT across five obfuscation attack types: synonym replacement, passive voice, sentence reordering, word insertion/deletion, back-translation.]*

### 5.6. Limitations

The GWIE algorithm introduces computational overhead. Standard inference operates in O(1) after embedding indexing, whereas GWIE requires iterative gradient queries (empirically 3–8 iterations × 30 IG steps × one forward pass each), increasing explanation latency by approximately 3–5× relative to detection alone. This trade-off is justifiable for academic integrity contexts where transparency outweighs millisecond-level speed. Future work will explore approximate gradient methods to reduce this overhead.

A critical limitation is potential false positives on AI-generated text. LLMs produce text with low perplexity and specific statistical patterns [36]. While the semantic stream detects paraphrased content effectively, it may struggle to differentiate between genuine student writing and LLM output when semantic vectors align closely. Future iterations will incorporate intrinsic detection metrics (burstiness, perplexity) as a supplementary AI authorship detection stream.

The current system operates at sentence-pair level. Document-level detection — segmenting two full documents into sentences and finding maximally similar cross-document sentence pairs — is an important extension for real academic submission checking, planned as future work.

---

## 6. Conclusions

This study addresses the growing challenge of adversarial paraphrasing in academic integrity by introducing HEPDS, a hybrid explainable plagiarism detection framework. Our contributions are:

1. **A domain-specific SBERT encoder** (`deplag_brain`) fine-tuned on a 59,226-pair combined corpus using CoSENT Loss, achieving validation Spearman ρ = 0.768 — a 34% relative improvement over the pre-trained base.

2. **A calibrated GBM classifier** operating on a 1537-dim interaction feature vector with F1-optimal threshold selection, achieving F1 = 0.89 on a fully held-out test set.

3. **Gradient-Weighted Iterative Erasure (GWIE)** — a novel counterfactual explanation algorithm that identifies the minimal token set whose removal flips the plagiarism decision, providing pedagogically actionable feedback.

4. **A complete web-based prototype** combining all components in a Flask microservice with an interactive XAI interface.

By separating semantic understanding (SBERT) from lexical verification (Jaccard) and decision-making (calibrated GBM), HEPDS detects adversarial paraphrasing that bypasses n-gram detectors [24]. The GWIE module transforms plagiarism detection from a surveillance function into a formative learning resource [45], bridging the trust gap between institutional policy and student practice [9].

**Future work:** (1) FAISS-indexed document-level detection for university-scale repositories; (2) GenAI authorship detection via perplexity/burstiness features; (3) user studies with educators and students to quantify the impact of counterfactual feedback on learning outcomes.

---

## Declaration of Competing Interests

The author declares no known competing financial interests or personal relationships that could have influenced the work reported in this paper.

## Funding Declaration

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Ethical Approval

This study does not involve human participants, animals, or sensitive personal data. Ethical approval was not required.

## Data Availability

The datasets used in this study (MSRP, PAWS, STS-Benchmark) are publicly available. All source code and trained model artefacts are available at the repository linked below.

## AI Use Disclosure

The author used AI-based language tools to improve grammar and readability. The scientific content, methodology, experimental design, and conclusions were reviewed and validated by the author.

## Code Availability

The full implementation (data pipeline, fine-tuning, classifier training, GWIE, evaluation, web server) is available on GitHub: [Repository Link]

## Author Contributions

Bipin Kumar: Conceptualization, Methodology, Software Development, Data Collection, Data Analysis, Manuscript Writing, Manuscript Revision.

---

## References

[1] R. Tang, Y. N. Chuang, X. Hu, "The Science of Detecting LLM-Generated Text," Communications of the ACM, 2024.

[2] A. Yusuf et al., "Generative AI and the future of higher education: a threat to academic integrity or reformation?" Int. J. Educ. Technol. High. Educ., 2024.

[3] K. Bittle, O. El-Gayar, "Generative AI and Academic Integrity in Higher Education: A Systematic Review," Information, 2025.

[4] M. Perkins et al., "Simple techniques to bypass GenAI text detectors," Int. J. Educ. Technol. High. Educ., 2024.

[5] A. M. Elkhatat, "Evaluating the authenticity of ChatGPT responses: a study on text-matching capabilities," Int. J. Educ. Integrity, 2023.

[6] D. Weber-Wulff et al., "Testing of detection tools for AI-generated text," Int. J. Educ. Integrity, 2023.

[7] T. Lancaster, "Academic Dishonesty or Academic Integrity? Using NLP to Investigate Positive Integrity," J. Academic Ethics, 2021.

[8] S. Karkoulian et al., "ChatGPT Unveiled: Understanding Perceptions of Academic Integrity," J. Academic Ethics, 2025.

[9] A. S. Nelson et al., "Students' Perceptions of Generative AI Use in Academic Writing," Education Sciences, 2025.

[10] L. Basyoni et al., "Generative AI-Driven Metaverse: The Promises and Challenges," IEEE Open Journal, 2025.

[11] K. Guba, A. Tsivinskaya, "Publish or Perish: The Role of AI in Scientific Misconduct," Higher Education Policy, 2024.

[12] B. Erol et al., "AI in Medical Education: Integrity Risks," Medical Teacher, 2025.

[13] Y. Ibrahim, "Artificial Intelligence in Legal Education," J. Legal Education, 2023.

[14] S. Awasthi et al., "Student Awareness of Plagiarism Policies in the AI Era," J. Academic Ethics, 2024.

[15] Y. Eshet, "The Detection Gap: Why Students Cheat with AI," Computers & Education, 2025.

[16] A. Amirzhanov et al., "Plagiarism types and detection methods: a systematic survey," Frontiers Comp. Sci., 2025.

[17] C. Zhou et al., "Paraphrase Identification With Deep Learning: A Review," IEEE Access, 2025.

[18] T. Foltýnek et al., "Academic plagiarism detection: a systematic literature review," ACM Computing Surveys, 2019.

[19] M. A. El-Rashidy et al., "An effective text plagiarism detection system based on feature selection and SVM," Multimedia Tools Appl., 2024.

[20] N. Reimers, I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.

[21] Y. Samen, "A Semantic Similarity Measure for Scholarly Document Based on n-gram," J. Web Engineering, 2022.

[22] J. Tang et al., "Efficient Plagiarism Detection via Sentence Embeddings and FAISS-based Retrieval," CLEF, 2025.

[23] L. Jieren et al., "Two-Stage Generative Plagiarism Detection: From TF-IDF/Jaccard Filtering to Transformer Classification," CLEF, 2025.

[24] D. M. Setu et al., "A comprehensive strategy for identifying plagiarism in academic submissions," J. Umm Al-Qura Univ., 2025.

[25] E. Hassan et al., "Intelligent text similarity assessment using Roberta with integrated chaotic perturbation," J. Big Data, 2025.

[26] S. Luo et al., "Local Interpretations for Explainable NLP: A Survey," ACM Computing Surveys, 2024.

[27] A. A. Najjar et al., "Leveraging Explainable AI for LLM Text Attribution," Information, 2025.

[28] M. Cesarini et al., "Explainable AI for Text Classification: Lessons from a Comprehensive Evaluation," Cognitive Computation, 2024.

[29] L. Schulte et al., "Studying the explanations using LIME and SHAP," Empirical Soft. Eng., 2024.

[30] M. A. Nasim et al., "Trustworthy XAI and Application," arXiv, 2024.

[31] G. Vilone, L. Longo, "Explainable Artificial Intelligence: a Systematic Review," Information Fusion, 2020.

[32] M. T. Ribeiro et al., "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," KDD, 2016.

[33] S. M. Lundberg, S. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, 2017.

[34] D. Macko et al., "Authorship Obfuscation in Multilingual Machine-Generated Text Detection," EMNLP, 2024.

[35] A. Maiti et al., "Semantic Reconstruction of Adversarial Plagiarism," arXiv, 2025.

[36] Y. Li et al., "Detecting Document-level Paraphrased Machine Generated Content," arXiv, 2024.

[37] P. Stefanović et al., "The influence of Gen-AI tools application for text data augmentation," Scientific Reports, 2025.

[38] R. Alharthi et al., "Responsible Detection and Mitigation of AI-Generated Text," Springer, 2025.

[39] Y. Zhang, "An ensemble deep learning model for author identification," Scientific Reports, 2025.

[40] G. M. Gritsai et al., "Stack More LLMs: Efficient Detection," Doklady Mathematics, 2024.

[41] Y. Zhang et al., "PAWS: Paraphrase Adversaries from Word Scrambling," NAACL, 2019.

[42] W. B. Dolan, C. Brockett, "Microsoft Research Paraphrase Corpus (MSRP)," IWP, 2005.

[43] A. Greiner-Petter et al., "Overview of the Plagiarism Detection Task at PAN 2025," CLEF, 2025.

[44] E. De Santis, A. Martino, A. Rizzi, "Human Versus Machine Intelligence," IEEE TPAMI, 2024.

[45] R. Parvathy et al., "Automated Code Assessment and Feedback: A Comprehensive Model," IEEE Access, 2025.

[46] M. Sundararajan, A. Taly, Q. Yan, "Axiomatic Attribution for Deep Networks (Integrated Gradients)," ICML, 2017.
