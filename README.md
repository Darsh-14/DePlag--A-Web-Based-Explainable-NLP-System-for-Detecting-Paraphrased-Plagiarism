DePlag: Hybrid & Explainable Paraphrase Detection
DePlag is a next-generation academic integrity framework that moves beyond simple string matching. By fusing the semantic depth of Sentence-T5 Transformers with traditional lexical overlap metrics, DePlag identifies sophisticated "Adversarial Paraphrasing" while providing students and educators with transparent, actionable feedback.

🚀 The Core Philosophy
Traditional plagiarism detectors are often "Black Boxes" that provide a percentage score without context. This leads to a Trust Gap between students and institutions. DePlag solves this by:

Detecting Meaning, Not Just Words: Using sentence-t5-base to capture conceptual similarity even when every word has been changed.

Explainable AI (XAI): Highlighting specific shared entities and linguistic patterns using spaCy NER.

Pedagogical Feedback: An "Ethical Advisory" system that coaches users on whether they have committed "Patchwriting," "Structural Plagiarism," or "Conceptual Overlap."

🛠️ Technical Architecture
The system utilizes a dual-stream architecture to ensure both high precision and forensic transparency:

The Semantic Brain: Utilizes a Sentence-T5 model to map sentences into a 768-dimensional vector space. Similarity is calculated using a custom NumPy-optimized Cosine Similarity implementation.

The Lexical Stream: A high-speed Jaccard Similarity engine that tracks word-for-word overlap to identify "copy-paste" or "near-verbatim" theft.

The Neural Classifier: A Logistic Regression model trained on 15,000 samples (from MSRP and PAWS benchmarks) that learns to weight the semantic and lexical signals to make a final "Plagiarism" determination.

Forensic NER: Integrated spaCy pipelines to extract and compare Named Entities, ensuring that critical technical terms and proper nouns are tracked across versions.

📦 Installation & Setup
1. Clone and Install
Bash

git clone https://github.com/yourusername/DePlag.git
cd DePlag
pip install -r requirements.txt
2. Download Linguistic Models
Bash

python -m spacy download en_core_web_sm
3. Initialize the Engine
Before running the app, you must train the classifier head using the provided dataset:

Bash

python train_semantic.py
python app.py
🧩 Key Features (As seen in app.py)
Real-time Inference: Powered by a Flask microservice with pre-loaded models for low-latency scoring.

Entity Evidence: Extracts and displays shared technical terms to prove similarity.

Ethical Advisory Logic: * Score > 0.85 (Semantic): Flags "Structural Plagiarism."

Overlap > 0.3 (Lexical): Flags "Patchwriting."

Lower Scores: Distinguishes between "Conceptual Similarity" and "Original Content."

🛡️ Privacy & Ethics
Security: This repository includes a .gitignore to protect sensitive files like kaggle.json and local .pkl models.

Goal: DePlag is intended as a writing assistant to help students learn proper attribution, rather than a purely punitive tool.