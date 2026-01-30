import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import os

DATA_PATH = "data/final_train_data.csv"
MODEL_SAVE_PATH = "models/my_fine_tuned_sbert"
BASE_MODEL = 'all-MiniLM-L6-v2'

def fine_tune():
    print("🚀 Loading Data for Fine-Tuning...")
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Data not found.")
        return

    # Load and clean data
    df = pd.read_csv(DATA_PATH).dropna().sample(5000) 
    print(f"   Using {len(df)} pairs for training...")

    train_examples = []
    for _, row in df.iterrows():
        score = float(row['label']) 
        train_examples.append(InputExample(texts=[row['text_a'], row['text_b']], label=score))

    # Create Data Loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Load the Base Model
    print(f"🧠 Loading Base Model ({BASE_MODEL})...")
    model = SentenceTransformer(BASE_MODEL)

    # CosineSimilarityLoss forces the model to give 1.0 for paraphrases and 0.0 for others
    train_loss = losses.CosineSimilarityLoss(model)

    
    print("🏋️ Starting Fine-Tuning (This updates the neural network weights)...")
    print("   (This might take 10-15 minutes on a CPU. Grab a coffee!)")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,  
        warmup_steps=100,
        output_path=MODEL_SAVE_PATH
    )

    print(f"\n Fine-Tuning Completed, Saved new brain to: {MODEL_SAVE_PATH}")
    print("Update app.py now to use this fine-tuned model")

if __name__ == "__main__":
    fine_tune()