import time
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import warnings


warnings.filterwarnings("ignore")

text1 = "The stock market crashed yesterday, causing panic among investors."
text2 = "Wall Street plummeted in value last night, leading to widespread fear in the financial sector."

# Model list
model_names = [
    'sentence-t5-base',                      
    'gtr-t5-base',                           
    'BAAI/bge-small-en-v1.5',                
    'BAAI/bge-base-en-v1.5',                 
    'Alibaba-NLP/gte-base-en-v1.5',          
    'mixedbread-ai/mxbai-embed-large-v1',    
    'all-mpnet-base-v2',                     
    'all-distilroberta-v1',                 
    'all-MiniLM-L6-v2',                      
    'all-MiniLM-L12-v2',                     
    'paraphrase-MiniLM-L3-v2',               
    'sentence-transformers/allenai-specter',             
    'princeton-nlp/sup-simcse-bert-base-uncased',        
    'sentence-transformers/quora-distilbert-base',       
    'sentence-transformers/stsb-roberta-base-v2',        
    'bert-base-nli-mean-tokens'              
]

def run_benchmark():
    results = []
    print(f"🚀 Starting BENCHMARK on {len(model_names)} Models...")
    print(f"   Test Pair:\n   A: {text1}\n   B: {text2}\n")

    for i, name in enumerate(model_names):
        print(f"   [{i+1}/{len(model_names)}] Testing: {name}...")
        try:
            model = SentenceTransformer(name, trust_remote_code=True)
            start_time = time.time()
            embeddings = model.encode([text1, text2])
            end_time = time.time()
            
            inference_time_ms = ((end_time - start_time) / 2) * 1000
            
            sim_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            try:
                size_proxy = model.get_sentence_embedding_dimension()
            except:
                size_proxy = "N/A"

            results.append({
                "Model Name": name,
                "Similarity Score": round(sim_score, 4),
                "Speed (ms/sent)": round(inference_time_ms, 2),
                "Embedding Dim": size_proxy
            })
            
            del model
            
        except Exception as e:
            print(f" Failed to load {name}: {e}")

    # --- REPORT ---
    df = pd.DataFrame(results)
    df = df.sort_values(by="Similarity Score", ascending=False)
    
    print("\n" + "="*80)
    print("FInal Benchmark Results")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    df.to_csv("benchmark_results_16.csv", index=False)
    print("\n✅ Saved to 'benchmark_results_16.csv'")

if __name__ == "__main__":
    run_benchmark()