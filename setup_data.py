import pandas as pd
import os

FILES = {
    "MSRP": "data/raw/msr_paraphrase_train.txt",   
    "PAWS": "data/raw/paws_train.csv",
    "STS":  "data/raw/sts-train.csv"
}

def load_and_verify():
    print("📂 Checking datasets in 'data/raw'...\n")
    
    # 1. CHECK MSR PARAPHRASE CORPUS
    if os.path.exists(FILES["MSRP"]):
        print("✅ Found MSRP dataset. Loading...")
        try:
            df_msrp = pd.read_csv(FILES["MSRP"], sep='\t', quoting=3, on_bad_lines='skip')
            df_msrp.columns = ['label', 'id1', 'id2', 'text_a', 'text_b']
            print(f"   -> Successfully loaded {len(df_msrp)} pairs.")
            print(f"   -> Example: {df_msrp.iloc[0]['text_a'][:50]}...")
        except Exception as e:
            print(f" Error loading MSRP: {e}")
    else:
        print(f" Missing MSRP file. Expected at: {FILES['MSRP']}")

    print("-" * 30)

    # 2. CHECK PAWS DATASET
    if os.path.exists(FILES["PAWS"]):
        print(" Found PAWS dataset.")
        try:
            df_paws = pd.read_csv(FILES["PAWS"], on_bad_lines='skip')
            print(f" Successfully loaded {len(df_paws)} pairs.")
        except:
            # If CSV fails, try Tab separated (TSV)
            try:
                df_paws = pd.read_csv(FILES["PAWS"], sep='\t', on_bad_lines='skip')
                print(f"Successfully loaded {len(df_paws)} pairs.")
            except Exception as e:
                print(f"Error loading PAWS: {e}")
    else:
        print(f"Missing PAWS file. Expected at: {FILES['PAWS']}")
        
    print("\nDone! If you see failure, check that the files arw in 'data/raw'.")

if __name__ == "__main__":
    load_and_verify()