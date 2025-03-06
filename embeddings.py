import numpy as np
import pandas as pd
import txtai
import os
import torch

# Enable CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set PyTorch threads only if using CPU
if device == "cpu":
    torch.set_num_threads(32)  # Adjust based on CPU cores

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Cache data loading
def load_data():
    path = r'C:\Users\kedar\Documents\Data and Text Mining\AS4\archive\dataset\train.csv'
    df = pd.read_csv(path).dropna()
    return df


# Load dataset
titles = load_data()

# Initialize txtai embeddings with a powerful model
embedding_model = "sentence-transformers/msmarco-distilbert-base-v4"  # Change to another if needed

print(f"Using embedding model: {embedding_model}")
embeddings = txtai.Embeddings({"path": embedding_model, "device": device})

# Check if embeddings exist, if not, create and save
if not os.path.exists("embeddings.db"):
    print("Indexing embeddings for the first time... This may take a while.")

    # Batch processing for faster indexing
    batch_size = 5000

    for i in range(0, len(titles), batch_size):
        batch = titles.iloc[i:min(i + batch_size, len(titles))]  # Ensures last batch is included
        embeddings.index([(j, title) for j, title in enumerate(batch["TITLE"].tolist(), start=i)])
        if i % 5000 == 0:
            print(f"Indexed {i} records...")

    embeddings.save("embeddings.db")
else:
    print("Loading existing embeddings...")
    embeddings.load("embeddings.db")

print("Embeddings are ready!")
