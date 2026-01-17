import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


chunks_df = pd.read_csv("documents_chunks.csv")
texts = chunks_df["chunk_text"].fillna("").astype(str).tolist()


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

print("Embedding dim =", model.get_sentence_embedding_dimension())

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)


np.save("documents_chunk_embeddings.npy", embeddings)

print("Saved: documents_chunk_embeddings.npy")


