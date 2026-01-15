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

print("Embeddings shape:", embeddings.shape)  


np.save("documents_chunk_embeddings.npy", embeddings)

meta = chunks_df[["doc_id", "chunk_id"]].copy()
meta.to_csv("documents_chunk_embeddings_meta.csv", index=False)

print("Saved: documents_chunk_embeddings.npy + documents_chunk_embeddings_meta.csv")

def embed_queries(queries, model=model):
    if isinstance(queries, str):
        queries = [queries]
    q_emb = model.encode(
        queries,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    return q_emb

