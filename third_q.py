import numpy as np
import pandas as pd
import faiss

EMB_PATH = "documents_chunk_embeddings.npy"
META_PATH = "documents_chunk_embeddings_meta.csv"
CHUNKS_PATH = "documents_chunks.csv"

INDEX_PATH = "faiss_index.bin"

embeddings = np.load(EMB_PATH).astype(np.float32) 
meta = pd.read_csv(META_PATH)
chunks_df = pd.read_csv(CHUNKS_PATH)

N, dim = embeddings.shape
print("Embeddings:", embeddings.shape)

USE_IP = True

if USE_IP:
    index = faiss.IndexFlatIP(dim)
else:
    index = faiss.IndexFlatL2(dim)

index.add(embeddings)
print("FAISS total vectors:", index.ntotal)

faiss.write_index(index, INDEX_PATH)
print("Saved index: ", INDEX_PATH)


