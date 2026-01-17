import numpy as np
import pandas as pd
import faiss

EMB_PATH = "documents_chunk_embeddings.npy"
INDEX_PATH = "faiss_index.bin"

embeddings = np.load(EMB_PATH).astype(np.float32) 
N, dim = embeddings.shape
print("Embeddings:", embeddings.shape)

index = faiss.IndexFlatIP(dim)

index.add(embeddings)
print("FAISS total vectors:", index.ntotal)

faiss.write_index(index, INDEX_PATH)
print("Saved index: ", INDEX_PATH)


