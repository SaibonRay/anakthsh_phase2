import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "documents_chunks.csv"
QUERIES_PATH = "queries.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

KEEP_QUERY_TEXT = False

index = faiss.read_index(INDEX_PATH)
chunks_df = pd.read_csv(CHUNKS_PATH)

qdf = pd.read_csv(QUERIES_PATH)
qdf["Text"] = qdf["Text"].fillna("").astype(str)

model = SentenceTransformer(MODEL_NAME)

RUN_NAME = "FAISS"  


def retrieve_one(query_text: str, k: int):
    q = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q, k)

    rows = []
    for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), start=1):
        r = chunks_df.iloc[int(i)]
        rows.append({
            "rank": rank,
            "score": float(score),
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "content": r["chunk_text"],  
        })
    return rows


for k in (20, 30, 50):
    out_rows = []

    for qid, qtext in zip(qdf["ID"].tolist(), qdf["Text"].tolist()):
        hits = retrieve_one(qtext, k)
        for h in hits:
            h["query_id"] = qid
            if KEEP_QUERY_TEXT:
                h["query_text"] = qtext
            out_rows.append(h)

    out_df = pd.DataFrame(out_rows).sort_values(["query_id", "rank"], ascending=[True, True])

    # 1) CSV όπως το θες
    cols = ["query_id", "rank", "doc_id", "content"]
    if KEEP_QUERY_TEXT:
        cols.append("query_text")
    out_df[cols].to_csv(f"results_k{k}.csv", index=False, encoding="utf-8")
    print(f"Saved -> results_k{k}.csv  rows={len(out_df)}")


    with open(f"results_k{k}.trec", "w", encoding="utf-8") as f:
        for _, r in out_df.iterrows():
            f.write(f'{r["query_id"]} Q0 {r["doc_id"]} {int(r["rank"])} {float(r["score"])} {RUN_NAME}\n')

    print(f"Saved -> results_k{k}.trec")
