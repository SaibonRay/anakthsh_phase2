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


def retrieve_one(query_text: str, k: int, overfetch: int = 200):
    q = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, idxs = index.search(q, overfetch)

    doc_best = {}

    for score, i in zip(scores[0], idxs[0]):
        r = chunks_df.iloc[int(i)]
        docid = r["doc_id"]

        
        if docid not in doc_best or score > doc_best[docid]["score"]:
            doc_best[docid] = {
                "doc_id": docid,
                "score": float(score),
                "chunk_id": r["chunk_id"],
                "content": r["chunk_text"],
            }

    
    ranked_docs = sorted(
        doc_best.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:k]

    
    rows = []
    for rank, d in enumerate(ranked_docs, start=1):
        rows.append({
            "rank": rank,
            "score": d["score"],
            "doc_id": d["doc_id"],
            "chunk_id": d["chunk_id"],
            "content": d["content"],
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