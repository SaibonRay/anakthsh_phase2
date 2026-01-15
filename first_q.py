import re
import pandas as pd
from transformers import AutoTokenizer


df = pd.read_csv("documents.csv")  


ID_COL = "ID"     
BODY_COL = "Text"  

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

# Αν υπάρχει BODY_COL, το χρησιμοποιούμε. Αλλιώς ψάχνουμε μια πιθανή στήλη κειμένου.
if BODY_COL in df.columns:
    df["full_text"] = df[BODY_COL].apply(safe_str)
else:
    text_candidates = [c for c in df.columns if c.lower() in ("text", "body", "content", "document")]
    if not text_candidates:
        raise ValueError("Δεν βρέθηκε προφανής στήλη κειμένου στο documents.csv")
    df["full_text"] = df[text_candidates[0]].apply(safe_str)


_html = re.compile(r"<[^>]+>")
_ctrl = re.compile(r"[\x00-\x1F\x7F]")
_ws = re.compile(r"\s+")

def clean_text(t: str) -> str:
    t = _html.sub(" ", t)           
    t = _ctrl.sub(" ", t)            
    t = t.replace("\u00a0", " ")     
    t = _ws.sub(" ", t).strip()     
    return t

df["full_text"] = df["full_text"].map(clean_text)


df = df[df["full_text"].str.len() > 0].copy()
if ID_COL in df.columns:
    df = df.drop_duplicates(subset=[ID_COL])
df = df.drop_duplicates(subset=["full_text"])


tokenizer_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

MAX_TOKENS = 320     
OVERLAP = 64        

rows = []
for i, r in df.iterrows():
    doc_id = r[ID_COL] if ID_COL in df.columns else i
    text = r["full_text"]

    
    enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_tensors=None)
    input_ids = enc["input_ids"]

    start = 0
    chunk_id = 0
    while start < len(input_ids):
        end = min(start + MAX_TOKENS, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        if chunk_text:
            rows.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text
            })

        if end == len(input_ids):
            break
        start = end - OVERLAP
        chunk_id += 1

chunks_df = pd.DataFrame(rows)


chunks_df.to_csv("documents_chunks.csv", index=False)

print("Docs:", len(df), "Chunks:", len(chunks_df))
print(chunks_df.head(3))
