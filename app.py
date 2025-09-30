import os
import io
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument

# -----------------------------
# Helpers
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    return "[Unsupported file type]"

def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.asarray(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    B = B.astype(np.float32)
    a /= (np.linalg.norm(a) + 1e-9)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (B @ a).flatten()

RUBRIC_PROMPT = """
You are a structured hiring evaluator.
Use ONLY the provided CV text as evidence. Do not invent facts.
Score each dimension 0-5:
- skills
- experience
- seniority
- domain
- tenure
constraints_pass: True unless JD specifies a hard requirement that is violated.
Return JSON with keys: scores, evidence, notes.
"""

def score_with_llm(client: OpenAI, model: str, jd_text: str, cv_text: str) -> dict:
    prompt = f"""
Job description:
{jd_text}

Candidate CV:
{cv_text[:15000]}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RUBRIC_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI CV Screener MVP", layout="wide")
st.title("AI CV Screener MVP")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    emb_model = st.selectbox("Embedding model", ["text-embedding-3-large", "text-embedding-3-small"])
    llm_model = st.selectbox("LLM scorer model", ["gpt-4o-mini", "gpt-4o"])
    shortlist_k = st.slider("Shortlist size", 3, 25, 10)
    alpha = st.slider("Weight: embeddings vs rubric", 0.0, 1.0, 0.45)

jd_text = st.text_area("Paste Job Description", height=200)
cv_files = st.file_uploader("Upload up to 25 CVs", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if st.button("Score candidates"):
    if not api_key:
        st.error("Enter your OpenAI API key.")
        st.stop()
    if not jd_text.strip():
        st.error("Provide a job description.")
        st.stop()
    if not cv_files:
        st.error("Upload at least one CV.")
        st.stop()

    client = OpenAI(api_key=api_key)

    records: List[Tuple[str, str]] = []
    for f in cv_files[:25]:
        f.seek(0)
        records.append((f.name, extract_text(f)))

    jd_vec = embed_texts(client, emb_model, [jd_text])[0]
    cv_vecs = embed_texts(client, emb_model, [t for _, t in records])
    sims = cosine_sim(jd_vec, cv_vecs)

    order = np.argsort(-sims)[:shortlist_k]
    shortlist = [(records[i][0], records[i][1], float(sims[i])) for i in order]

    results = []
    for fname, cv_text, emb_sim in shortlist:
        judged = score_with_llm(client, llm_model, jd_text, cv_text)
        s = judged["scores"]
        rubric = 0.35*s["skills"] + 0.35*s["experience"] + 0.10*s["seniority"] + 0.15*s["domain"] + 0.05*s["tenure"]
        final = (1 - alpha) * emb_sim + alpha * (rubric/5.0)
        results.append({
            "cv_file": fname,
            "embedding_sim": round(emb_sim, 4),
            "skills": s["skills"],
            "experience": s["experience"],
            "seniority": s["seniority"],
            "domain": s["domain"],
            "tenure": s["tenure"],
            "constraints_pass": s["constraints_pass"],
            "final_score": round(final, 4),
            "notes": "; ".join(judged.get("notes", []))
        })

    df = pd.DataFrame(results).sort_values("final_score", ascending=False)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="ranked_candidates.csv", mime="text/csv")
