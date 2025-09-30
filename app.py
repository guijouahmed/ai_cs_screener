import os, io, json, time
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---------- Helpers ----------
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
Use ONLY the provided CV text and role attributes as evidence. Do not invent facts.

Score each dimension 0-5: skills, experience, seniority, domain, tenure.
Also extract:
- positive_evidence: short quotes or phrases showing alignment
- negative_evidence: short quotes or notes showing gaps/risks

Return a JSON object with:
- scores {skills, experience, seniority, domain, tenure, constraints_pass}
- evidence {positive[], negative[]}
- notes []
"""

def score_with_llm(client: OpenAI, model: str, jd_text: str, attributes: str, cv_text: str) -> dict:
    prompt = f"""
Job description:
{jd_text}

Additional role attributes:
{attributes}

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
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "scores": {
                "skills": 0, "experience": 0, "seniority": 0,
                "domain": 0, "tenure": 0, "constraints_pass": True
            },
            "evidence": {"positive": [], "negative": []},
            "notes": ["LLM returned invalid JSON"]
        }

# ---------- UI ----------
st.set_page_config(page_title="AI CV Screener MVP", layout="wide")
st.title("AI CV Screener MVP")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    emb_model = st.selectbox("Embedding model", ["text-embedding-3-large", "text-embedding-3-small"])
    llm_model = st.selectbox("LLM scorer model", ["gpt-4o-mini", "gpt-4o"])
    shortlist_k = st.slider("Shortlist size", 3, 25, 10)
    alpha = st.slider("Weight: embeddings vs rubric", 0.0, 1.0, 0.45)

st.subheader("1) Job Description")
jd_mode = st.radio("JD input", ["Paste text", "Upload file"], horizontal=True)
jd_text = ""
if jd_mode == "Paste text":
    jd_text = st.text_area("Paste JD", height=200)
else:
    jd_file = st.file_uploader("Upload JD (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    if jd_file:
        jd_text = extract_text(jd_file)
        st.text_area("JD preview", jd_text[:5000], height=200, disabled=True)

st.subheader("2) Additional attributes")
attributes = st.text_area("Enter extra attributes you want to check (comma or line separated)", height=100)

st.subheader("3) Upload CVs (max 25)")
cv_files = st.file_uploader("Upload CVs", type=["txt", "pdf", "docx"], accept_multiple_files=True)

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
    total = len(shortlist)
    prog = st.progress(0, text=f"Scoring {total} candidates...")
    status = st.empty()
    start_ts = time.time()

    for i, (fname, cv_text, emb_sim) in enumerate(shortlist, start=1):
        status.write(f"Scoring {i}/{total}: {fname}")
        judged = score_with_llm(client, llm_model, jd_text, attributes, cv_text)

        defaults_scores = {
            "skills": 0, "experience": 0, "seniority": 0,
            "domain": 0, "tenure": 0, "constraints_pass": True
        }
        s = {**defaults_scores, **(judged.get("scores") or {})}
        ev = judged.get("evidence") or {}
        pos = "; ".join((ev.get("positive") or [])[:3])
        neg = "; ".join((ev.get("negative") or [])[:3])

        rubric = (
            0.35 * s["skills"] +
            0.35 * s["experience"] +
            0.10 * s["seniority"] +
            0.15 * s["domain"] +
            0.05 * s["tenure"]
        )
        final = (1 - alpha) * emb_sim + alpha * (rubric / 5.0)

        results.append({
            "cv_file": fname,
            "embedding_sim": round(emb_sim, 4),
            "skills": s["skills"],
            "experience": s["experience"],
            "seniority": s["seniority"],
            "domain": s["domain"],
            "tenure": s["tenure"],
            "constraints_pass": bool(s["constraints_pass"]),
            "final_score": round(final, 4),
            "positive_evidence": pos,
            "negative_evidence": neg,
            "notes": "; ".join(judged.get("notes", [])[:3])
        })

        elapsed = time.time() - start_ts
        avg_per = elapsed / i
        remaining = total - i
        eta_sec = int(avg_per * remaining)
        prog.progress(i / total, text=f"Scored {i}/{total}. ETA ~{eta_sec}s")

    status.write("Scoring complete")
    prog.progress(1.0)

    df = pd.DataFrame(results).sort_values("final_score", ascending=False)
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="ranked_candidates.csv",
                       mime="text/csv")
