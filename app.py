# ===============================
# File: app.py (CLEAN + FINAL)
# ===============================

import os
import io
import json
from typing import List, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import openai as openai_pkg
from openai import OpenAI

# Lightweight text extraction
from pypdf import PdfReader
from docx import Document as DocxDocument

# -----------------------------
# Helpers
# -----------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "
".join(parts)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with io.BytesIO(file_bytes) as bio:
            doc = DocxDocument(bio)
        return "
".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"[DOCX extraction failed: {e}]"


def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".txt"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
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
    """Cosine similarity between 1 vector a and matrix B (rows). NumPy only."""
    a = a.astype(np.float32)
    B = B.astype(np.float32)
    a /= (np.linalg.norm(a) + 1e-9)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (B @ a).flatten()


RUBRIC_SYSTEM_PROMPT = (
    "You are a structured hiring evaluator.
"
    "Use ONLY the provided CV text as evidence. Do not invent facts.
"
    "Score each dimension 0-5 using the rubric:
"
    "- skills: overlap of must-have skills from JD vs CV
"
    "- experience: direct hands-on alignment with responsibilities in JD
"
    "- seniority: scope/impact/ownership compared to JD level
"
    "- domain: industry or product-domain match
"
    "- tenure: stability; penalize frequent very short stints if pattern exists
"
    "constraints_pass: True unless the JD specifies a hard constraint that is clearly violated.
"
    "Return compact JSON with keys: scores{skills,experience,seniority,domain,tenure,constraints_pass},
"
    "evidence{skills,experience,seniority,domain,tenure}[quotes], and notes[]."
)


def score_with_llm(client: OpenAI, model: str, jd_text: str, cv_text: str) -> dict:
    """Use Chat Completions with JSON output (works on all 1.x SDKs)."""
    prompt = f"""
Job description:
<<<JD>>>
{jd_text}
<<<END JD>>>

Candidate CV:
<<<CV>>>
{cv_text[:15000]}
<<<END CV>>>
"""
    resp = client.chat.completions.create(
        model=model,  # e.g., "gpt-4o-mini" or "gpt-4o"
        messages=[
            {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # Fallback defensive structure if model returned non-JSON (shouldn't with json_object)
        return {
            "scores": {"skills": 0, "experience": 0, "seniority": 0, "domain": 0, "tenure": 0, "constraints_pass": True},
            "evidence": {"skills": [], "experience": [], "seniority": [], "domain": [], "tenure": []},
            "notes": ["LLM returned invalid JSON", content[:300]]
        }


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI CV Screener MVP", layout="wide")

st.title("AI CV Screener MVP")

with st.sidebar:
    st.header("Configuration")
    # Show SDK version for debugging
    st.caption(f"openai version: {getattr(openai_pkg, '__version__', 'unknown')}")

    # API key: use env var or Streamlit secrets or manual entry
    default_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")

    emb_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-large", "text-embedding-3-small"],
        index=0,
    )
    llm_model = st.selectbox(
        "LLM scorer model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Models supporting JSON output are recommended."
    )
    shortlist_k = st.slider("Shortlist size (by embeddings)", 3, 25, 10)
    alpha = st.slider("Weight: embeddings vs rubric (0→embeddings, 1→rubric)", 0.0, 1.0, 0.45)

st.subheader("1) Upload Job Description")
jd_mode = st.radio("JD input", ["Paste text", "Upload file"], horizontal=True)
jd_text = ""
if jd_mode == "Paste text":
    jd_text = st.text_area("Job Description", height=200, placeholder="Paste the JD here…")
else:
    jd_file = st.file_uploader("Upload JD (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], accept_multiple_files=False)
    if jd_file:
        jd_text = extract_text(jd_file)
        st.text_area("Preview JD text", jd_text[:5000], height=200)

st.subheader("2) Upload CVs (max 25)")
cv_files = st.file_uploader(
    "Upload CV files (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

if cv_files:
    if len(cv_files) > 25:
        st.warning("Please upload at most 25 CVs for this prototype.")

run = st.button("Score candidates")

if run:
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar or set it in Secrets.")
        st.stop()
    if not jd_text.strip():
        st.error("Provide a job description text.")
        st.stop()
    if not cv_files or len(cv_files) == 0:
        st.error("Upload at least one CV.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Extract CV texts
    records: List[Tuple[str, str]] = []
    for f in cv_files[:25]:
        with st.spinner(f"Extracting text: {f.name}"):
            f.seek(0)
            text = extract_text(f)
            records.append((f.name, text))

    # Embeddings for JD and CVs
    with st.spinner("Computing embeddings…"):
        jd_vec = embed_texts(client, emb_model, [jd_text])[0]
        cv_vecs = embed_texts(client, emb_model, [t for _, t in records])
        sims = cosine_sim(jd_vec, cv_vecs)

    # Shortlist
    order = np.argsort(-sims)[:shortlist_k]
    shortlist = [(records[i][0], records[i][1], float(sims[i])) for i in order]

    st.write(f"Shortlisted {len(shortlist)} of {len(records)} by semantic similarity.")

    # LLM rubric scoring
    results = []
    for fname, cv_text, emb_sim in shortlist:
        with st.spinner(f"Scoring with LLM: {fname}"):
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
            "evidence_skills": "; ".join(judged.get("evidence", {}).get("skills", [])[:3]),
            "evidence_experience": "; ".join(judged.get("evidence", {}).get("experience", [])[:3]),
            "notes": "; ".join(judged.get("notes", [])[:3])
        })

    df = pd.DataFrame(results).sort_values("final_score", ascending=False)

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="ranked_candidates.csv", mime="text/csv")

    st.caption("Scoring = blend of embedding similarity and rubric-based LLM scoring. Adjust alpha in the sidebar.")
