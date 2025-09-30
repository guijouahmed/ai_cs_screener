# ===============================
# File: app.py
# ===============================

import os
import io
import json
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import streamlit as st
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
        return "\n".join(parts)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with io.BytesIO(file_bytes) as bio:
            doc = DocxDocument(bio)
        return "\n".join([p.text for p in doc.paragraphs])
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
    # OpenAI embeddings API: returns list of vectors
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.asarray(vecs, dtype=np.float32)


RUBRIC_SCHEMA = {
    "name": "cv_fit_score",
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    "skills": {"type": "number", "minimum": 0, "maximum": 5},
                    "experience": {"type": "number", "minimum": 0, "maximum": 5},
                    "seniority": {"type": "number", "minimum": 0, "maximum": 5},
                    "domain": {"type": "number", "minimum": 0, "maximum": 5},
                    "tenure": {"type": "number", "minimum": 0, "maximum": 5},
                    "constraints_pass": {"type": "boolean"}
                },
                "required": [
                    "skills",
                    "experience",
                    "seniority",
                    "domain",
                    "tenure",
                    "constraints_pass"
                ]
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "experience": {"type": "array", "items": {"type": "string"}},
                    "seniority": {"type": "array", "items": {"type": "string"}},
                    "domain": {"type": "array", "items": {"type": "string"}},
                    "tenure": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["skills", "experience", "seniority", "domain", "tenure"]
            },
            "notes": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["scores", "evidence", "notes"],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = (
    "You are a structured hiring evaluator.\n"
    "Use ONLY the provided CV text as evidence. Do not invent facts.\n"
    "Score each dimension 0-5 using the rubric:\n"
    "- skills: overlap of must-have skills from JD vs CV\n"
    "- experience: direct hands-on alignment with responsibilities in JD\n"
    "- seniority: scope/impact/ownership compared to JD level\n"
    "- domain: industry or product-domain match\n"
    "- tenure: stability; penalize frequent very short stints if pattern exists\n"
    "constraints_pass: True unless the JD specifies a hard constraint that is clearly violated.\n"
    "Return JSON matching the schema exactly. Evidence must be short quotes copied from the CV text."
)


def score_with_llm(client: OpenAI, model: str, jd_text: str, cv_text: str) -> dict:
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
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": RUBRIC_SCHEMA},
    )
    # Prefer output_text; fallback to digging into content if needed
    try:
        payload = resp.output_text
    except Exception:
        payload = resp.output[0].content[0].text
    return json.loads(payload)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI CV Screener MVP", layout="wide")

st.title("AI CV Screener MVP")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Key is kept only in session memory.")
    if not api_key and os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")

    emb_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-large", "text-embedding-3-small"],
        index=0,
    )
    llm_model = st.selectbox(
        "LLM scorer model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Models supporting structured JSON are recommended."
    )
    shortlist_k = st.slider("Shortlist size (by embeddings)", 3, 25, 10)
    alpha = st.slider("Weight: embeddings vs rubric (0→all embeddings, 1→all rubric)", 0.0, 1.0, 0.45)

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
        st.error("Enter your OpenAI API key in the sidebar.")
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
            # Important: reset file pointer for repeated reads
            f.seek(0)
            text = extract_text(f)
            # simple PII scrubbing (optional): remove emails
            # text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)
            records.append((f.name, text))

    # Embeddings for JD and CVs
    with st.spinner("Computing embeddings…"):
        jd_vec = embed_texts(client, emb_model, [jd_text])[0]
        cv_vecs = embed_texts(client, emb_model, [t for _, t in records])
        sims = 1.0 - cdist([jd_vec], cv_vecs, metric="cosine").flatten()

    # Shortlist
    order = np.argsort(-sims)[:shortlist_k]
    shortlist = [(records[i][0], records[i][1], float(sims[i])) for i in order]

    st.write(f"Shortlisted {len(shortlist)} of {len(records)} by semantic similarity.")

    # LLM rubric scoring
    results = []
    for fname, cv_text, emb_sim in shortlist:
        with st.spinner(f"Scoring with LLM: {fname}"):
            try:
                judged = score_with_llm(client, llm_model, jd_text, cv_text)
            except Exception as e:
                judged = {
                    "scores": {
                        "skills": 0, "experience": 0, "seniority": 0, "domain": 0, "tenure": 0, "constraints_pass": True
                    },
                    "evidence": {"skills": [], "experience": [], "seniority": [], "domain": [], "tenure": []},
                    "notes": [f"LLM scoring failed: {e}"]
                }
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
            "evidence_skills": "; ".join(judged["evidence"]["skills"][:3]),
            "evidence_experience": "; ".join(judged["evidence"]["experience"][:3]),
            "notes": "; ".join(judged.get("notes", [])[:3])
        })

    df = pd.DataFrame(results).sort_values("final_score", ascending=False)

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="ranked_candidates.csv", mime="text/csv")

    st.caption("Scoring = blend of embedding similarity and rubric-based LLM scoring. Adjust alpha in the sidebar.")