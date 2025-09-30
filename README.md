# ===============================
# File: README.md
# ===============================

# AI CV Screener MVP (Streamlit)

This is a one-page prototype that:
1) Uploads up to 25 CVs and a Job Description (JD)
2) Computes semantic similarity with OpenAI embeddings (shortlist)
3) Re-scores the shortlist with a structured JSON LLM rubric ("reads" JD + CV together)
4) Shows a ranked table and lets you download a CSV

## Prerequisites
- Python 3.10+
- An OpenAI API key with access to `text-embedding-3-large` and `gpt-4o-mini` (or `gpt-4o`)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...  # Windows: set OPENAI_API_KEY=sk-...
```

## Run
```bash
streamlit run app.py
```

Open the local URL in your browser. In the sidebar you can paste your OpenAI key if you didn't export it.

## Use
- Paste or upload the JD
- Upload 1–25 CVs as .txt, .pdf, or .docx
- Click "Score candidates"
- Adjust the alpha slider to change weighting between embeddings and rubric scores

## Notes
- Documents are processed in-memory; nothing is persisted. For real use, store artifacts in S3 and add auth.
- PDF/DOCX extraction is basic; upgrade to a dedicated parser if needed.
- The LLM rubric returns quotes from the CV as evidence for explainability.
- Keep CV text under ~15k chars per candidate (truncated in this MVP).

## Next steps
- Add a vector database when dataset grows (FAISS, pgvector).
- Add bias/PII scrubbing before scoring.
- Export a per-candidate evidence report.
- Move to FastAPI + a queue for batch throughput.
