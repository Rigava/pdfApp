import streamlit as st
import json
import nltk
import tiktoken
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from io import BytesIO

# ---------------- CONFIG ----------------
MODEL_NAME = "text-embedding-3-large"
encoding = tiktoken.encoding_for_model(MODEL_NAME)

# ---------------- FUNCTIONS ----------------
def sentence_aware_token_chunks(text, max_tokens=600, overlap_tokens=80):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))

        if sentence_tokens > max_tokens:
            continue

        if current_tokens + sentence_tokens > max_tokens:
            chunks.append({
                "text": " ".join(current_chunk),
                "token_count": current_tokens
            })

            # overlap logic (sentence-level)
            overlap_sentences = []
            overlap_token_count = 0
            for s in reversed(current_chunk):
                s_tokens = len(encoding.encode(s))
                if overlap_token_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_token_count += s_tokens
                else:
                    break

            current_chunk = overlap_sentences.copy()
            current_tokens = overlap_token_count

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "token_count": current_tokens
        })

    return chunks


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="PDF → LLM Chunker", layout="wide")

st.title("📄 PDF to LLM-Ready Chunks")
st.caption("Sentence-aware • Token-based • RAG-ready")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("Max tokens per chunk", 300, 1200, 600)
with col2:
    overlap_tokens = st.slider("Overlap tokens", 0, 200, 80)

if uploaded_files:
    if st.button("🚀 Process PDFs"):
        all_chunks = []
        progress = st.progress(0)

        for idx, uploaded_file in enumerate(uploaded_files):
            text = extract_text_from_pdf(uploaded_file)
            chunks = sentence_aware_token_chunks(
                text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source_file": uploaded_file.name,
                    "chunk_id": i,
                    "token_count": chunk["token_count"],
                    "text": chunk["text"]
                })

            progress.progress((idx + 1) / len(uploaded_files))

        st.success(f"✅ Generated {len(all_chunks)} chunks")

        # Preview
        st.subheader("🔍 Chunk Preview")
        st.json(all_chunks[:3])

        # Download
        json_bytes = json.dumps(
            all_chunks,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Chunked JSON",
            data=json_bytes,
            file_name="pdf_llm_chunks.json",
            mime="application/json"
        )
