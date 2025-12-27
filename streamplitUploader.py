import streamlit as st
import json
import tiktoken
import nltk
import re
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize

#Run it once
nltk.download('punkt_tab')

# ---------------- CONFIG ----------------
MODEL_NAME = "text-embedding-3-large"
encoding = tiktoken.encoding_for_model(MODEL_NAME)

# ---------------- HELPERS ----------------
def is_heading(line):
    line = line.strip()
    return (
        line.isupper()
        or bool(re.match(r"^\d+(\.\d+)*\s+", line))
        or (len(line.split()) <= 6 and not line.endswith("."))
    )

def extract_sentences_with_metadata(pdf_file):
    reader = PdfReader(pdf_file)
    data = []

    current_section = "UNKNOWN"

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        lines = text.split("\n")
        for line in lines:
            if is_heading(line):
                current_section = line.strip()

            sentences = sent_tokenize(line)
            for sentence in sentences:
                data.append({
                    "sentence": sentence,
                    "page": page_num,
                    "section": current_section
                })

    return data


def sentence_aware_token_chunks(sentences, max_tokens=600, overlap_tokens=80):
    chunks = []
    current = []
    current_tokens = 0

    for item in sentences:
        sent = item["sentence"]
        sent_tokens = len(encoding.encode(sent))

        if sent_tokens > max_tokens:
            continue

        if current_tokens + sent_tokens > max_tokens:
            chunks.append(build_chunk(current, current_tokens))

            # overlap logic
            overlap = []
            overlap_tokens_used = 0
            for s in reversed(current):
                s_tokens = len(encoding.encode(s["sentence"]))
                if overlap_tokens_used + s_tokens <= overlap_tokens:
                    overlap.insert(0, s)
                    overlap_tokens_used += s_tokens
                else:
                    break

            current = overlap.copy()
            current_tokens = overlap_tokens_used

        current.append(item)
        current_tokens += sent_tokens

    if current:
        chunks.append(build_chunk(current, current_tokens))

    return chunks


def build_chunk(sent_items, token_count):
    return {
        "text": " ".join(s["sentence"] for s in sent_items),
        "token_count": token_count,
        "page_start": sent_items[0]["page"],
        "page_end": sent_items[-1]["page"],
        "section": sent_items[-1]["section"]
    }

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="PDF → LLM Chunker (Metadata)", layout="wide")

st.title("📄 PDF to LLM-Ready Chunks")
st.caption("Sentence-aware • Token-based • Page & Section metadata")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("Max tokens per chunk", 300, 1200, 600)
with col2:
    overlap_tokens = st.slider("Overlap tokens", 0, 200, 80)

if uploaded_files and st.button("🚀 Process PDFs"):
    all_chunks = []
    progress = st.progress(0)

    for idx, pdf in enumerate(uploaded_files):
        sentence_data = extract_sentences_with_metadata(pdf)
        chunks = sentence_aware_token_chunks(
            sentence_data,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source_file": pdf.name,
                "chunk_id": i,
                **chunk
            })

        progress.progress((idx + 1) / len(uploaded_files))

    st.success(f"✅ Generated {len(all_chunks)} chunks")

    st.subheader("🔍 Chunk Preview")
    st.json(all_chunks[:2])

    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps(all_chunks, indent=2, ensure_ascii=False),
        file_name="pdf_llm_chunks_with_metadata.json",
        mime="application/json"
    )
