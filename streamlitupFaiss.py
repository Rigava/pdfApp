import streamlit as st
import json
import re
import os
import faiss
import nltk
import tiktoken
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

#Run it once
nltk.download('punkt_tab')
# ---------------- CONFIG ----------------
MODEL_NAME = "all-MiniLM-L6-v2"   # fast + excellent
TOKEN_MODEL = "text-embedding-3-large"

INDEX_DIR = "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

encoding = tiktoken.encoding_for_model(TOKEN_MODEL)
embedder = SentenceTransformer(MODEL_NAME)

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

        for line in text.split("\n"):
            if is_heading(line):
                current_section = line.strip()

            for sent in sent_tokenize(line):
                data.append({
                    "sentence": sent,
                    "page": page_num,
                    "section": current_section
                })

    return data

def sentence_aware_chunks(sentences, max_tokens=600, overlap_tokens=80):
    chunks, current, tokens = [], [], 0

    for s in sentences:
        t = len(encoding.encode(s["sentence"]))
        if t > max_tokens:
            continue

        if tokens + t > max_tokens:
            chunks.append(build_chunk(current, tokens))
            current, tokens = overlap_sentences(current, overlap_tokens)

        current.append(s)
        tokens += t

    if current:
        chunks.append(build_chunk(current, tokens))

    return chunks

def overlap_sentences(sentences, overlap_tokens):
    overlap, tokens = [], 0
    for s in reversed(sentences):
        t = len(encoding.encode(s["sentence"]))
        if tokens + t <= overlap_tokens:
            overlap.insert(0, s)
            tokens += t
        else:
            break
    return overlap, tokens

def build_chunk(sentences, token_count):
    return {
        "text": " ".join(s["sentence"] for s in sentences),
        "token_count": token_count,
        "page_start": sentences[0]["page"],
        "page_end": sentences[-1]["page"],
        "section": sentences[-1]["section"]
    }

# ---------------- STREAMLIT UI ----------------
st.set_page_config("PDF → FAISS Ingestion", layout="wide")
st.title("📄 PDF → FAISS Vector DB (RAG Ready)")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("Max tokens per chunk", 300, 1200, 600)
with col2:
    overlap_tokens = st.slider("Overlap tokens", 0, 200, 80)

if uploaded_files and st.button("🚀 Ingest into FAISS"):
    all_chunks = []
    texts = []

    for pdf in uploaded_files:
        sentences = extract_sentences_with_metadata(pdf)
        chunks = sentence_aware_chunks(
            sentences,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source_file": pdf.name,
                "chunk_id": i,
                **chunk
            })
            texts.append(chunk["text"])

    st.info("🔢 Generating embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    st.info("📦 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index + metadata
    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    st.success(f"✅ Ingested {len(all_chunks)} chunks into FAISS")

    st.download_button(
        "⬇️ Download metadata JSON",
        data=json.dumps(all_chunks, indent=2, ensure_ascii=False),
        file_name="faiss_metadata.json",
        mime="application/json"
    )

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_faiss():
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)

index, metadata = load_faiss()
embedder = load_embedder()

# ---------------- SEARCH ----------------
def semantic_search(query, k=TOP_K):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


# ---------------- STREAMLIT UI ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve documents
    with st.spinner("🔍 Searching documents..."):
        results = semantic_search(user_query)

    # Build answer (retrieval-only for now)
    answer_text = ""
    for i, r in enumerate(results, start=1):
        answer_text += f"""
**Result {i}**
- 📄 **Source**: {r['source_file']}
- 📍 **Pages**: {r['page_start']}–{r['page_end']}
- 🧩 **Section**: {r['section']}

{r['text']}

---
"""

    # Show assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text
    })

    with st.chat_message("assistant"):
        st.markdown(answer_text)
