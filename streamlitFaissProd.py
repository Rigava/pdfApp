import streamlit as st
import json
import re
import os
import faiss
import nltk
import tiktoken
import numpy as np
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import google.genai as genai

# ========================= CONFIG =========================

st.set_page_config("Production RAG App", layout="wide")

INDEX_DIR = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"
TOKEN_MODEL = "text-embedding-3-large"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 5

os.makedirs(INDEX_DIR, exist_ok=True)

# One-time downloads
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")

download_nltk()

# Tokenizer
encoding = tiktoken.encoding_for_model(TOKEN_MODEL)

# Gemini client
client = genai.Client(api_key=st.secrets.GOOGLE_API_KEY)


# ========================= CACHED MODELS =========================

@st.cache_resource
def get_embedder():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_faiss():
    if not os.path.exists(f"{INDEX_DIR}/index.faiss"):
        return None, None

    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")

    with open(f"{INDEX_DIR}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


# ========================= SESSION INIT =========================

def init_session():

    if "embedder" not in st.session_state:
        st.session_state.embedder = get_embedder()

    if "index" not in st.session_state:
        index, metadata = load_faiss()
        st.session_state.index = index
        st.session_state.metadata = metadata

    if "messages" not in st.session_state:
        st.session_state.messages = []

init_session()


# ========================= PDF PROCESSING =========================

def is_heading(line):

    return (
        line.isupper()
        or bool(re.match(r"^\d+(\.\d+)*\s+", line))
        or (len(line.split()) <= 6 and not line.endswith("."))
    )


def extract_sentences(pdf):

    reader = PdfReader(pdf)
    data = []
    section = "UNKNOWN"

    for page_num, page in enumerate(reader.pages, start=1):

        text = page.extract_text()
        if not text:
            continue

        for line in text.split("\n"):

            if is_heading(line):
                section = line.strip()

            for sent in sent_tokenize(line):

                data.append(
                    {
                        "sentence": sent,
                        "page": page_num,
                        "section": section,
                    }
                )

    return data


def build_chunk(sentences):

    return {
        "text": " ".join([s["sentence"] for s in sentences]),
        "page_start": sentences[0]["page"],
        "page_end": sentences[-1]["page"],
        "section": sentences[-1]["section"],
    }


def chunk_sentences(sentences, max_tokens=600, overlap=80):

    chunks = []
    current = []
    tokens = 0

    for s in sentences:

        t = len(encoding.encode(s["sentence"]))

        if tokens + t > max_tokens:

            chunks.append(build_chunk(current))

            overlap_sents = current[-2:]
            current = overlap_sents
            tokens = sum(len(encoding.encode(x["sentence"])) for x in current)

        current.append(s)
        tokens += t

    if current:
        chunks.append(build_chunk(current))

    return chunks


# ========================= INGESTION =========================

def ingest_pdfs(files):

    embedder = st.session_state.embedder

    all_chunks = []
    texts = []

    for pdf in files:

        sentences = extract_sentences(pdf)
        chunks = chunk_sentences(sentences)

        for i, chunk in enumerate(chunks):

            chunk["source_file"] = pdf.name
            chunk["chunk_id"] = i

            all_chunks.append(chunk)
            texts.append(chunk["text"])

    embeddings = embedder.encode(texts).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")

    with open(f"{INDEX_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    st.session_state.index = index
    st.session_state.metadata = all_chunks


# ========================= SEARCH =========================

def search(query):

    embedder = st.session_state.embedder
    index = st.session_state.index
    metadata = st.session_state.metadata

    query_vec = embedder.encode([query]).astype("float32")

    distances, indices = index.search(query_vec, TOP_K)

    return [metadata[i] for i in indices[0]]


# ========================= PROMPT =========================

def build_prompt(question, chunks):

    context = ""

    for i, c in enumerate(chunks, start=1):

        context += f"""
Source {i}
File: {c['source_file']}
Pages: {c['page_start']}–{c['page_end']}
Section: {c['section']}
Text: {c['text']}
"""

    return f"""
Answer using ONLY the provided sources.

Question:
{question}

Sources:
{context}

Rules:
- Be concise
- Cite sources
- Do not hallucinate
"""


# ========================= LLM =========================

def generate_answer(prompt):

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text


# ========================= UI =========================

st.title("📄 Production RAG App")


# ---------- INGEST ----------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Ingest PDFs") and uploaded_files:

    with st.spinner("Processing PDFs..."):
        ingest_pdfs(uploaded_files)

    st.success("Ingestion complete")


# ---------- CHAT ----------

st.divider()
st.subheader("Chat")


if st.session_state.index is None:

    st.info("Please ingest documents first")

else:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])


    query = st.chat_input("Ask something")

    if query:

        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Searching..."):
            chunks = search(query)

        prompt = build_prompt(query, chunks)

        with st.spinner("Thinking..."):
            answer = generate_answer(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)
