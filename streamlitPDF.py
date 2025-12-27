# https://colab.research.google.com/drive/1P7pc4GXOk1WyVg-qgXDTxNoYIgsx5Xrg#scrollTo=g3S60vBpNwvj
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_classic.chains import RetrievalQA

# ---------------- CONFIG ----------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# --------------------------------------

st.set_page_config(page_title="PDF RAG with Gemini", layout="wide")
st.title("📄🔍 PDF RAG with Streaming Gemini")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Chunk Settings")

chunk_size = st.sidebar.slider(
    "Chunk size (characters)",
    min_value=300,
    max_value=2000,
    value=1000,
    step=100,
)

overlap = st.sidebar.slider(
    "Chunk overlap",
    min_value=0,
    max_value=500,
    value=150,
    step=50,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Changing chunk size or overlap will rebuild the FAISS index."
)

# ---------------- Utils ----------------
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

def extract_documents_from_pdfs(uploaded_files, chunk_size, overlap):
    documents = []

    for file in uploaded_files:
        reader = PdfReader(file)
        full_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        chunks = chunk_text(full_text, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file.name,
                        "chunk_id": i
                    }
                )
            )

    return documents

# ---------------- Streaming Callback ----------------
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# ---------------- Prompt ----------------
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------- File Upload ----------------
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# ---------------- Build FAISS ----------------
if uploaded_files:
    with st.spinner("🔨 Building FAISS index..."):
        documents = extract_documents_from_pdfs(
            uploaded_files, chunk_size, overlap
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vectorstore = FAISS.from_documents(
            documents, embeddings
        )

        st.session_state["retriever"] = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    st.success(f"✅ Indexed {len(documents)} chunks")

# ---------------- Ask Question ----------------
question = st.text_input("Ask a question about the uploaded PDFs")

if question and "retriever" in st.session_state:
    st.markdown("### 🤖 Answer (streaming)")
    answer_container = st.empty()

    callback = StreamlitCallbackHandler(answer_container)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        streaming=True,
        callbacks=[callback],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state["retriever"],
        chain_type="stuff",
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )

    result = qa_chain(question)

    with st.expander("📚 Source Chunks"):
        for doc in result["source_documents"]:
            st.markdown(
                f"**{doc.metadata['source']} (chunk {doc.metadata['chunk_id']})**"
            )
            st.write(doc.page_content[:500] + "...")
