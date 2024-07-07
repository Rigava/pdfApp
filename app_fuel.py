import streamlit as st
import requests
from pypdf import PdfReader
import os
import google.generativeai as palm
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import GooglePalm
from htmlTemplates import bot_template, user_template, css
from PIL import Image
# from tabula import read_pdf
# import io
from langchain_community.document_loaders import PyPDFLoader

key =st.secrets.API_KEY

def init():
    st.set_page_config(
        page_title="Chat with your PDFs",
        page_icon=":books"
    )

# def get_pdf_text(doc):
#     text = ""
#     for pdf in doc:
#         pdf_reader=PdfReader(pdf) # read each page
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text
def get_pdf_text(url):
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()
    text_list = []
    # pdf = PdfReader(doc)
    # num_pages = len(pdf.pages)

    for page in range(pages):
        page_text = pdf.pages[page].extract_text()
        text_list.append(page_text)
    text = "\n".join(text_list)
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):  
    # For Huggingface Embeddings
    embeddings = GooglePalmEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    # HuggingFace Model
    llm = GooglePalm(model ='models/gemini-1.0-pro')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(question):
    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        # st.write(i,message
        #          )
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    init()
    st.write(css,unsafe_allow_html=True)

    st.header("Chat with your pdf :books:")
    user_question = st.text_input("Ask a question about your document: ",key="user_input")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}","Hello bot"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello Human"),unsafe_allow_html=True)
    
    with st.sidebar:        
        # pdf_docs = st.file_uploader("Upload your PDFs here and click to submit",accept_multiple_files=True)     
        # if pdf_file is not None:
        url = "https://raw.githubusercontent.com/Rigava/DataRepo/main/FuelEU_faq_2_e[1].pdf"
        # download = requests.get(url).content
        # pdf_docs = io.BytesIO(download)


                
        if st.button("Submit"):
           with st.spinner("processing"):
                #get the pdfs to text
                raw_text = get_pdf_text(url)
                #get the text chunk
                text_chunk = get_text_chunks(raw_text)
                #create vector store
                vectorstore= get_vector_store(text_chunk)
                #Create Conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Embedding done!")



if __name__ == '__main__':
    main()

