
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
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
from langchain.chains.question_answering import load_qa_chain

key =st.secrets.API_KEY
def init():
    st.set_page_config(
        page_title="Summary tool",
        page_icon=":books"
    )

def get_pdf_text(doc):
    text = ""
    for pdf in doc:
        pdf_reader=PdfReader(pdf) # read each page
        for page in pdf_reader.pages:
            text += page.extract_text()
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
    embeddings = GooglePalmEmbeddings(google_api_key =key)
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore


def get_conversation_chain(vector_store):
    # Load_qa_chain
    llm = GooglePalm(model ='models/text-bison-001',google_api_key =key)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    query = "What is the summary"
    docs = vector_store.similarity_search(query)
    
    
    return chain.run(input_documents=docs, question=query)

def handle_user_input(question):
    question = "Provide a concise summary of the document"
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

    st.header("Summary of your pdf :books:")
    user_question = st.text_input("Ask a question about your document: ",key="user_input")
   

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}","Hello bot"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello Human. Please upload the file"),unsafe_allow_html=True)
    
    with st.sidebar:        
        pdf_docs = st.file_uploader("Upload your PDFs here and click to submit",accept_multiple_files=True)
        if st.button("Submit"):
           with st.spinner("processing"):
                #get the pdfs to text
                raw_text = get_pdf_text(pdf_docs)
                #get the text chunk
                text_chunk = get_text_chunks(raw_text)
                #create vector store
                vectorstore= get_vector_store(text_chunk)
                #Create Conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Embedding done!")





if __name__ == '__main__':
    main()
