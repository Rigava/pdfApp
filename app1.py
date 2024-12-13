# D:\GEN AI_HUGGINGFACE\pdfApp
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from htmlTemplates import bot_template, user_template, css
from PIL import Image

def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
        print("API_TOKEN is not set")
        exit(1)
    else:
        print("API_TOKEN is set")
    st.set_page_config(
        page_title="Chat with your PDFs",
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
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=24,separators=[" ", ",", "\n"])
    chunks = text_splitter.create_documents([text])
    return chunks

def get_vector_store(text_chunks):  
    # For Huggingface Embeddings
    embeddings = GPT4AllEmbeddings()
    vectorstore = SKLearnVectorStore.from_documents(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatGroq(
    temperature=0,
    groq_api_key = "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",
    model_name = 'llama-3.1-70b-versatile')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain
# def get_conversation_chain(vector_store):
#     # ConversationalRetrievalChain
#     llm = ChatGooglePalm(model ='models/gemini-1.0-pro',google_api_key =key)
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm = llm,
#         retriever = vector_store.as_retriever(),
#         memory = memory
#     )
#     return conversation_chain


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
                st.success("Embedding done!")
                st.session_state.conversation = get_conversation_chain(vectorstore)
               
                # res= get_conversation_chain1(vectorstore)
                # st.write(res)





if __name__ == '__main__':
    main()
