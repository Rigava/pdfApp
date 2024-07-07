from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # Updated Embedding Package Here... Causes problems otherwise. (UUID cannot be imported.)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.chat_models.google_palm import ChatGooglePalm
from langchain.chains import RetrievalQA
import streamlit as st

# Toggle to the secret keys when deploying in streamlit community

# key =st.secrets.API_KEY

# main functions
def split_text_documents(docs: list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separators=[" ", ",", "\n"])
    documents = text_splitter.split_documents(docs)
    return documents

def text_to_doc_splitter(text: str):
    spliiter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 0, length_function = len, add_start_index = True,)
    document = spliiter.create_documents([text])
    return document

def extract_text_from_url(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = []
    for lines in soup.findAll('div', {'class': 'description__text'}):
        text.append(lines.get_text())
    
    lines = (line.strip() for line in text)
    text = '\n'.join(line for line in lines if line)
    
    document = text_to_doc_splitter(text)
    return document

def load_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    document = text_to_doc_splitter(text)
    return document

# LLM functions
def run_llm(url, pdf,key, temperature):
    pdf_doc = load_pdf(pdf)
    job_post = extract_text_from_url(url)

    pdf_doc.extend(job_post)
    documents = split_text_documents(pdf_doc)    
    embeddings = GooglePalmEmbeddings(google_api_key =key)
    vectordb = FAISS.from_documents(documents, embedding = embeddings)
    llm = ChatGooglePalm(model ='models/gemini-1.0-pro',google_api_key =key,temperature=temperature)

    pdf_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 4}),
        chain_type="stuff",
    )
    return pdf_qa

def get_cover_letter(pdf_qa):

    query = """ Write a cover letter for given CV and Job posting. You are passed a \
                CV and a job description. Use only information in the CV as examples to put into \
                the Cover Letter. Relate the examples you choose to the skills discussed in the \
                job description, but DO NOT use the job description as a source for examples for the \
                Cover Letter. Follow the template outlined below for the appearance of the returned \
                result. It is up to you to generate the text for the BODY. This should be 4-6 paragraphs. \
                Use an intelligent but conversational style for your response and prioritize more recent experience in the CV.\
                Sign the letter in the NAME FROM CV section with the name extraced from the CV passed to you.
                
                Template: 
                Dear Hiring Manager, 
                [BODY]
                Thank you, 
                [NAME FROM CV]
                """

    result = pdf_qa.invoke(query)

    return result

def get_resume_improvements(pdf_qa):
    
    query = """ Using the attached Job Posting, can you identify the keywords and most important skills highlighted \
                in the Job Posting? Can you then use those keywords and highlighted skills to identify what elements \
                of the attached CV should be highlighted more than others. Can you also highlight what elements should \
                be downplayed or minimized that fall out of the scope of the Job Posting? \
                
                You are passed a CV and a job description. Use the job description as a guide to \
                identify improvements that could be made to the CV. Improvements can include, but are \
                not limited to, mentioning certain skills present in the job description that are not present in \
                the CV, and highlighting experiences that would better align with the \
                job description. You should provide 3-10 recommendations in your responses. 
                
                Template:
                Main Skills In Job Posting:
                [KEYWORDS AND IMPORTANT SKILLS]
                
                Things To Highlight In Your CV:
                [THING IN CV TO HIGHLIGHT]
                
                Things To Get Rid Of In Your CV:
                [THINGS TO REMOVE IN CV]
                """
    
    result = pdf_qa.invoke(query)

    return result
#Streamlit app
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



def generate_response(url, cv, key, temperature, cv_improvements, cover_letter):
  
  pdf_qa = run_llm(url, cv, key, temperature)

  if cv_improvements:
      output = get_resume_improvements(pdf_qa)
      with st.container():
        st.subheader('Potential CV Improvements:')
        st.write(output['result'])
        st.divider()
      
  if cover_letter:
      output = get_cover_letter(pdf_qa)
      with st.container():
        st.subheader('Generated Cover Letter:')
        st.write(output['result'])
        st.divider()

  return

def main():

  try:
     # Retrieve the OpenAI API key from the environment variable
      key = "AIzaSyAKEaaM7fWIErN3VbikjP_T5m0UfhBy5iE"
  except:
      key = ''
     
  
  st.sidebar.subheader('Choose Model Function:')
  cv_improvements = st.sidebar.checkbox('CV Suggestions', value=True)
  cover_letter = st.sidebar.checkbox('Cover Letter Generator', value=True)

  if key == '':
        st.sidebar.subheader('Enter API Key:')
        openai_api_key = st.sidebar.text_input('API Key', '', type='password')
  
  st.sidebar.subheader('Set Model Temperature:')
  temperature = st.sidebar.slider("Model Temperature", 0.0, 2.0, step=0.1, value=0.3)
  
  st.title('Job Application Assistant')
  st.markdown(''' \
          Welcome to the Job Application Assistant! \n\n\
          The way this application works is by taking a LinkedIn Public Job Posting URL \n\
          and a PDF copy of your CV to generate suggestions on how to \n\
          improve your CV and/or to generate a tailored cover letter for the job. \n\n\
          Please fill out the fields on this page and hit submit when ready! \n\n\
          ''')
  
  with st.expander(':red[Disclaimer]'):
    st.write(""" 
            This application is offered as a free tool. It is connected to and uses ChatGPT and the OpenAI API. \
            Anything entered here is passed to ChatGPT. Do not input personally identifiable information, \
            confidential information, or anything of the sort. The creator of this tool is not responsible for the use \
            or outcomes of this publicly used application and you agree to use this at your own risk. 
            """)
  
  
  with st.form('my_form'):
      text = st.text_area('Enter LinkedIn Job URL:', '')
      files = st.file_uploader("Upload Files:", type=["pdf"], accept_multiple_files=False)
      
      submitted = st.form_submit_button('Submit')
          
  if submitted:
      if key != '' and key.startswith('AI'):
        if text != '':
          if files != None:
            #try:
              with st.spinner('Please wait for the model to load. This may take a minute...'):
                generate_response(text, files, key, temperature, cv_improvements, cover_letter)
            # except:
            #    st.error('An Error Occured With The Model! Please Try Again', icon="🚨")
          else:
             st.warning('There may be a problem. Please Check Your Uploaded File.', icon='⚠')
        else:
          st.warning('There may be a problem. Please Check Your URL.', icon='⚠')
      else:
        st.warning('There may be a problem. Please Check Your OpenAI API Key.', icon='⚠')
  return

if __name__ == "__main__":
   main()
