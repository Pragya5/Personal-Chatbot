#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('conda info --envs')


# In[2]:


#get_ipython().system('conda activate genai/')


# ### Importing the necessary libraries

# In[3]:

import mimetypes
import os
import imghdr
import warnings
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import os
from PIL import Image
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import docx
import fitz
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from io import BytesIO
from dotenv import load_dotenv


# ### Warning Settings

# In[4]:


# Set logging level to WARNING
logging.basicConfig(level=logging.WARNING)

# Suppress specific warning by category
warnings.filterwarnings("ignore")


# ### Configuring the GOOGLE_API_KEY 

# In[5]:


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ### Checking if it is set correctly

# In[6]:


if google_api_key is not None:
    print(f"GOOGLE_API_KEY is set: {google_api_key}")
else:
    print("GOOGLE_API_KEY is not set.")

# ### Initialising a model for handling images (Gemini pro vision)
img_model=genai.GenerativeModel('gemini-pro-vision') 

def get_gemini_response(input,image,user_prompt):
    response=img_model.generate_content([input,image[0],user_prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    

# ### Function to be able to accept different text formats (.docx, .pdf, .txt)

# In[7]:

input_prompt="""
Instructions:
1. You are an expert in understanding invoices. We will upload a a image as invoice
2. You will have to answer any questions based on the uploaded image.
3. Do not give the wrong answer.
4. Do not give out details about sensitive data such as Account number, ATM Pin etc and say "We cannot disclose user's sensitive data such as account number and ATM pin.
5. If you cant find the answer, simply return "Cant find the answer to this question in the given context"
"""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf.name.lower().endswith('.pdf'):
            pdf_reader = fitz.open("pdf", pdf.read())
            for page_num in range(pdf_reader.page_count):
                page = pdf_reader[page_num]
                text += page.get_text()
            pdf_reader.close()
        elif pdf.name.lower().endswith('.docx'):
            doc = docx.Document(BytesIO(pdf.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + '\n'
        elif pdf.name.lower().endswith('.txt'):
            text += pdf.read().decode('utf-8') + '\n'
        else:
            st.warning(f"Unsupported file format: {pdf.name}")
    return text


# ### Function to convert the documents to chunks

# In[8]:


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks


# ### Converting the chunks to vector embeddings and storing into FAISS database

# In[9]:


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# ### Prompt engineering

# In[10]:


def get_conversational_chain():
    prompt_template = """
    **Instructions:**
    -Try to find the answer as close to the question as possible.
    -Accept questions in capital letters.
    -Do not to answer questions related to Bank details. 
    - Provide a comprehensive and accurate answer based on the information given in the context.
    -Try to match questions having two three words matching from the front or the back.
    - If the answer is not present in the provided context, explicitly state "Answer not available in the context" without attempting to guess.
    - Avoid providing incorrect or speculative information.

**Context:**
{context}

**Question:**
{question}

**Answer:**
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# ### Function to input user query and convert to vector embeddings for comparison

# In[11]:


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])


# ### Streamlit app code

# In[12]:


os.chdir(r'C:\Users\prgya\OneDrive\Desktop\Genai docs')
def main():
    st.set_page_config(page_title="SOP Investigator", layout='wide')

    # Check if the app is running for the first time to use the effect =snow
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
        st.snow()
    _, exp_col, _ = st.columns([2.5,5,1])
    with exp_col:
        st.image('Investigator.png', width=300)


    _, exp_col, _ = st.columns([0.79,2,1])
    with exp_col:
        with st.expander("**🧙🏻‍♂️ How to Use This Application**"):
            st.markdown("""
                        Upload different types of files ! 🧙🏻‍♂️

                        
Walkthrough: Generative AI Application

Welcome to our Generative AI application! This user-friendly tool is designed to seamlessly handle both textual data and images, providing insightful answers to your queries. Let's take a walkthrough to understand how to make the most of this versatile application.

1. Landing Page:
Upon visiting the application's landing page, you'll be greeted by an inviting interface. The clean layout features a logo and a title that clearly indicates the purpose of the application: "AI-Powered Question Answering."

2. Asking Questions:
To get started, you can type your questions directly into the input box provided on the main section of the page. The application is designed to understand a wide range of queries, so feel free to ask about any topic you have in mind.

3. Textual Data Input:
If you have textual data that you'd like the AI to analyze, there's a dedicated section for uploading files. Whether it's a PDF document, a Word file, or a plain text file, simply click on the "Upload" button and select the file you wish to analyze. The application will process the text and generate relevant insights.

4. Image Input:
For image-based queries, use the separate file uploader designed for images. Upload PNG, JPG, or JPEG files to receive answers based on the content of the images. The application utilizes advanced image processing capabilities to extract meaningful information.

5. Submitting Queries:
After entering your questions or uploading files, you can submit your queries by clicking the relevant "Submit" button. The application will then initiate the AI analysis process, providing you with a comprehensive response.

6. Results:
Once the AI has processed your input, the results will be displayed in a visually appealing manner. For textual data, you may see detailed insights and extracted information. If images were uploaded, the application might provide visualizations or answers based on the image content.
""")
            
            st.info("""
                    This application is not intended to be a replacement for the existing Q&A applications.
                    For a comprehensive reference of objects and methods, make sure to explore the official documentation @https://python.langchain.com/docs/get_started/introduction.
                    """)
            
            st.markdown("""
That concludes our walkthrough of the AI-powered Question Answering application. I hope you find it helpful and enjoy exploring the capabilities it offers. Happy querying! 🚀

""")
    _, exp_col, _ = st.columns([1.125,2,1])
    with exp_col:
        st.title("SOP Application 🧙🏻‍♂️")

    
    with st.sidebar:
        with st.expander("**🧙🏻‍♂️ How to Use This Application**"):
            st.markdown("""
                        Upload different types of files ! 🧙🏻‍♂️

                        
Walkthrough: Generative AI Application

Welcome to our Generative AI application! This user-friendly tool is designed to seamlessly handle both textual data and images, providing insightful answers to your queries. Let's take a walkthrough to understand how to make the most of this versatile application.

1. Landing Page:
Upon visiting the application's landing page, you'll be greeted by an inviting interface. The clean layout features a logo and a title that clearly indicates the purpose of the application: "AI-Powered Question Answering."

2. Asking Questions:
To get started, you can type your questions directly into the input box provided on the main section of the page. The application is designed to understand a wide range of queries, so feel free to ask about any topic you have in mind.

3. Textual Data Input:
If you have textual data that you'd like the AI to analyze, there's a dedicated section for uploading files. Whether it's a PDF document, a Word file, or a plain text file, simply click on the "Upload" button and select the file you wish to analyze. The application will process the text and generate relevant insights.

4. Image Input:
For image-based queries, use the separate file uploader designed for images. Upload PNG, JPG, or JPEG files to receive answers based on the content of the images. The application utilizes advanced image processing capabilities to extract meaningful information.

5. Submitting Queries:
After entering your questions or uploading files, you can submit your queries by clicking the relevant "Submit" button. The application will then initiate the AI analysis process, providing you with a comprehensive response.

6. Results:
Once the AI has processed your input, the results will be displayed in a visually appealing manner. For textual data, you may see detailed insights and extracted information. If images were uploaded, the application might provide visualizations or answers based on the image content.
""")
            
            st.info("""
                    This application is not intended to be a replacement for the existing Q&A applications.
                    For a comprehensive reference of objects and methods, make sure to explore the official documentation @https://python.langchain.com/docs/get_started/introduction.
                    """)
            
            st.markdown("""
That concludes our walkthrough of the AI-powered Question Answering application. I hope you find it helpful and enjoy exploring the capabilities it offers. Happy querying! 🚀

""")


    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)

    # Tab 1 content
    with col1:
        st.write("### 🗄 1. Upload docs")
        st.write("This is where you can find information about docs.")
        pdf_docs = st.file_uploader("🤷🏻 1. Upload your Docs", type=['docx', 'pdf', 'txt'], accept_multiple_files=True)
        if st.button("Submit Docs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Success")
        user_question = st.text_input("Hi, what would you like to know?")
        if user_question:
            user_input(user_question)


    # Tab 2 content
    with col2:

        # Tab 2: Upload images
        st.write("### 🚚 2. Upload images")
        st.write("This is where you can find information about your images.")
        imgs = st.file_uploader("🤷🏻 2. Upload your images", type=['.png', 'jpg', 'jpeg'], accept_multiple_files=True)
        find = st.button("Submit Images")
        input_text = st.text_input("Input Prompt: ", key="input")
        for img in imgs:
            if find:
                image=Image.open(img)
                st.image(img, caption="Uploaded Image.", use_column_width=True)
            if input_text is not "":
                image_data=input_image_details(img)
                response=get_gemini_response(input_prompt,image_data,input_text)
                st.subheader(f"The result is for {img.name}")
                st.write(response)

# ### Main() funtion call

# In[13]:


if __name__ == "__main__":
    main()

