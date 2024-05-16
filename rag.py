import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


# Cache the function to load and process PDF documents
@st.cache_resource()
def load_and_process_pdfs(pdf_folder_path):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


# Cache the function to initialize the vector store with documents
@st.cache_resource()
def initialize_vector_store(_splits):
    return FAISS.from_documents(
        documents=_splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    )


pdf_folder_path = ".\PDFs"
splits = load_and_process_pdfs(pdf_folder_path)
vector_store = initialize_vector_store(splits)

prompt_template = """You are an AI expert. You need to answer the question related to AI. 
Given below is the context and question of the user. Don't answer question outside the context provided. Also return some provided metadata.
context = {context}
question = {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, verbose=True)


def format_docs(docs):
    return "".join(
        [
            f"source: {str(doc.metadata['source'])}\nbuilding part id: {doc.metadata['page']}\ncontent: {doc.page_content}\n\n"
            for doc in docs
        ]
    )


rag_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Streamlit app
st.title("AI Expert chatbot")
user_input = st.text_input("Enter your question about AI:", "")
if st.button("Submit"):
    try:
        response = rag_chain.invoke(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")
