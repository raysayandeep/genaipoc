from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Literal
from langchain_aws import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

def initBedrockEmbedding():
    embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    return embedding

def initOllamaEmbedding():
    embedding = OllamaEmbeddings(model="llama3")
    return embedding

def customPDFLoader(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    chunk_size = 500  # Number of characters per chunk
    chunk_overlap = 50  # Number of characters to overlap between chunks
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n", " ", ""],is_separator_regex=False)
    chunks = splitter.split_documents(documents)
    return chunks

def faissVectorStore(embedding,documents):
    vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding,
    )
    return vectorstore

def vectorRetriever(vectorstore):
    return vectorstore.as_retriever()

def initLlm():
    llm =ChatGroq(model_name="Gemma2-9b-It")
    return llm