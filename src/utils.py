from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Literal
import os
from langchain_aws import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
import faiss
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import pandas as pd

from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

def initBedrockEmbedding():
    embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    return embedding

def initOllamaEmbedding():
    embedding = OllamaEmbeddings(model="llama3")
    return embedding

def initfakeEmbedding():
    embedding = FakeEmbeddings(size=4096)
    return embedding

def customPDFLoader(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    chunk_size = 500  # Number of characters per chunk
    chunk_overlap = 50  # Number of characters to overlap between chunks
    separators=["\n\n","\n", " ", ""] # seperators
    #splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n", " ", ""],is_separator_regex=False)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

def customCSVLoader(filepath):
    documents = []
    file_list = []
    if os.path.exists(filepath):
        file_list = processExcel(filepath)
    for file in file_list:
        loader = CSVLoader(file_path = file,csv_args={"delimiter": ",","quotechar": '"'})
        docs = loader.load()
        for i in range(len(docs)):
            documents.append(docs[i])
    
    return documents



def customExcelLoader(filepath):
    try:
        loader = UnstructuredExcelLoader(filepath, mode="elements")
        docs = loader.load()
        return docs
    except Exception as e:
        print(e)
        return False

def postgresVectorStore(embedding,filename,connection):
    #connection = "postgresql+psycopg://langchain:langchain@localhost:5433/langchain"

    connection = connection
    collection_name = 'vector_store'

    vector_store = PGVector(
    embeddings=embedding,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    )
    return vector_store

def addDataToPostgresVectorStore(vector_store,documents):
    uuids = [str(uuid4()) for _ in range(len(documents))]
    try:
        ind = vector_store.add_documents(documents, ids=uuids)
        return ind
    except Exception as e:
        return e
    
def faissVectorStoreOld(embedding,documents):
    vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding,
    )
    return vectorstore

def createFaissVectorStore(embedding,filename):
    index_legth = len(embedding.embed_query(filename))
    index = faiss.IndexFlatL2(index_legth)
    try:
        vector_store = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            )
        return vector_store
    except Exception:
        return False

def addDataToFaissVectorStore(vector_store,documents):
    uuids = [str(uuid4()) for _ in range(len(documents))]
    try:
        vector_store.add_documents(documents,ids=uuids)
        return True
    except Exception:
        return False

def saveFaissVectorStore(vector_store,filename):
    if vector_store:
        try:
            vector_store.save_local(filename+"dbindex")
            return True
        except Exception:
            return False
        
def loadFaissVectorStore(filename,embedding):
    dbname = filename+"dbindex"
    try:
        vector_store = FAISS.load_local(dbname, embedding, allow_dangerous_deserialization=True)
        return vector_store
    except Exception:
        return False

def vectorRetriever(vector_store):
    #return vectorstore.as_retriever()
    #return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 100})

def initLlm():
    llm =ChatGroq(model_name="Gemma2-9b-It")
    return llm

def processExcel(filepath):
    file_list = []
    if os.path.exists(filepath):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        df_data = pd.read_excel(filepath,sheet_name=None)
        if isinstance(df_data, dict):
            if not os.path.exists(f"temp/{filename}"):
                os.mkdir(f"temp/{filename}")
            for key in df_data.keys():
                new_file = f'temp/{filename}/{filename}_{key}.csv'
                df_data[key].to_csv(new_file,index=False)
                file_list.append(new_file)
    return file_list

if __name__ == '__main__':
    
    

    """PostgreSQL"""
    """ filepath = "../samples/SampleData.xlsx"
    filename = os.path.basename(filepath)

    #documents = customPDFLoader(filename)
    documents = customExcelLoader(filepath)
    #documents = customCSVLoader(filepath)
    #print(*(docs.page_content for docs in documents))
    print(type(documents))
    embedding = initOllamaEmbedding()
    connection = "postgresql://postgres:sdr@localhost:5432/postgres"
    vector_store = postgresVectorStore(embedding,filename,connection)
    if documents:
        if vector_store:
            success = addDataToPostgresVectorStore(vector_store,documents)
            print(success)
            if success:
                print("Data Saved")
    else:
        print("Error") """
    
        

    """FAISS"""
    """ filepath = "../samples/SampleData.xlsx"
    filename = os.path.basename(filepath)
    documents = customCSVLoader(filepath)
    embedding = initOllamaEmbedding()
    vector_store = createFaissVectorStore(embedding,filename)
    if vector_store:
        addDataToFaissVectorStore(vector_store,documents)
        saved = saveFaissVectorStore(vector_store,filename)
        if saved:
            print("Data Saved") """

    """Retrieval"""
    filepath = "../samples/SampleData.xlsx"
    filename = os.path.basename(filepath)
    
    embedding = initOllamaEmbedding()
    #vector_store = loadFaissVectorStore(filename,embedding)
    connection = "postgresql://postgres:sdr@localhost:5432/postgres"
    vector_store = postgresVectorStore(embedding,filename,connection)
    #vector_store = loadFaissVectorStore(filename,embedding)
    if vector_store:
        print("Vector OK")
        retriever = vectorRetriever(vector_store)
        
    
        groq_api_key = ""
        llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-70b-versatile")

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        )
        question="Waht is pencil unit price for central region"
        #print(retriever.invoke(question))
        print(rag_chain.invoke(question).content)



   
    