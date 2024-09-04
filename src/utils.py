from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Literal
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
    collection_name = filename

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
    except Exception:
        return False
    
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
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})

def initLlm():
    llm =ChatGroq(model_name="Gemma2-9b-It")
    return llm

if __name__ == '__main__':
    
    

    """PostgreSQL"""
    #filename = "sample_m.pdf"
    #documents = customPDFLoader(filename)
    """ filename = 'saledata.xlsx'
    embedding = initfakeEmbedding()
    connection = "postgresql://postgres:sdr@localhost:5432/postgres"
    vector_store = postgresVectorStore(embedding,filename,connection)
    documents = customExcelLoader(filename)
    if documents:
        if vector_store:
            success = addDataToPostgresVectorStore(vector_store,documents)
            print(success)
            if success:
                print("Data Saved")
    else:
        print("Error") """
    
        

    """FAISS"""
    """ filename = "sample_m.pdf"
    documents = customPDFLoader(filename)
    embedding = initfakeEmbedding()
    vector_store = createFaissVectorStore(embedding,filename)
    if vector_store:
        addDataToFaissVectorStore(vector_store,documents)
        saved = saveFaissVectorStore(vector_store,filename)
        if saved:
            print("Data Saved") """

    """Retrieval"""
    filename = "sample_m.pdf"
    embedding = initfakeEmbedding()
    #vector_store = loadFaissVectorStore(filename,embedding)
    connection = "postgresql://postgres:sdr@localhost:5432/langchain"
    vector_store = postgresVectorStore(embedding,filename,connection)
    if vector_store:
        retriever = vectorRetriever(vector_store)
        print(retriever.invoke("latex"))
    

   
    