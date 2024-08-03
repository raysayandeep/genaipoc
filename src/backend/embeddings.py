from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector
load_dotenv()


chat = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")

conn = psycopg.connect(dbname='poc_db', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)