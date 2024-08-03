from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from database import Base

class Document(Base):
    __tablename__='documents'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    createdate = Column(DateTime)

class Conversation(Base):
    __tablename__='conversations'
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String)
    answer = Column(String)
    file_id = Column(Integer, ForeignKey("documents.id"))

