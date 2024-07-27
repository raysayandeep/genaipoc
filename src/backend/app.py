from flask import Flask
import os
import sys

path = os.getcwd()
out = os.path.abspath(os.path.join(path, os.pardir))
sys.path.insert(0, out+'\\llm')
app = Flask(__name__)
import llm_groq

@app.route("/")
def hello_world():
    path = sys.path
    return path


@app.route("/llm")
def hello(llm="llama-3.1-70b-versatile", temp=0, question="Explain why sky is blue in 20 words"):
    result = llm_groq.run_groq(llm, temp, question)
    response = {}
    response['content'] = result.content
    response['metadata'] = result.response_metadata
    return response
