from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


def run_groq(model, temperature, input):
    chat = ChatGroq(temperature=temperature, model_name=model)
    system = "You are a helpful assistant."
    human = "{input}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)])
    chain = prompt | chat
    result = chain.invoke({"input": input})
    return result


if __name__ == '__main__':
    result = run_groq("llama-3.1-70b-versatile", 0, "Explain why sky is blue")
    print("Content:")
    print(result.content)

    print("Metadata:")
    print(result.response_metadata)
