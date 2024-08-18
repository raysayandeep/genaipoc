from langchain_core.prompts import ChatPromptTemplate

def docGraderPrompt():
    grader_system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
    )
    return grade_prompt

def questionRewriterPrompt():
    system = """You are a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval without any preamble or explanation. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate only one improved question.",
        ),
    ])
    return re_write_prompt

def ragPrompt():
    system = """You are a system to answer the following question.\n
     Answer the following question based only on the provided context without any preamble or explanation.
     """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ( 
            "human", """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:" "
        """)])
    return prompt

def hallucinationGraderPrompt():
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
    )
    return hallucination_prompt

def answerGraderPrompt():
    system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
    )
    return answer_prompt