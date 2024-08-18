from langgraph.graph import END, StateGraph, START
from graph import GraphState,GradeDocuments,GradeHallucinations,GradeAnswer
from utils import initBedrockEmbedding,initOllamaEmbedding,customPDFLoader,faissVectorStore,vectorRetriever,initLlm
from prompts import docGraderPrompt,ragPrompt,hallucinationGraderPrompt,answerGraderPrompt,questionRewriterPrompt
from IPython.display import Image, display
from langchain_core.output_parsers import StrOutputParser

def retrieveDocument(state):
    print("---RETRIEVE---")
    question = state["question"]
    if not state["counter"]:
        counter = 0
    else:
        counter = state["counter"]
    print("Question:",question)
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question,"counter": counter + 1}

def gradeDocuments(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    counter = state["counter"]
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question, "counter": counter}

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    if state["counter"] < 4:
        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "not relevant"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "relevant"
    else:
        print("---DECISION: not found---")
        return "unsuccessful"

def generateResponse(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    counter = state["counter"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation,"counter": counter + 1}

def transformQuery(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    counter = state["counter"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question.content, "counter": counter}

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    # Check hallucination
    if state["counter"] < 8:
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    print("---DECISION: not found---")
    return "unsuccessful"

#embedding = initBedrockEmbedding()
embedding = initOllamaEmbedding()
llm = initLlm()
structured_llm_grader_doc = llm.with_structured_output(GradeDocuments)
grade_prompt = docGraderPrompt()
retrieval_grader = grade_prompt | structured_llm_grader_doc
rag_prompt = ragPrompt()
rag_chain = rag_prompt | llm
re_write_prompt = questionRewriterPrompt()
question_rewriter = re_write_prompt | llm
structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
hallucination_prompt = hallucinationGraderPrompt()
hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination
structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
answer_grader_prompt = answerGraderPrompt()
answer_grader = answer_grader_prompt | structured_llm_grader_answer

if __name__ == '__main__':

    file_path = "samples/sample_m.pdf"
    docs = customPDFLoader(file_path)
    vectorstore = faissVectorStore(embedding,docs)
    retriever = vectorRetriever(vectorstore)
    #question = "What are the Reserved characters"
    question = "what is ispell"

    # lang Graph workflow mapping
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieveDocument)
    workflow.add_node("doc_grader", gradeDocuments)
    workflow.add_node("generate", generateResponse)
    workflow.add_node("transform_query",transformQuery)

    workflow.add_edge(START,"retrieve")
    workflow.add_edge("retrieve","doc_grader")
    workflow.add_conditional_edges(
        "doc_grader",
        decide_to_generate,
        {
            "not relevant":"transform_query",
            "relevant":"generate",
            "unsuccessful":END,
        }
        )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
            "unsuccessful":END,
        },
        )
    app = workflow.compile()


    img = Image(app.get_graph().draw_mermaid_png())
    with open("src/graph.png", "wb") as f:
        f.write(img.data)

    inputs = {"question": question}

    for event in app.stream(inputs):
        for key, value in event.items():
            if key == 'generate':
                print("Question:",value["question"])
                print("Answer:",value["generation"].content)
