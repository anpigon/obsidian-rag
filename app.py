# %%
import operator
import os
from typing import Annotated, List, Literal

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.cache import SQLiteCache
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field

from helper.rag import create_embeddings, create_retriever

set_llm_cache(SQLiteCache(database_path="./.cached/llm_cache.db"))

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "true"

load_dotenv()

logging.langsmith("obsidian-rag")

DOCS_PATH = "./obsidian-help/ko/"
CACHE_PATH = "./.cache/"
STORE_PATH = "./.vectorstore/"


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

#%%
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

#%%
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router
print(question_router.invoke({"question": "What are the types of agent memory?"}))

#%%
class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
structured_llm = llm.with_structured_output(Joke)
structured_llm.invoke("Tell me a joke about cats")

#%%
embeddings = create_embeddings(model_name="BAAI/bge-m3", cache_path=CACHE_PATH)

retriever = create_retriever(path=DOCS_PATH, embeddings=embeddings)

web_search_tool = TavilySearchResults(k=3)


# %%
class GraphState(MessagesState):
    search_query: str  # Search Query for retriever
    question: str  # User question
    generation: str  # LLM generation
    max_retries: int  # Max number of retries for answer generation
    documents: List[str]  # List of retrieved documents
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]


# %%
def generate_retriever_query(state: GraphState) -> GraphState:
    question = state["question"]
    system = """Your task is to interpret the user’s question and generate a detailed and optimized vector database search query. Use the information in the user’s input to create the most effective query possible for retrieving relevant, precise, and insightful results from the database. Be thorough in your reasoning and ensure each step builds toward a comprehensive final search query.

When generating the search query, follow these steps:

1.	Deeply understand the user’s intent: Analyze the user’s input carefully. Identify the key entities, concepts, goals, and any implicit context or information the user might expect. Look for nuanced details that could improve the search.
2.	Query expansion and enrichment: Based on the identified entities or topics, expand the query using synonyms, related concepts, or alternative ways to frame the question. Consider different angles or aspects that could provide a broader and deeper set of results. Ensure the query is flexible enough to capture variations in phrasing or topic scope.
3.	Search optimization and filtering: Refine the query to enhance its relevance and precision. Prioritize the most critical terms and filter out ambiguous, redundant, or irrelevant parts. Balance generality with specificity to achieve an optimized query that is neither too narrow nor too broad.
4.	Context-aware adjustments: Incorporate any domain-specific knowledge or context if applicable. If the question involves specialized terminology, technical language, or jargon, ensure the query reflects this to enhance retrieval accuracy.
5.	Final query creation: Formulate the final query in a structured format best suited for vector-based retrieval systems. The query should be both succinct and comprehensive, effectively capturing the core intent and context of the original input.
6.	Respond with only the final search query in Korean.

final search query:
"""

    response = llm.invoke([("system", system), ("human", question)])
    search_query = StrOutputParser().invoke(response)

    return {"search_query": search_query}


# generate_retriever_query({"question": "옵시디언의 가격은?"})


# %%
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    search_query = state["search_query"]

    # Retrieval
    documents = retriever.invoke(search_query)
    return {"documents": documents, "question": question}


def generate(state: GraphState) -> GraphState:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    message = state["message"]
    documents = state.get("documents", [])

    rag_prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:
""")
    rag_chain = rag_prompt | llm | StrOutputParser()

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    message = [
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "message": message,
    }


def grade_documents(state: GraphState) -> GraphState:
    """
    검색된 문서가 질문과 관련이 있는지 여부를 결정합니다.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

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
    return {"documents": filtered_docs, "question": question}


def transform_query(state: GraphState) -> GraphState:
    """
    쿼리를 변형하여 더 나은 질문을 만듭니다.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state: GraphState) -> GraphState:
    """
    재구문된 질문을 기반으로 웹 검색을 합니다.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}


### Edges ###


# %%
def route_question(state: GraphState) -> GraphState:
    """
     웹 검색 또는 RAG로 라우팅

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""),
            ("human", question),
        ]
    )

    question_router = route_prompt | structured_llm_router
    source = question_router.invoke({"question": question})
    print(source)
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


route_question({"question": "오늘 날씨는?"})


# %%
def decide_to_generate(state: GraphState) -> GraphState:
    """
    답변을 생성할지, 질문을 다시 생성할지 결정합니다.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> GraphState:
    """
    문서 기반으로 정확하게 답변했는지 할루이네이션을 확인합니다.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_grader = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.""",
                ),
                (
                    "human",
                    "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
                ),
            ]
        )
        | structured_llm_grader
    )

    answer_grader = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.""",
                ),
                (
                    "human",
                    "User question: \n\n {question} \n\n LLM generation: {generation}",
                ),
            ]
        )
        | structured_llm_grader
    )

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
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


# %%
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

response = app.invoke({"question": "옵시디언에서 동기화하는 방법은?"})
print(response["generation"])
# %%
