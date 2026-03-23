#!/usr/bin/env python
# coding: utf-8

# In[140]:


get_ipython().system('pip install python-dotenv langchain langchain-core langchain-community langchain-google-genai chromadb langchain-text-splitters beautifulsoup4 sentence-transformers einops langchainhub langsmith faiss-cpu pydantic rank_bm25 sentence-transformers')


# ## Declaration

# In[141]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import TypedDict, List
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCivlT_iEMFOTjyU4YCtIKzgktzXjLe22g"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
warnings.filterwarnings("ignore")



class QueryOutput(BaseModel):
    rewritten_query: str
    expanded_queries: List[str]
    step_back_query: str

class QueryOptimizerOutput(BaseModel):
    rewritten_query: str
    expanded_queries: List[str]
    step_back_query: str

class GraphState(TypedDict):
    query: str
    rewritten_query: str
    expanded_queries: List[str]
    step_back_query: str
    documents: List[Document]
    answer: str
    iteration: int
    retrieval_feedback: dict
    doc_scores: List[float]

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
structured_llm = llm.with_structured_output(QueryOutput)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ## Query Rewritting & Expansion
# 
# * User Query  ->  Query Rewriting  ->  Step-Back Query  ->  Query Expansion
# 
# 
# | Step      | Why                                      |
# | --------- | ---------------------------------------- |
# | Rewrite   | cleans the query and removes ambiguity   |
# | Step-back | captures **higher-level concept**        |
# | Expansion | generates **multiple search variations** |
# 

# In[142]:


optimizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert query optimizer. Return ONLY JSON."),
    ("human", """
        User Query:
        {query}

        Perform:
        1. Rewrite query
        2. Generate 6 variations
        3. Generate step-back query

        Return JSON:

        {{
        "rewritten_query": "...",
        "expanded_queries": ["...", "...", "...", "..."],
        "step_back_query": "..."
        }}
    """)
])

def query_intelligence_node(state):

    iteration = state.get("iteration", 0)

    # PASS 1 (normal)
    if iteration == 0:
        query = state["query"]

        prompt = optimizer_prompt.invoke({
            "query": query
        })

    # PASS 2 (feedback-driven)
    else:
        feedback = state.get("retrieval_feedback", {})

        feedback_text = f"""
        Previous retrieval failed.

        Reason: {feedback.get("reason", "low relevance")}
        Max Score: {feedback.get("max_score")}
        Avg Score: {feedback.get("avg_score")}
        """

        refined_query = f"""
        {feedback_text}

        Original Query: {state["query"]}
        Previous Rewritten Query: {state.get("rewritten_query")}

        Refine the query to improve retrieval.
        Make it more specific and include better keywords.
        """

        prompt = optimizer_prompt.invoke({
            "query": refined_query
        })

    result = structured_llm.invoke(prompt)

    print(f"\n[Iteration {iteration}]")
    print("Rewritten:", result.rewritten_query)

    return {
        "rewritten_query": result.rewritten_query,
        "expanded_queries": result.expanded_queries,
        "step_back_query": result.step_back_query
    }




# VECTOR-DB, EMBEDDINGS & INGESTION

# In[143]:


embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code": True})

documents = [
    Document(page_content="RAG systems retrieve documents before generation."),
    Document(page_content="Faithfulness measures whether answers are grounded in retrieved documents."),
    Document(page_content="Query expansion improves recall in retrieval systems."),
]

vectorstore = Chroma.from_documents(
    documents=documents,
    collection_name="verirag-chroma",
    embedding=embeddings,
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5


# RETRIEVER

# In[144]:


def hybrid_retrieve_node(state):
    queries = (
        [state["rewritten_query"]] +
        state["expanded_queries"] +
        [state["step_back_query"]]
    )

    all_docs = []
    print("----------------------------")
    for q in queries:
        print("Query: ",q)
        all_docs.extend(bm25_retriever.invoke(q))
        all_docs.extend(vector_retriever.invoke(q))
        print(all_docs)

    # Deduplicate
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    print(unique_docs)

    return {"documents": unique_docs}


# ## Cross Encoder & Re-Ranking

# In[145]:


def rerank_and_filter_node(state):

    docs = state.get("documents", [])
    query = state["query"]

    if not docs:
        return {
            "documents": [],
            "answer": "I don't know",
            "retrieval_feedback": {"reason": "no_docs"}
        }

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    top_docs = doc_scores[:5]
    top_scores = [score for _, score in top_docs]

    max_score = max(top_scores)
    avg_score = sum(top_scores) / len(top_scores)

    threshold = max_score * 0.75

    if max_score < threshold:
        return {
            "documents": [],
            "doc_scores": top_scores,
            "retrieval_feedback": {
                "reason": "low_relevance",
                "max_score": float(max_score),
                "avg_score": float(avg_score)
            }
        }

    return {
        "documents": [doc for doc, _ in top_docs],
        "doc_scores": top_scores
    }


# In[151]:


def retry_decision_node(state):

    iteration = state.get("iteration", 0)
    max_iterations = 2

    # If already answered → stop
    if state.get("answer") == "I don't know" and iteration >= max_iterations:
        return "generate"

    # If no docs → retry
    if not state.get("documents") and iteration < max_iterations:
        return "retry"

    return "generate"

def Refine_query_node(state):
    return {
        "iteration": state.get("iteration", 0) + 1
    }


# ## GENERATION

# In[147]:


generation_prompt = ChatPromptTemplate.from_template(
"""
You are a strict assistant.

Answer ONLY using the provided context.
If the answer is not clearly present, say "I don't know".

Question:
{query}

Context:
{context}

Answer:
"""
)

def generate_answer_node(state):

    # If reranker already decided to stop
    if state.get("answer") == "I don't know":
        return {"answer": "I don't know"}

    docs = state.get("documents", [])

    if not docs:
        return {"answer": "I don't know"}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = generation_prompt.format(
        query=state["query"],
        context=context
    )

    response = llm.invoke(prompt)

    return {"answer": response.content}


# ## Graph Compliation

# In[ ]:


builder = StateGraph(GraphState)

builder.add_node("query_intelligence", query_intelligence_node)
builder.add_node("hybrid_retrieve", hybrid_retrieve_node)
builder.add_node("rerank", rerank_and_filter_node)
builder.add_node("retry_decision", retry_decision_node)
builder.add_node("Refine_query", Refine_query_node)
builder.add_node("generate", generate_answer_node)

builder.set_entry_point("query_intelligence")

builder.add_edge("query_intelligence", "hybrid_retrieve")
builder.add_edge("hybrid_retrieve", "rerank")
builder.add_edge("rerank", "retry_decision")

builder.add_conditional_edges(
    "retry_decision",
    retry_decision_node,
    {
        "retry": "Refine_query",
        "generate": "generate"
    }
)

builder.add_edge("Refine_query", "query_intelligence")


builder.add_edge("Refine_query", "query_intelligence")
builder.add_edge("generate", END)

graph=builder.compile()

from IPython.display import Image, display # type: ignore
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))


# ## Inferencing RAG

# In[ ]:


result = graph.invoke({
    # "query": "How do we measure faithfulness in computer terminal?"
    "query": "How do we measure faithfulness in RAG?",
    "iteration": 0
})
print("-----------------------------")
print(result["answer"])

