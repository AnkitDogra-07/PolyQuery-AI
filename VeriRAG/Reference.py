#!/usr/bin/env python
# coding: utf-8

# ### This is not just RAG. This code implements a self-correcting, guarded RAG agent that tries to solve three hard problems in production RAG systems:
# - Bad retrieval (irrelevant docs)
# - Poor question formulation
# - Hallucinated answers

# | Capability                 | Present | How                                              |
# | -------------------------- | ------- | ------------------------------------------------ |
# | Agentic RAG                | ✅       | LangGraph state machine                          |
# | Tool-based RAG             | ⚠️      | Retriever wrapped as tool (earlier), direct here |
# | Self-correcting RAG        | ✅       | Query rewrite + retry                            |
# | Retrieval grading          | ✅       | LLM-based document relevance grader              |
# | Generation grading         | ✅       | Hallucination + answer graders                   |
# | Multi-stage RAG            | ✅       | Retrieve → Filter → Generate → Validate          |
# | Guardrails                 | ✅       | Binary gates everywhere                          |
# | Deterministic control flow | ✅       | Explicit graph transitions                       |
# | Hybrid retrieval           | ❌       | Dense only                                       |
# | Web search                 | ❌       | No external search                               |
# | Re-ranking                 | ⚠️      | Binary filter, not ranking                       |
# | Metrics-based eval         | ❌       | LLM-as-judge only                                |
# 

# In[ ]:


get_ipython().system('pip install python-dotenv langchain langchain-core langchain-community langchain-google-genai chromadb langchain-text-splitters beautifulsoup4 sentence-transformers einops langchainhub langsmith')


# In[49]:


import os
from dotenv import load_dotenv
import warnings # Suppress all warnings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field    # A Pydantic helper to add metadata to a field
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCivlT_iEMFOTjyU4YCtIKzgktzXjLe22g"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

warnings.filterwarnings("ignore")


# In[50]:


embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code": True})
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
print(embeddings)
print(llm)
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)


# In[51]:


# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()
print(retriever)

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]
print(vectorstore)


# #### Let's look into the retriever grader

# In[52]:


# Data model
# A Pydantic schema, Defines the expected output format from the LLM, so our LLM can give us different answers like this documetntaion is not related and all...
# so to restricts the model response we have written this and as it makes our code less fragile
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeDocuments)


system = """You are a grader checking if a document is relevant to a user’s question.The check has to be done very strictly..  
If the document has words or meanings related to the question, mark it as relevant.  
Give a simple 'yes' or 'no' answer to show if the document is relevant or not."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content



print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


# ### let's look into the data generation 

# In[54]:


### Generate
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser

client = Client()

prompt = client.pull_prompt("rlm/rag-prompt")
# print(prompt)

prompt.pretty_print()


# In[55]:


rag_chain = prompt | llm


# In[56]:


# Run
generation = rag_chain.invoke({"context": docs, "question": question})


# In[57]:


generation.pretty_print()


# ### Hallucination Grader

# In[59]:


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeHallucinations)


# In[60]:


# Prompt
system = """You are a grader checking if an LLM generation is grounded in or supported by a set of retrieved facts.  
Give a simple 'yes' or 'no' answer. 'Yes' means the generation is grounded in or supported by a set of retrieved the facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)


# In[61]:


hallucinations_grader = hallucination_prompt | structured_llm_grader
hallucinations_grader.invoke({"documents": docs, "generation": generation})


# #### Answer Grader

# In[62]:


### Answer Grader
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

question = "agent memory"

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})


# ### Question Re-writer

# In[63]:


system = """You are a question re-writer that converts an input question into a better optimized version for vector store retrieval document.  
You are given both a question and a document.  
- First, check if the question is relevant to the document by identifying a connection or relevance between them.  
- If there is a little relevancy, rewrite the question based on the semantic intent of the question and the context of the document.  
- If no relevance is found, simply return "question not relevant."  
Your goal is to ensure the rewritten question aligns well with the document for better retrieval."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human","""Here is the initial question: \n\n {question} \n,
             Here is the document: \n\n {documents} \n ,
             Formulate an improved question. if possible other return 'question not relevant'."""
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()


# In[64]:


question="who is a current indian prime minister?"


# In[65]:


question_rewriter.invoke({"question":question,"documents":docs})


# ### From here the Langgraph workflow will start

# In[66]:


from typing import List
from typing_extensions import TypedDict
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    filter_documents: List[str]
    unfilter_documents: List[str]


# 

# In[67]:


def retrieve(state:AgentState):
    print("----RETRIEVE----")
    question=state['question']
    documents=retriever.invoke(question)
    return {"documents": documents, "question": question}


# In[69]:


def grade_documents(state:AgentState):
    print("----CHECK DOCUMENTS RELEVANCE TO THE QUESTION----")
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    unfiltered_docs = []
    for doc in documents:
        score=retrieval_grader.invoke({"question":question, "document":doc})
        grade=score.binary_score

        if grade=='yes':
            print("----GRADE: DOCUMENT RELEVANT----")
            filtered_docs.append(doc)
        else:
            print("----GRADE: DOCUMENT NOT RELEVANT----")
            unfiltered_docs.append(doc)
    if len(unfiltered_docs)>1:
        return {"unfilter_documents": unfiltered_docs,"filter_documents":[], "question": question}
    else:
        return {"filter_documents": filtered_docs,"unfilter_documents":[],"question": question}




# In[70]:


def decide_to_generate(state:AgentState):
    print("----ACCESS GRADED DOCUMENTS----")
    state["question"]
    unfiltered_documents = state["unfilter_documents"]
    filtered_documents = state["filter_documents"]


    if unfiltered_documents:
        print("----ALL THE DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY----")
        return "transform_query"
    if filtered_documents:
        print("----DECISION: GENERATE----")
        return "generate"



# In[71]:


def generate(state:AgentState):
    print("----GENERATE----")
    question=state["question"]
    documents=state["documents"]

    generation = rag_chain.invoke({"context": documents,"question":question})
    return {"documents":documents,"question":question,"generation":generation}


# In[72]:


from langgraph.graph import END, StateGraph, START
def transform_query(state:AgentState):
    question=state["question"]
    documents=state["documents"]

    print(f"this is my document{documents}")
    response = question_rewriter.invoke({"question":question,"documents":documents})
    print(f"----RESPONSE---- {response}")
    if response == 'question not relevant':
        print("----QUESTION IS NOT AT ALL RELEVANT----")
        return {"documents":documents,"question":response,"generation":"question was not at all relevant"}
    else:   
        return {"documents":documents,"question":response}


# In[73]:


def decide_to_generate_after_transformation(state:AgentState):
    question=state["question"]

    if question=="question not relevant":
        return "query_not_at_all_relevant"
    else:
        return "Retriever"


# In[74]:


import pprint
def grade_generation_vs_documents_and_question(state:AgentState):
    print("---CHECK HELLUCINATIONS---")
    question= state['question']
    documents = state['documents']
    generation = state["generation"]

    score = hallucinations_grader.invoke({"documents":documents,"generation":generation})

    grade = score.binary_score

    #Check hallucinations
    if grade=='yes':
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        print("---GRADE GENERATION vs QUESTION ---")

        score = answer_grader.invoke({"question":question,"generation":generation})

        grade = score.binary_score

        if grade=='yes':
            print("---DECISION: GENERATION ADDRESS THE QUESTION ---")
            return "useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---TRANSFORM QUERY")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---TRANSFORM QUERY")
        "not useful"



# In[75]:


from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(AgentState)
workflow.add_node("Docs_Vector_Retrieve", retrieve)
workflow.add_node("Grading_Generated_Documents", grade_documents) 
workflow.add_node("Content_Generator", generate)
workflow.add_node("Transform_User_Query", transform_query)


# In[76]:


workflow.add_edge(START, "Docs_Vector_Retrieve")


# In[77]:


workflow.add_edge("Docs_Vector_Retrieve","Grading_Generated_Documents")


# In[78]:


workflow.add_conditional_edges("Grading_Generated_Documents",
                            decide_to_generate,
                            {
                            "generate": "Content_Generator",
                            "transform_query": "Transform_User_Query"
                            }
                            )


# In[79]:


workflow.add_conditional_edges("Content_Generator",
                            grade_generation_vs_documents_and_question,
                            {
                            "useful": END,
                            "not useful": "Transform_User_Query",
                            }
)


# In[80]:


workflow.add_conditional_edges("Transform_User_Query",
                  decide_to_generate_after_transformation,
                  {"Retriever":"Docs_Vector_Retrieve",
                   "query_not_at_all_relevant":END})


# In[81]:


app = workflow.compile()


# In[82]:


from IPython.display import Image, display # type: ignore
display(Image(app.get_graph(xray=True).draw_mermaid_png()))


# In[83]:


inputs = {"question": "Explain how the different types of agent memory work?"}


# In[84]:


app.invoke(inputs)["generation"]


# In[85]:


inputs = {"question": "who is a prompt engineering?"}


# In[86]:


app.invoke(inputs)


# In[87]:


inputs = {"question": "what is role of data structure while creating agentic pattern?"}


# In[88]:


app.invoke(inputs)

