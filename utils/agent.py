"""
agent.py
=========================
LangGraph-based RAG agent for MP2.

Pipeline:
1) retrieve(state)        -> collects context via Pyserini BM25 (+ optional compression)
2) generate_answer(state) -> prompts an LLM (Ollama) with the retrieved context

State keys (GraphState):
- question: str                     # user question (required)
- context: List[str]                # accumulated evidence passages
- retriever: Optional[BaseRetriever]# allow injection (for testing)
- final_answer: Optional[str]       # model's answer
- error: Optional[str]              # error message if any
- current_step: str                 # "retrieve" | "generate_answer"
"""

import logging
from typing import List
import re
from typing import List, Dict

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate

from utils.llm import OllamaLLM
from utils.retriever import create_retriever
from utils.state import GraphState
from utils.config import Config
class Colors:
    BLUE = '\033[38;2;157;176;206m'      # 9db0ce
    PURPLE = '\033[38;2;116;121;155m'    # 74799B
    CYAN = '\033[38;2;184;216;227m'      # b8d8e3
    MAUVE = '\033[38;2;206;160;170m'     # cea0aa
    PINK = '\033[38;2;254;225;221m'      # fee1dd
    ROSE = '\033[38;2;233;194;197m'      # e9c2c5
    DUSTY = '\033[38;2;212;165;165m'     # d4a5a5
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()


# ---------------------------------------------------------------------
# Graph construction (entry for main.py)
# ---------------------------------------------------------------------
def create_agent():
    """
    Build and compile the LangGraph workflow:
       retrieve -> generate_answer -> END
    """
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # Entry and edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

def decompose_query(query: str) -> List[str]:
    """Use LLM to decompose a complex query into 2-4 sub-queries."""
    llm = OllamaLLM(model=config.OLLAMA_MODEL)
    
    prompt = ChatPromptTemplate.from_template("""
You are a query decomposition assistant. Break down the following complex question into 2-4 simpler, focused sub-queries.

Original question: {query}

Generate 2-4 sub-queries that cover different aspects of the original question. 
Format: Output only the sub-queries, one per line, numbered.

Sub-queries:""")
    
    chain = prompt | llm
    response = chain.invoke({"query": query})
    
    lines = response.strip().split('\n')
    sub_queries = []
    for line in lines:
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        if cleaned and len(cleaned) > 10:  # Valid sub-query
            sub_queries.append(cleaned)
    
    return sub_queries[:4] if sub_queries else [query]


def fuse_results(results_by_query: Dict[str, List]) -> List[str]:
    """
    Fuse retrieval results from multiple sub-queries.
    Remove duplicates and rank by frequency/score.
    """
    doc_scores = {}  # docid -> (content, total_score, count)
    
    for sub_query, docs in results_by_query.items():
        for doc in docs:
            if isinstance(doc, dict):
                docid = doc.get("docid", str(doc))
                content = doc.get("content", "")
                score = doc.get("score", 0.0)
            else:
                docid = str(doc)
                content = doc if isinstance(doc, str) else str(doc)
                score = 1.0
            
            if docid in doc_scores:
                old_content, old_score, count = doc_scores[docid]
                doc_scores[docid] = (old_content, old_score + score, count + 1)
            else:
                doc_scores[docid] = (content, score, 1)
    
    ranked = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    
    return [content for content, score, count in ranked[:10]]



# ---------------------------------------------------------------------
# Node 1: Retrieval
# ---------------------------------------------------------------------

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve evidence passages. If query is long, decompose and fuse results.
    """
    try:
        logger.info("Starting retrieval")
        
        query = (state.get("question") or "").strip()
        if not query:
            state["error"] = "Empty question"
            state["current_step"] = "retrieve"
            return state
        
        retriever = state.get("retriever") or create_retriever()
        
        sent = re.split(r'[.!?]+', query.strip())
        sent_count = len([s for s in sent if s.strip()])
        if len(query) > 200 or sent_count > 1:
            if len(query) > 200:
                logger.info("This query is longer than 200 characters!! Decomposing into the following subqueries")
            else:
                logger.info("This query includes more than one sentence!! Decomposing into the following subqueries")
            
            sub_queries = decompose_query(query)
            print(f"{Colors.PURPLE}subqueries:{Colors.RESET}")
            for i, sq in enumerate(sub_queries, 1):
                print(f"{Colors.PURPLE}  {i}. {sq}{Colors.RESET}")         
            
            # removed decomp for now to decrease runtime
            results_by_query = {}
            '''for sq in sub_queries:
                docs = retriever.invoke(sq)
                if not docs and hasattr(retriever, "base_retriever"):
                    docs = retriever.base_retriever.invoke(sq)
                results_by_query[sq] = docs
            fused_context = fuse_results(results_by_query)
            state["context"] = fused_context'''
            
            docs = retriever.invoke(query)
            if not docs and hasattr(retriever, "base_retriever"):
                docs = retriever.base_retriever.invoke(query)

            ctx = [getattr(d, "page_content", str(d)) for d in docs]
            state["context"] = ctx
            
            
        else:#normal retireval
            docs = retriever.invoke(query)
            if not docs and hasattr(retriever, "base_retriever"):
                docs = retriever.base_retriever.invoke(query)
            
            ctx = [getattr(d, "page_content", str(d)) for d in docs]
            state["context"] = ctx
        
        state["current_step"] = "retrieve"
        return state
        
    except Exception as e:
        logger.exception("Error in retrieve")
        state["error"] = str(e)
        state["current_step"] = "retrieve"
        return state



# ---------------------------------------------------------------------
# Helpers for prompting
# ---------------------------------------------------------------------
def build_prompt(max_sentences: int = 7) -> ChatPromptTemplate:
    """
    Create the QA prompt.

    Notes:
    - You can experiment with different instruction styles here (e.g., chain-of-thought
      vs. concise answers, citing sources, etc.).
    """
    template = f"""You are an assistant for question answering.

Use the context below to answer the question. Make reasonable inferences based on the information provided.


Context:
{{context}}

Question:
{{question}}

Answer in at most {max_sentences} sentences, concise and to the point.
Answer:"""
    return ChatPromptTemplate.from_template(template)


# ---------------------------------------------------------------------
# Node 2: Answer generation
# ---------------------------------------------------------------------
def generate_answer(state: GraphState) -> GraphState:
    """
    Generate a concise answer using Ollama with the retrieved context.
    Safely stringifies all context items before prompting.

    Notes:
    - In Task 3, the retrieval step may store *structured* context in state["context"],
      e.g. a list like:
          [
            {"subquery": "what is RAG", "results": ["RAG is ...", "A RAG agent ..."]},
            {"subquery": "how do retrievers and vector DBs work together",
             "results": ["retriever uses embeddings ...", "vector DB stores ..."]},
            "Doc from baseline run"
          ]
      You can first turn each structured item into a text block like
          "Sub-query: ...\n---\nretrieved docs...\n---"
      in the retrieval step, which is compatible with the original format.
    """
    try:
        logger.info("Starting answer generation")

        # Initialize local LLM (Ollama)
        llm = OllamaLLM(model=config.OLLAMA_MODEL)

        ########## Prepare context string ##########
        # Flatten state["context"] into one string for the LLM.
        # If later you have multiple sub-queries, you can format them here.
        context_strings: List[str] = []
        for item in state.get("context") or []:
            context_strings.append(item if isinstance(item, str) else str(item))
        context_strings = "\n".join(context_strings)
        ###########################################

        # Build prompt and call the model
        prompt = build_prompt(max_sentences=7)
        chain = prompt | llm

        response = chain.invoke({
            "question": state.get("question", ""),
            "context": context_strings
        })

        # Update state
        state["final_answer"] = response
        state["current_step"] = "generate_answer"
        logger.info("Answer generation completed")
        return state

    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        state["error"] = str(e)
        state["current_step"] = "generate_answer"
        return state
