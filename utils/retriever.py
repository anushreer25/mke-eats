"""
retriever.py
=========================
Implements a BM25-based retriever using Pyserini for MP2 (LangGraph RAG Agent).

This module:
1. Preprocesses the raw text corpus into JSON files compatible with Pyserini.
2. Builds a Lucene index.
3. Wraps Pyserini’s BM25 retriever into a LangChain retriever.
4. Adds an optional contextual compression layer using an LLM.
"""

import os
import json
import logging
import subprocess
from tqdm import tqdm
from typing import List, Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pyserini.search.lucene import LuceneSearcher
from pydantic import PrivateAttr
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
logger = logging.getLogger(__name__)
config = Config()

# ---------------------------------------------------------------------
# 1. Define a LangChain-compatible BM25 Retriever
# ---------------------------------------------------------------------
class PyseriniBM25Retriever(BaseRetriever):
    """
    A LangChain retriever built on top of Pyserini’s BM25 searcher.
    It retrieves the top-k documents for a given query. You can change
    to your MP1 retriever here.
    """

    index_dir: str                     # Path to the Lucene index
    k: int = 5                         # Number of results to return
    k1: float = 0.9                    # BM25 hyperparameter
    b: float = 0.4                     # BM25 hyperparameter
    _searcher: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Initialize Pyserini searcher
        searcher = LuceneSearcher(self.index_dir)
        searcher.set_bm25(self.k1, self.b)
        self._searcher = searcher

        logger.info(
            f"PyseriniBM25Retriever ready — index={self.index_dir}, "
            f"k={self.k}, k1={self.k1}, b={self.b}"
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return top-k relevant documents for a given query.

        Notes:
        - This method is called once per query.
        - For multi-subquery retrieval in Task 3, call this method multiple times
          (one per sub-query) in the upper-level node and then fuse the results.
        """
        hits = self._searcher.search(query, k=self.k)
        docs: List[Document] = []

        for h in hits:
            raw = self._searcher.doc(h.docid).raw() or ""
            text, title = raw, None
            try:
                obj = json.loads(raw)
                text = obj.get("contents") or obj.get("text") or raw
                title = obj.get("title")
            except Exception:
                pass

            docs.append(
                Document(
                    page_content=text,
                    metadata={"docid": h.docid, "score": float(h.score), "title": title},
                )
            )
        return docs


# ---------------------------------------------------------------------
# 2. Helper functions for preprocessing and indexing
# ---------------------------------------------------------------------
def preprocess_corpus(input_file: str, output_dir: str):
    """
    Convert a plain-text corpus (.dat or .txt) into JSON documents
    that Pyserini can index.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, "r") as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            doc = {
                "id": str(i),  # Each doc needs a unique ID
                "contents": line.strip(),
            }
            with open(os.path.join(output_dir, f"doc{i}.json"), "w") as out:
                json.dump(doc, out)


def build_index(input_dir: str, index_dir: str):
    """
    Build a BM25 index using Pyserini’s command-line interface.
    """
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]

    print("Building index (this may take a few minutes)...")
    subprocess.run(cmd, check=True)
    print(f"Index built successfully at {index_dir}")


# ---------------------------------------------------------------------
# 3. High-level API for index creation and retriever construction
# ---------------------------------------------------------------------
def create_index(cname: str) -> str:
    """
    Create an index for the given corpus name (cname).
    The raw data should be under data/{cname}/{cname}.dat
    """
    base_dir = f"data/{cname}"
    corpus_file = os.path.join(base_dir, f"{cname}.txt")
    processed_corpus_dir = f"processed_corpus/{cname}"
    index_dir = f"indexes/{cname}"

    # Step 1. Preprocess
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}")

    # Step 2. Build index
    build_index(processed_corpus_dir, index_dir)
    return index_dir


def create_retriever():
    """
    Create a LangChain retriever pipeline:
    BM25 (Pyserini) → optional contextual compression (LLM).
    """
    try:
        index_dir = create_index(config.PYSERINI_CNAME)

        # 1. Create the base BM25 retriever
        base_retriever = PyseriniBM25Retriever(
            index_dir=index_dir,
            k=config.RETRIEVER_K,
            k1=config.PYSERINI_K1,
            b=config.PYSERINI_B,
        )
        logger.info(f"BM25 retriever created on {index_dir}")
        '''
        # 2. (Optional) Add contextual compression with an LLM
        # This step uses the model defined in utils/config.py
        llm = config.get_llm()  # make sure Config implements get_llm()
        compressor = LLMChainExtractor.from_llm(llm)

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )'''

        logger.info("Full retriever created successfully.")
        return base_retriever

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise
