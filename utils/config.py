import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file

        # Ollama settings
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        # Retriever settings
        self.RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "pyserini").lower()
        self.RETRIEVER_K = int(os.getenv("RETRIEVER_K", "7"))

        self.PYSERINI_CNAME = os.getenv("PYSERINI_CNAME", "combined_corpus")
        self.PYSERINI_K1 = float(os.getenv("PYSERINI_K1", "0.9"))
        self.PYSERINI_B = float(os.getenv("PYSERINI_B", "0.4"))

    def __repr__(self):
        return (f"Config("
                f"OLLAMA_MODEL={self.OLLAMA_MODEL}, "
                f"RETRIEVER_K={self.RETRIEVER_K}), "
                f"PYSERINI_CNAME={self.PYSERINI_CNAME}, "
                f"PYSERINI_K1={self.PYSERINI_K1}, "
                f"PYSERINI_B={self.PYSERINI_B}, ")

    def get_llm(self):
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        llm = ChatOllama(
            model=self.OLLAMA_MODEL,
        )
        return llm | StrOutputParser()
