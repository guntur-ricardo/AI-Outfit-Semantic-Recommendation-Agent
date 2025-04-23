import os
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    Wraps a LangChain FAISS vectorstore for semantic search.
    Initialization is explicit so we can control when the index is loaded.
    """

    def __init__(self, index_dir: str = "langchain_faiss"):
        self.index_dir = index_dir
        self._vectorstore = None
        self._embeddings = None

    def initialize(self):
        """
        Load the embeddings model and the FAISS index from disk.
        Call this once during application startup.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        logger.info("Initializing OpenAI embeddings model")
        self._embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        logger.info("Loading FAISS vectorstore from '%s'", self.index_dir)
        self._vectorstore = FAISS.load_local(self.index_dir, self._embeddings, allow_dangerous_deserialization=True)
        logger.info("Vectorstore loaded successfully")

    def query(self, query_text: str, top_k: int = 5):
        """
        Run a semantic search against the loaded FAISS index.
        Returns a list of dicts with product_id, title, description, and score.
        """
        if self._vectorstore is None:
            raise RuntimeError("Vectorstore not initialized; call initialize() first")

        logger.debug("Running semantic search for query=%r, top_k=%d", query_text, top_k)
        docs_and_scores = self._vectorstore.similarity_search_with_score(
            query_text, k=top_k
        )
        results = []
        for doc, score in docs_and_scores:
            md = doc.metadata
            results.append({
                "product_id": md.get("product_id"),
                "title": md.get("title"),
                "description": md.get("description"),
                "score": float(score),
            })
        logger.debug("Semantic search returned %d results", len(results))
        return results


# Single, shared instance for the running app
search_service = SemanticSearchService()

