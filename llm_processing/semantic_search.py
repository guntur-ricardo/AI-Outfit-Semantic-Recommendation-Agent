import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from data_processing.preprocess import load_dataset, preprocess_text

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
        load_dotenv()

    def build(self):
        """
        Build the FAISS index from scratch, writing to disk.
        """
        logger.info("Building new FAISS vectorstore at '%s'", self.index_dir)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Load and preprocess dataset
        df = load_dataset()
        df['cleaned'] = df['description'].apply(preprocess_text)
        texts = df['cleaned'].tolist()

        # Build metadata dicts including average_rating
        metadatas = []
        for _, row in df.iterrows():
            metadatas.append({
                'product_id': row['product_id'],
                'title': row['title'],
                'description': row['description'],
                'average_rating': row.get('average_rating', None)
            })

        # Create and save vectorstore
        vectorstore = FAISS.from_texts(
            texts,
            embedding=self._embeddings,
            metadatas=metadatas
        )
        vectorstore.save_local(self.index_dir)
        logger.info("FAISS vectorstore built and saved.")

    def initialize(self):
        """
        Ensure the vectorstore exists on disk and load it for querying.
        """
        # Build if missing
        if not os.path.isdir(self.index_dir):
            self.build()
        logger.info("Loading FAISS vectorstore from '%s'", self.index_dir)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self._vectorstore = FAISS.load_local(
            self.index_dir,
            self._embeddings,
            allow_dangerous_deserialization=True
        )
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
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            logger.info(md)
            results.append({
                "product_id": md.get("product_id"),
                "title": md.get("title"),
                "description": md.get("description"),
                "score": float(score),
                'average_rating': md.get('average_rating')
            })
        logger.debug("Semantic search returned %d results", len(results))
        return results


# Single, shared instance for the running app
search_service = SemanticSearchService()
