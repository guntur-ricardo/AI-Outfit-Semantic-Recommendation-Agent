import os
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from data_processing.preprocess import load_dataset, preprocess_text
import logging
from logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Initialize LangChain embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using LangChain's embed_documents,
    which handles batching internally. Includes debug logs.
    Returns a NumPy array of shape (len(texts), embedding_dim).
    """
    logger.debug("embed_texts: starting with %d texts", len(texts))

    # Preprocess and coerce texts
    cleaned = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            logger.debug("embed_texts: non-string at index %d: %r", i, t)
            t = str(t)
        ct = preprocess_text(t)
        cleaned.append(ct)
        if i < 3:
            logger.debug("embed_texts: cleaned[%d] = %r", i, ct)

    # Generate embeddings
    logger.debug("embed_texts: calling embed_documents on cleaned texts")
    embeddings = embeddings_model.embed_documents(cleaned)
    logger.debug("embed_texts: received %d embeddings", len(embeddings))

    if embeddings:
        logger.debug("embed_texts: each embedding dimension = %d", len(embeddings[0]))
    else:
        logger.warning("embed_texts: no embeddings returned")

    arr = np.array(embeddings, dtype=np.float32)
    logger.debug("embed_texts: returning array of shape %s", arr.shape)
    return arr


def build_and_save_index(output_path: str = 'data/embeddings.pkl'):
    """
    Loads dataset, combines rich text fields, preprocesses, generates embeddings, and serializes index.
    """
    logger.info("build_and_save_index: loading dataset")
    df = load_dataset()

    # Combine title, description, categories, and details into one text field
    def make_combined(row):
        title = row.get('title') or ''
        description = row.get('description') or ''
        parts = [str(title), str(description)]
        cats = row.get('categories') or ''
        if cats:
            parts.append(f"Categories: {cats}")
        dets = row.get('details') or ''
        if dets:
            parts.append(f"Details: {dets}")
        return " ".join(parts)

    # Apply combination and preprocess
    df['combined'] = df.apply(make_combined, axis=1)
    df['cleaned'] = df['combined'].apply(preprocess_text)
    texts = df['cleaned'].tolist()

    logger.info("build_and_save_index: generating embeddings for %d texts", len(texts))
    vectors = embed_texts(texts)

    index_data = {
        'product_ids': df['product_id'].tolist(),
        'embeddings': vectors
    }
    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f)
    logger.info("build_and_save_index: saved embeddings to %s", output_path)


# pipenv run python -m data_processing.index_embeddings
if __name__ == '__main__':
    build_and_save_index()
