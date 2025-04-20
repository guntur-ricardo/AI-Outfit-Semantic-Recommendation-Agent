import os
import pickle
import numpy as np
import openai
from dotenv import load_dotenv
from data_processing.preprocess import load_dataset, preprocess_text

# Load environment and API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Model for embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts via OpenAI v1 API.
    Returns a NumPy array of shape (len(texts), embedding_dim).
    """
    print("Generating Embeddings...")
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors)


def build_and_save_index(output_path: str = 'embeddings.pkl'):
    """
    Load dataset, preprocess descriptions, generate embeddings, and serialize index.
    """
    dataset = load_dataset()
    dataset['cleaned'] = dataset['description'].apply(preprocess_text)
    texts = dataset['cleaned'].tolist()
    embeddings = embed_texts(texts)

    index_data = {
        'product_ids': dataset['product_id'].tolist(),
        'embeddings': embeddings
    }
    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"Saved embeddings for {len(texts)} products → {output_path}")


# pipenv run python -m data_processing.index_embeddings
if __name__ == "__main__":
    build_and_save_index()
