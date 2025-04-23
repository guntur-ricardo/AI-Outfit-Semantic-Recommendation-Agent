"""
Integration Tests for Semantic Recommendation Pipeline

This script performs two sanity checks:

1. FAISS Test:
   - Loads the raw embeddings (embeddings.pkl) and the FAISS index (faiss_fashion.index).
   - Queries the index with the first embedding vector itself.
   - Verifies that the index returns the same vector (score=1.0) followed by its nearest neighbors.
   - Ensures the embedding-to-index alignment is correct and the inner-product (cosine) similarity pipeline works end-to-end.

2. LangChain Test:
   - Invokes the high-level `query_semantic_langchain` function with a sample human-like query.
   - Validates that the LangChain-based FAISS vectorstore returns a list of product recommendations.
   - Prints L2 distances (lower is more similar) to confirm the semantic search component is functioning.

These tests serve as a quick health check whenever:
- You regenerate embeddings or rebuild the FAISS index.
- You update the semantic search logic or pipeline.
- You want confidence that both the low-level and high-level retrieval components are aligned and operational.
"""
import pickle
import numpy as np
import faiss
from llm_processing.semantic_search import search_service
from llm_processing.recommendation_chain import recommendation_chain

# USAGE: pipenv run python -m scripts.test_search
# --- Test raw FAISS index ---
print("\n[FAISS Vector Store Test]")
# Load embeddings and FAISS index
with open('data/embeddings.pkl', 'rb') as f:
    index_data = pickle.load(f)
embeddings = index_data['embeddings'].astype(np.float32)
product_ids = index_data['product_ids']
index = faiss.read_index('data/faiss_fashion.index')

# Normalize and query first vector
query_vec = embeddings[0:1]
faiss.normalize_L2(query_vec)
sim_scores, neighbor_idxs = index.search(query_vec, 5)
print("Top FAISS matches (Normalized all vectors: Higher = better):")
for score, idx in zip(sim_scores[0], neighbor_idxs[0]):
    print(f"ID: {product_ids[idx]}, Score: {score:.4f}")

# --- Test LangChain semantic search ---
print("\n[LangChain Semantic Search Test]")

# Ensure the service is initialized
search_service.initialize()

# Run the query
results = search_service.query("beach summer outfit", top_k=5)

print("Top LangChain matches:")
for r in results:
    print(
        f"ID: {r['product_id']}, "
        f"Title: {r['title']}, "
        f"Score: {r['score']:.4f}, "
        f"Avg Rating: {r['average_rating']}"
    )

# Scores BEFORE enriched dataset (only description)
# [FAISS Test]
# Top FAISS matches (Normalized all vectors: Higher = better:
# ID: B08BHN9PK5, Score: 1.0000
# ID: B07VWJM737, Score: 0.8643
# ID: B01MS8TRZY, Score: 0.8563
# ID: B08B9QPRBG, Score: 0.8547
# ID: B01M1BFCCP, Score: 0.8541

# --------------------------------------
# Scores with enriched dataset (description, categories, average rating)
# [FAISS Test]
# Top FAISS matches (Normalized all vectors: Higher = better):
# ID: B08BHN9PK5, Score: 1.0000
# ID: B07VWJM737, Score: 0.8902
# ID: B08B9QPRBG, Score: 0.8754
# ID: B01MS8TRZY, Score: 0.8736
# ID: B01M1BFCCP, Score: 0.8724

# Fake sample products
print("LLM Chain Invoked Test:")
hits = [
  {"title": "Blue Beach Shirt", "description": "Lightweight linen...", "score": 0.85, "average_rating": "5"},
  {"title": "Sandals", "description": "Comfort foam...", "score": 0.82, "average_rating": "4"},
  {"title": "Green beach bag", "description": "Carry all your sand castle tools in this bag", "score": 0.62, "average_rating": "3"},
]
print(recommendation_chain.run("beach summer outfit", hits))
