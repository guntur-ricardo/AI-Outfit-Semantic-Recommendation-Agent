import pickle
import numpy as np
import faiss
from data_processing.semantic_search import query_semantic_langchain

# --- Test raw FAISS index ---
print("\n[FAISS Test]")
# Load embeddings and FAISS index
with open('embeddings.pkl', 'rb') as f:
    index_data = pickle.load(f)
embeddings = index_data['embeddings'].astype(np.float32)
product_ids = index_data['product_ids']
index = faiss.read_index('faiss_fashion.index')

# Normalize and query first vector
query_vec = embeddings[0:1]
faiss.normalize_L2(query_vec)
sim_scores, neighbor_idxs = index.search(query_vec, 5)
print("Top FAISS matches (Normalized all vectors: Higher = better:")
for score, idx in zip(sim_scores[0], neighbor_idxs[0]):
    print(f"ID: {product_ids[idx]}, Score: {score:.4f}")

# --- Test LangChain semantic search ---
print("\n[LangChain Test]")
results = query_semantic_langchain("beach summer outfit", top_k=5)
print("Top LangChain matches (L2 Distance: Lower = better:")
for r in results:
    print(f"ID: {r['product_id']}, Score: {r['score']:.4f}")
