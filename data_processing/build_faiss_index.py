import pickle
import numpy as np
import faiss

# 1. Load embeddings.pkl
with open('data/embeddings.pkl', 'rb') as f:
    index_data = pickle.load(f)
product_ids = index_data['product_ids']
embeddings = index_data['embeddings'].astype(np.float32)

# 2. Normalize embeddings for cosine similarity.
# AKA metric aligns with more human notions of closeness
# IE: cosine 1.0 = exact semantic match; 0.0 = no semantic overlap
# Other search strats: ANN (approximate nearest neighbor),
faiss.normalize_L2(embeddings)

# 3. Build FAISS index (inner product on normalized = cosine)
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
# 4. Map custom IDs, in this case product IDs so we can use them in the results
id_index = faiss.IndexIDMap(index)
id_index.add_with_ids(embeddings, np.arange(len(product_ids)))

# 5. Save FAISS index so we can skip embedding on every restart
faiss.write_index(id_index, 'data/faiss_fashion.index')
print(f"FAISS index built and saved. Total vectors: {id_index.ntotal}")
