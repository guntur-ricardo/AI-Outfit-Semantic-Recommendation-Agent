import pickle
import numpy as np
import faiss

# 1. Load embeddings.pkl
with open('embeddings.pkl', 'rb') as f:
    index_data = pickle.load(f)
product_ids = index_data['product_ids']
embeddings = index_data['embeddings'].astype(np.float32)

# 2. Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# 3. Build FAISS index (inner product on normalized = cosine)
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
# 4. Map custom IDs
id_index = faiss.IndexIDMap(index)
id_index.add_with_ids(embeddings, np.arange(len(product_ids)))

# 5. Save FAISS index
faiss.write_index(id_index, 'faiss_fashion.index')
print(f"FAISS index built and saved. Total vectors: {id_index.ntotal}")
