import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from data_processing.preprocess import load_dataset, preprocess_text

# Load environment & API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize embeddings function
embeddings_fn = OpenAIEmbeddings(openai_api_key=api_key)

# Load and preprocess descriptions
df = load_dataset()
df['cleaned'] = df['description'].apply(preprocess_text)
texts = df['cleaned'].tolist()
product_ids = df['product_id'].tolist()

# Build metadata per item with human‑readable fields
metadatas = [
    {
        'product_id': pid,
        'description': desc
    }
    for pid, desc in zip(df['product_id'], df['description'])
]

# Build a LangChain FAISS vectorstore from raw texts and metadata
print("Building LangChain FAISS vectors...")
vectorstore = FAISS.from_texts(
    texts,
    embedding=embeddings_fn,
    metadatas=metadatas
)

# Save index locally
print("Saving FAISS vectors locally...")
vectorstore.save_local('langchain_faiss')
print("Save Completed...")


def query_semantic_langchain(query: str, top_k: int = 5):
    """
    Perform semantic search via LangChain FAISS vectorstore.
    Returns list of dicts: {'product_id': str, 'description': str, 'score': float}.
    """
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    return [
        {
            'product_id': doc.metadata['product_id'],
            'description': doc.metadata['description'],
            'score': float(score)
        }
        for doc, score in docs_and_scores
    ]
