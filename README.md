# README.md

## Project Overview
This repository implements a **semantic recommendation microservice** for an e‑commerce fashion product line. Rather than simple keyword search, it empowers users to submit *natural‑language* queries (e.g., “I need an outfit to go to the beach this summer”) and receive semantically relevant product suggestions.

**Core Components**:
1. **Data Sampling & Preprocessing** (`scripts/generate_sample_dataset.py`)
2. **Embedding Generation** (`data_processing/index_embeddings.py`) using OpenAI embeddings (auto‑batched via LangChain).
3. **Vector Indexing** (`data_processing/build_faiss_index.py`) in FAISS for similarity search.
4. **Semantic API** (`main.py`) powered by FastAPI exposing a `/recommend` endpoint.

## Getting Started
### Environment Variables!!!
1. Copy everything from .env.example to a .env file in the project root.
2. Insert your openAI API key or use my candidate one I left in the example

Ensure you have `pipenv` installed, then at the project root:
I'm on version 2024.4.0 but there shouldn't be any issues if you have a slightly different version:
```bash
# Install pipenv if you don't have it
pip install pipenv --user
```
```bash
# Install dependencies and set up everything:
pipenv run setup
```

This runs:
1. `pipenv run python scripts/generate_sample_dataset.py` Streams data from huggingface  
2. `pipenv run python -m data_processing.index_embeddings` Loads, generates embeddings and stores them for easy local dev 
3. `pipenv run python data_processing/build_faiss_index.py` 

After `make setup`, you have:
- `data/amazon_fashion_sample.csv`
- `embeddings.pkl`
- `faiss_fashion.index`

## Running Tests
I include a very simple integration test `scripts/test_search.py` that:

1. Validates the **raw FAISS** pipeline by querying the first embedding vector against `faiss_fashion.index`, ensuring alignment and similarity logic.  
2. Validates the **LangChain** semantic search path by submitting a sample human‑language query and inspecting the returned recommendations.

Run the tests via:

```bash
pipenv run test
```

## Running API Server!
```bash
pipenv run start
```

## Hitting the Server
Via Postman
```bash
URL: http://127.0.0.1:8000/recommendationChat
With raw json body:
{
    "query": "what should i wear to the dog park as a dude",
    "include_products": false,
    "top_k": 10
}
```

Via terminal CURL
```bash
curl --location 'http://127.0.0.1:8000/recommendationChat' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Looking for outfits to wear in space as a woman astronaut",
    "include_products": false,
    "top_k": 10
}'
```

## Design Decisions
- **Streaming vs. On‑Disk Storage**:
  -  For this prototype, I opted to keep everything **in memory** and leverage the Hugging Face streaming API to pull only the subset of data we need on demand. This approach keeps the code simple, minimizes external dependencies. Faster local dev!
  - If we scale beyond 100K–200K vectors or need persistent indexes, we could switch to an **on‑disk** vector store using Chroma or Qdrant
  - We can also implement distributed sharding and use FAISS or manually query each shard, then combine the results.
  Sharding can be implemented with or without the disk implementation.

- **Enriched Text Composition**:
  - Originally I was only loading in the title of the product as they seemed to be more descriptive at a glance. By querying my endpoint with various requests. 
  - I was seeing relevant products returned. I inspected the rest of the columns and decided to include description, categories and average rating. 
  - I believe description and categories have key words that may assist the semantic search in the vector space.
  - I loaded in average rating and given enough time, I believe we could add an additional weighing layer that impacts the overall ranking score depending on rating.

## Next Steps
- Integrate user‑facing front‑end (React + MUI) to consume `/recommend`.  
- Experiment with **sharded** or **disk‑backed** FAISS indexes for larger datasets.  
- Enrich search by incorporating user reviews, ratings, and metadata into re‑ranking logic.

---

