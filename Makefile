# Makefile - Simplified project setup targets

.PHONY: all preprocess embed index setup

# Default target: run all steps
default: setup

# 1. Fetch and prepare the sample dataset
preprocess:
	@echo "[1/3] Generating sample dataset..."
	pipenv run python scripts/generate_sample_dataset.py

# 2. Generate embeddings via OpenAI (auto-batched)
embed:
	@echo "[2/3] Building embeddings..."
	pipenv run python -m data_processing.index_embeddings

# 3. Build the FAISS index for local semantic search
index:
	@echo "[3/3] Building FAISS index..."
	pipenv run python data_processing/build_faiss_index.py

# Aggregate target: run all three in sequence
setup: preprocess embed index
	@echo "\n✅ All setup steps completed successfully! You can now run 'pipenv run uvicorn main:app --reload' to start the API server"

