# Makefile - Simplified project setup targets

.PHONY: all preprocess embed index setup

# Default target: run all steps
default: setup

install:
	@echo "[1/4] Installing pip environment and modules"
	pipenv sync

# 2. Fetch and prepare the sample dataset
preprocess:
	@echo "[2/4] Generating sample dataset..."
	pipenv run python scripts/generate_sample_dataset.py

# 3. Generate embeddings via OpenAI (auto-batched)
embed:
	@echo "[3/4] Building embeddings..."
	pipenv run python -m data_processing.index_embeddings

# 4. Build the FAISS index for local semantic search
index:
	@echo "[4/4] Building FAISS index..."
	pipenv run python data_processing/build_faiss_index.py

# Aggregate target: run all three in sequence
setup: install preprocess embed index
	@echo "\n✅ All setup steps completed successfully! You can now run 'pipenv run start' to start the API server"

