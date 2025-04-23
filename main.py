import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from logging_config import setup_logging
from llm_processing.semantic_search import search_service
from llm_processing.recommendation_chain import recommendation_chain

# Initialize global logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Dressing")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 20


@app.on_event("startup")
def startup_event():
    try:
        search_service.initialize()
    except Exception as e:
        logger.error("Failed to initialize search service: %s", e)
        # Crash early if we can't serve searches
        raise


@app.post("/recommend")
def recommend(req: QueryRequest):
    try:
        recs = search_service.query(req.query, req.top_k)
        return {"recommendations": recs}
    except Exception as e:
        logger.error("Error in /recommend: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal search error")


@app.post("/recommendationChat")
def recommendationChat(req: QueryRequest):
    # 1. get the raw semantic hits
    sem_results = search_service.query(req.query, req.top_k)
    # 2. feed them + the original query into the LLM chain
    try:
        chat_response = recommendation_chain.run(req.query, sem_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM recommendation failed: {e}")
    # 3. return both for transparency
    return {
        "semantic_recommendations": sem_results,
        "chat_recommendation": chat_response
    }
