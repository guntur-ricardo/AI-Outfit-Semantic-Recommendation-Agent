from fastapi import FastAPI
from pydantic import BaseModel
from data_processing.semantic_search import query_semantic_langchain


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


app = FastAPI()


@app.post("/recommend")
def recommend(req: QueryRequest):
    results = query_semantic_langchain(req.query, req.top_k)
    return {"recommendations": results}
