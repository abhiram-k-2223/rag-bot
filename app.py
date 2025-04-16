from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from rag_utils import RAGProcessor
import os

app = FastAPI(title="Scope Club RAG API", description="A RAG API for Scope Club Q&A using FAISS and MiniLM")

# Initialize RAG processor
rag_processor = None

class Query(BaseModel):
    text: str
    num_results: int = 3

class QAPair(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    results: List[QAPair]
    scores: List[float]

@app.on_event("startup")
async def startup_event():
    global rag_processor
    rag_processor = RAGProcessor()
    # Load the documetn i.e scope_final.txt
    if os.path.exists("scope_final.txt"):
        rag_processor.load_text("scope_final.txt")
    else:
        raise HTTPException(status_code=500, detail="scope.txt file not found")

@app.post("/query", response_model=Response)
async def query_rag(query: Query):
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")
    
    try:
        results, scores = rag_processor.query(query.text, query.num_results)
        return Response(
            results=[QAPair(question=r["question"], answer=r["answer"]) for r in results],
            scores=scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 