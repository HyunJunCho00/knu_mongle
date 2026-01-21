from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.agents.main_agent import KNUAgent

app = FastAPI(title="KNU International Student Agent")
agent = KNUAgent()

class QueryRequest(BaseModel):
    query: str
    context: Optional[dict] = None

class QueryResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"message": "Welcome to KNU International Student Agent"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        response = agent.process_query(request.query, request.context)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
