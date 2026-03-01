from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import os
import uvicorn

# DB Connection Pool & Checkpointer
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Agent Import
from src.agents.main_agent import KNUAgent
from src.core.config import settings

# Global Agent Instance
agent: Optional[KNUAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작/종료 시 DB 연결을 관리하는 수명 주기 함수
    """
    # 1. 환경 변수에서 DB_URI 가져오기 (Cloud Run 환경변수 또는 .env)
    # settings.DB_URI가 우선, 없으면 os.environ 조회
    db_uri = getattr(settings, "DB_URI", None) or os.getenv("DB_URI")
    
    if not db_uri:
        print("[Warning] DB_URI not found! Using MemorySaver (Data lost on restart).")
        # DB가 없으면 임시 메모리 저장소 사용 (로컬 테스트용)
        from langgraph.checkpoint.memory import MemorySaver
        global agent
        agent = KNUAgent(checkpointer=MemorySaver())
        yield
    else:
        print("[System] Connecting to Supabase (PostgreSQL)...")
        # 2. Supabase(PostgreSQL) 연결 풀 생성
        # max_size=20: 동시 접속 처리용 여유 공간
        async with AsyncConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            
            # 3. 체크포인트 테이블 자동 생성 (최초 1회 실행됨)
            await checkpointer.setup()
            
            # 4. 에이전트에 DB 연결 주입하여 초기화
            global agent
            agent = KNUAgent(checkpointer=checkpointer)
            
            print("[System] Agent initialized with Long-term Memory.")
            yield
            # 앱 종료 시 pool은 자동으로 닫힘

app = FastAPI(title="KNU International Student Agent", lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None 

class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool

@app.get("/")
def read_root():
    return {"status": "ok", "service": "KNU Agent Service running on Cloud Run"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    사용자 질문을 받아 에이전트 답변을 스트리밍으로 반환
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    # thread_id가 없으면 새로 생성 (새로운 대화)
    thread_id = request.thread_id or str(uuid.uuid4())
    
    async def event_generator():
        try:
            # 에이전트 처리 결과 스트리밍
            async for chunk in agent.process_query(request.query, thread_id):
                yield chunk
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        event_generator(), 
        media_type="text/plain",
        headers={"X-Thread-ID": thread_id} # 클라이언트가 이 ID를 저장해둬야 대화가 이어짐
    )

@app.post("/approve")
async def approve_endpoint(request: ApprovalRequest):
    """
    승인 대기 중인 작업(이메일 등)을 승인하거나 거절
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    async def event_generator():
        try:
            # 승인 결과에 따른 후속 작업 스트리밍
            async for chunk in agent.approve_tool(request.thread_id, request.approved):
                yield chunk
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )

if __name__ == "__main__":
    # 로컬 테스트 실행용
    uvicorn.run(app, host="0.0.0.0", port=8080)