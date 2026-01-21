from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from src.core.config import settings
from src.tools.kakao_map import KakaoMapTool
from src.tools.email_service import EmailService
import operator
import json

# --- 1. State 정의 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: str
    next_step: str
    context_data: dict  # 검색 결과나 추출된 정보를 담는 곳

# --- 2. LLM 설정 ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    api_key=settings.GROQ_API_KEY
)

# --- 3. 노드(Node) 함수 정의 ---

def router_node(state: AgentState):
    """사용자의 첫 발화를 분석하여 의도를 분류합니다."""
    last_msg = state['messages'][-1].content
    
    prompt = f"""
    Analyze the user's query and classify the intent into one of the following:
    - LOCATION: Asking for places, restaurants, or navigation.
    - INFO_SEARCH: Asking for university notices, scholarships, visa info (Need to search DB).
    - OCR_VISA: User provided an image or asks to analyze a document.
    - GENERAL: General greetings or irrelevant queries.
    
    Query: {last_msg}
    
    Return ONLY the intent keyword.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    intent = response.content.strip()
    return {"intent": intent}

def search_node(state: AgentState):
    """Qdrant Hybrid Search를 수행합니다."""
    query = state['messages'][-1].content
    
    # 실제로는 src.utils.searcher.KNUSearcher 호출
    # 여기서는 모의(Mock) 응답
    print(f"Searching for: {query}")
    retrieved_info = "[검색결과] 2026-1학기 수강신청 기간은 2월 10일부터입니다. (출처: 국제교류처)"
    
    return {"context_data": {"search_result": retrieved_info}}

def kakao_node(state: AgentState):
    """카카오맵 도구를 호출합니다."""
    query = state['messages'][-1].content
    tool = KakaoMapTool()
    # 키워드 추출 로직 필요
    result = tool.search_places(query) # 단순화
    return {"context_data": {"location_result": str(result)}}

def answer_node(state: AgentState):
    """최종 답변을 생성합니다."""
    intent = state['intent']
    context = state.get('context_data', {})
    
    if intent == "INFO_SEARCH":
        final_prompt = f"Based on this info: {context.get('search_result')}, answer the user: {state['messages'][-1].content}"
    elif intent == "LOCATION":
        final_prompt = f"Summarize these locations: {context.get('location_result')} for the user."
    else:
        final_prompt = state['messages'][-1].content

    response = llm.invoke([HumanMessage(content=final_prompt)])
    return {"messages": [response]}

# --- 4. 그래프(Graph) 구성 ---

workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("router", router_node)
workflow.add_node("search_db", search_node)
workflow.add_node("kakao_tool", kakao_node)
workflow.add_node("generate_answer", answer_node)

# 엔트리 포인트 설정
workflow.set_entry_point("router")

# 조건부 엣지 (Router의 판단에 따라 분기)
def route_decision(state):
    intent = state['intent']
    if intent == "LOCATION":
        return "kakao_tool"
    elif intent == "INFO_SEARCH":
        return "search_db"
    else:
        return "generate_answer" # GENERAL 등은 바로 답변

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "kakao_tool": "kakao_tool",
        "search_db": "search_db",
        "generate_answer": "generate_answer"
    }
)

# 도구 사용 후 답변 생성으로 이동
workflow.add_edge("kakao_tool", "generate_answer")
workflow.add_edge("search_db", "generate_answer")
workflow.add_edge("generate_answer", END)

# 컴파일
agent_app = workflow.compile()