import operator
import json
from typing import TypedDict, Annotated, List, Dict, Optional, AsyncGenerator, Literal, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
# MemorySaver 제거 -> 외부 주입 방식 사용

from src.core.config import settings


# --- LangChain 도구 래핑 ---
@tool
def search_knu_info(query: str, department: str = "공통") -> str:
    """
    경북대학교 공지사항, 장학금, 학사 정보, 비자 정보 등을 검색합니다.
    중요: 검색 정확도를 위해 'query' 파라미터는 반드시 '한국어(Korean)'로 번역하여 입력해야 합니다.
    """
    results = searcher_tool.search(query, target_dept=department)
    if not results:
        return "관련 정보를 찾을 수 없습니다."
    
    formatted = []
    for r in results:
        formatted.append(f"- 제목: {r['title']}\n  내용요약: {r['content'][:200]}...\n  링크: {r['url']}")
    return "\n".join(formatted)

@tool
def search_places_near_knu(query: str) -> str:
    """
    경북대학교 근처의 맛집, 편의점, 병원 등 장소를 찾거나 길찾기 정보를 제공합니다.
    중요: 'query'는 되도록 '한국어'로 입력하는 것이 정확도가 높습니다.
    """
    result = map_tool.search_near_knu(query)
    if not result.get("success"):
        return f"장소 검색 실패: {result.get('error')}"
    
    places = result.get("places", [])
    if not places:
        return "근처에 해당되는 장소가 없습니다."
        
    formatted = []
    for p in places[:5]:
        formatted.append(f"- {p['name']} ({p['category']}): {p['distance_text']} 거리, {p['walk_time']}\n  주소: {p['address']}")
    return "\n".join(formatted)

@tool
def read_document_text(image_path: str) -> str:
    """
    이미지 파일(신분증, 통장, 신청서 등)에서 글자를 읽어옵니다 (OCR).
    """
    result = ocr_tool.extract_text_from_image(image_path)
    if result.get('error'):
        return f"읽기 실패: {result['error']}"
    return f"문서 내용:\n{result['text']}"

@tool
def fill_application_form(template_type: str, data: Dict) -> str:
    """
    신청서 양식(Word/PDF)을 작성합니다.
    """
    result = form_tool.fill_form(template_type, data)
    if result['success']:
        return f"작성 완료! 파일 저장 위치: {result['output_path']}"
    else:
        return f"작성 실패: {result['warnings']}"

@tool
def send_email_draft(recipient: str, subject: str, body: str) -> str:
    """
    이메일을 전송합니다. 반드시 승인 절차를 거칩니다.
    """
    result = email_tool.send_email(recipient, subject, body)
    return f"전송 결과: {result['status']}"

TOOLS = [search_knu_info, search_places_near_knu, read_document_text, fill_application_form, send_email_draft]

# --- 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    approval_status: str  # approved, rejected, pending

# --- 에이전트 클래스 ---
class KNUVertexAgent:
    def __init__(self, checkpointer: Any):
        """
        Args:
            checkpointer: AsyncPostgresSaver 인스턴스를 주입받음
        """
        self.checkpointer = checkpointer
        
        # Vertex AI Gemini 설정 (Gemini 3.0 Flash)
        self.llm = ChatVertexAI(
            
            model_name="gemini-3.0-flash", 
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION,
            temperature=0, 
            convert_system_message_to_human=True 
        )
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        
        # 워크플로우 컴파일
        self.app = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(TOOLS))
        
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            self._route_tools,
            {
                "tools": "tools",
                "approval_required": END, # Human-in-the-loop 대기
                END: END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        # 주입받은 PostgresSaver 사용
        return workflow.compile(checkpointer=self.checkpointer)

    def _agent_node(self, state: AgentState):
        messages = state['messages']
        
        system_msg = """당신은 경북대학교(KNU) 외국인 유학생을 돕는 AI 에이전트입니다.
        
        [행동 수칙]
        1. **언어**: 사용자의 질문 언어로 답변하세요.
        2. **도구 호출**: 검색 도구의 'query' 인자는 반드시 **한국어**로 번역해서 넣으세요.
           - User: "Scholarship deadline?" -> Tool: search_knu_info(query="장학금 마감일")
        3. **판단**: 확실하지 않으면 추측하지 말고 검색하거나 문서를 읽으세요.
        """
        
        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_msg)] + messages
            
        # 거절된 상태라면 도구 호출 안 함
        if state.get("approval_status") == "rejected":
             return {
                "messages": [AIMessage(content="사용자가 작업을 취소했습니다. 다른 도움이 필요하신가요?")],
                "approval_status": "pending" # 상태 초기화
            }

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _route_tools(self, state: AgentState) -> Literal["tools", "approval_required", END]:
        last_message = state['messages'][-1]
        
        if not last_message.tool_calls:
            return END
            
        # 민감한 도구 체크
        SENSITIVE_TOOLS = ["send_email_draft", "fill_application_form"]
        for tool_call in last_message.tool_calls:
            if tool_call["name"] in SENSITIVE_TOOLS:
                if state.get("approval_status") != "approved":
                    return "approval_required"
        
        return "tools"

    async def process_query(self, query: str, thread_id: str) -> AsyncGenerator[Any, None]:
        config = {"configurable": {"thread_id": thread_id}}
        
        # 현재 상태 확인 (승인 대기 중이었는지)
        current_state = await self.app.aget_state(config)
        if current_state.next and not query: 
            # 쿼리가 비어있는데 상태가 남아있으면 이어서 진행 (승인 후 로직 등)
            inputs = None
        else:
            inputs = {"messages": [HumanMessage(content=query)]}

        async for event in self.app.astream_events(inputs, config, version="v1"):
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content
            
            elif kind == "on_tool_start":
                yield f"\n[System: {event['name']} 실행 중...]\n"
                
            elif kind == "on_tool_end":
                # 도구 결과는 JSON 형태로 보낼 수도 있음
                pass

        # 종료 후 상태 확인 (승인 대기 여부)
        snapshot = await self.app.aget_state(config)
        if snapshot.next:
            last_msg = snapshot.values["messages"][-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                 # 클라이언트가 인식할 수 있는 승인 요청 마커 전송
                yield "\n[APPROVAL_REQUIRED]" 

    async def approve_tool(self, conversation_id: str, approved: bool) -> AsyncGenerator[str, None]:
        """승인/거절 처리 후 워크플로우 재개 (스트리밍)"""
        config = {"configurable": {"thread_id": conversation_id}}
        
        status = "approved" if approved else "rejected"
        
        # 상태 업데이트
        await self.app.aupdate_state(config, {"approval_status": status})
        
        # 멈춘 지점부터 다시 실행 (None 입력으로 재개)
        async for event in self.app.astream_events(None, config, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content
            elif kind == "on_tool_start":
                yield f"\n[System: {event['name']} 실행 중...]\n"