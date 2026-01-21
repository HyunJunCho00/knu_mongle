from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from src.core.config import settings
from src.tools.kakao_map import KakaoMapTool
from src.tools.email_service import EmailService
from src.tools.retriever import KNUSearcher
from src.tools.ocr import OCRTool
from src.tools.form_filler import FormFillerTool
import operator
import json

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: str
    tool_calls: List[dict]
    tool_results: List[dict]
    final_answer: str

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=settings.GROQ_API_KEY
)

@tool
def search_database(query: str, dept: str = None) -> str:
    """
    Search KNU database for academic information, notices, scholarships, visa info.
    Args:
        query: User's search query
        dept: Optional department filter
    Returns:
        Search results as formatted string
    """
    searcher = KNUSearcher()
    results = searcher.search(query, target_dept=dept, final_k=5)
    
    if not results:
        return "No results found"
    
    formatted = []
    for idx, r in enumerate(results, 1):
        formatted.append(f"[{idx}] {r['title']}\n{r['content'][:200]}...\nURL: {r['url']}\nDept: {r['dept']}\n")
    
    return "\n".join(formatted)

@tool
def search_location(query: str, latitude: str = None, longitude: str = None) -> str:
    """
    Search for places, restaurants, or facilities using Kakao Map API.
    Args:
        query: Place name or category to search
        latitude: Optional user location latitude
        longitude: Optional user location longitude
    Returns:
        List of places with navigation links
    """
    kakao = KakaoMapTool()
    results = kakao.search_places(query, x=longitude, y=latitude)
    
    if "error" in results:
        return f"Error: {results['error']}"
    
    documents = results.get("documents", [])
    if not documents:
        return "No places found"
    
    formatted = []
    for idx, place in enumerate(documents[:5], 1):
        nav_link = kakao.get_navigation_link(
            place['place_name'],
            place['x'],
            place['y']
        )
        formatted.append(
            f"[{idx}] {place['place_name']}\n"
            f"Address: {place.get('road_address_name', place.get('address_name'))}\n"
            f"Phone: {place.get('phone', 'N/A')}\n"
            f"Navigation: {nav_link}\n"
        )
    
    return "\n".join(formatted)

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR (for visa documents, forms, etc.)
    Args:
        image_path: Path to the image file
    Returns:
        Extracted text
    """
    ocr = OCRTool()
    text = ocr.extract_text_from_image(image_path)
    return text

@tool
def parse_visa_document(image_path: str) -> str:
    """
    Parse visa-related documents and extract structured information.
    Args:
        image_path: Path to visa document image
    Returns:
        Structured visa information
    """
    ocr = OCRTool()
    text = ocr.extract_text_from_image(image_path)
    info = ocr.parse_visa_info(text)
    return json.dumps(info, ensure_ascii=False, indent=2)

@tool
def fill_form_template(template_type: str, user_data: dict) -> str:
    """
    Fill out application forms (HWP, Word, PDF) with user data.
    Args:
        template_type: Type of form (scholarship, visa, enrollment, etc.)
        user_data: Dictionary containing user information
    Returns:
        Path to filled form file
    """
    filler = FormFillerTool()
    template_path = f"templates/{template_type}.docx"
    output_path = f"output/{template_type}_filled.docx"
    
    result = filler.fill_form(template_path, user_data, output_path)
    return f"Form filled successfully: {result}"

@tool
def draft_email(recipient: str, subject: str, body: str) -> str:
    """
    Draft an email for the user to review before sending.
    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body content
    Returns:
        Email draft information
    """
    email_service = EmailService()
    draft = email_service.draft_email(recipient, subject, body)
    return json.dumps(draft, ensure_ascii=False, indent=2)

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Send an email on behalf of the user (requires confirmation).
    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body content
    Returns:
        Send status
    """
    email_service = EmailService()
    result = email_service.send_email(recipient, subject, body)
    return json.dumps(result, ensure_ascii=False)

TOOLS = [
    search_database,
    search_location,
    extract_text_from_image,
    parse_visa_document,
    fill_form_template,
    draft_email,
    send_email
]

tools_by_name = {tool.name: tool for tool in TOOLS}

def router_node(state: AgentState):
    """Analyze user intent and determine required tools"""
    last_msg = state['messages'][-1].content
    
    system_prompt = f"""You are an intent classifier for a university assistant system.
Available tools:
- search_database: Academic info, notices, scholarships, visa information
- search_location: Places, restaurants, facilities (Kakao Map)
- extract_text_from_image: OCR for any image
- parse_visa_document: Structured visa document parsing
- fill_form_template: Fill application forms
- draft_email: Create email drafts
- send_email: Send emails

Analyze the user query and respond with ONLY a JSON object:
{{
  "intent": "INFO_SEARCH|LOCATION|OCR|VISA|FORM|EMAIL|GENERAL",
  "required_tools": ["tool_name1", "tool_name2"],
  "parameters": {{"param_key": "param_value"}}
}}

User query: {last_msg}"""
    
    response = llm.invoke([SystemMessage(content=system_prompt)])
    
    try:
        result = json.loads(response.content)
        return {
            "intent": result.get("intent", "GENERAL"),
            "tool_calls": [{
                "tool": t,
                "params": result.get("parameters", {})
            } for t in result.get("required_tools", [])]
        }
    except:
        return {"intent": "GENERAL", "tool_calls": []}

def tool_executor_node(state: AgentState):
    """Execute all required tools"""
    results = []
    
    for tool_call in state.get("tool_calls", []):
        tool_name = tool_call["tool"]
        params = tool_call["params"]
        
        if tool_name in tools_by_name:
            try:
                tool_func = tools_by_name[tool_name]
                result = tool_func.invoke(params)
                results.append({
                    "tool": tool_name,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "error": str(e)
                })
    
    return {"tool_results": results}

def answer_generator_node(state: AgentState):
    """Generate final answer using tool results"""
    tool_results = state.get("tool_results", [])
    user_query = state['messages'][-1].content
    
    if not tool_results:
        prompt = f"User query: {user_query}\n\nProvide a helpful response."
    else:
        results_text = "\n\n".join([
            f"Tool: {r['tool']}\nResult: {r.get('result', r.get('error'))}"
            for r in tool_results
        ])
        prompt = f"""User query: {user_query}

Tool Results:
{results_text}

Based on the above information, provide a comprehensive and helpful answer to the user."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=response.content)],
        "final_answer": response.content
    }

def route_after_intent(state: AgentState) -> Literal["execute_tools", "generate_answer"]:
    """Decide whether to execute tools or answer directly"""
    if state.get("tool_calls"):
        return "execute_tools"
    return "generate_answer"

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("execute_tools", tool_executor_node)
workflow.add_node("generate_answer", answer_generator_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_after_intent,
    {
        "execute_tools": "execute_tools",
        "generate_answer": "generate_answer"
    }
)

workflow.add_edge("execute_tools", "generate_answer")
workflow.add_edge("generate_answer", END)

agent_app = workflow.compile()

class KNUAgent:
    def __init__(self):
        self.app = agent_app
    
    def process_query(self, query: str, context: dict = None):
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "intent": "",
            "tool_calls": [],
            "tool_results": [],
            "final_answer": ""
        }
        
        result = self.app.invoke(initial_state)
        return result.get("final_answer", "Unable to process query")