import json
from mcp.server.fastmcp import FastMCP
from tools.retriever import campus_search_tool
from tools.web_search_tool import web_search_tool
from tools.kakao_map import (
    search_places_tool,
    get_directions_tool,
    geocode_tool,
    reverse_geocode_tool
)


# MCP 서버 초기화
mcp = FastMCP("Campus_Support_MCP")

# 도구 등록
@mcp.tool()
async def search_documents(query: str, department: str = None, limit: int = 5) -> str:
    """캠퍼스 문서 검색 (학칙, 공지사항 등)"""
    return await campus_search_tool(query, department, limit)

@mcp.tool()
async def search_internet(query: str, max_results: int = 5) -> str:
    """일반 웹 검색 (뉴스, 정보 등)"""
    return await web_search_tool(query, max_results)

@mcp.tool()
async def search_places(keyword: str, region: str = "") -> str:
    """주변 장소 검색 (맛집, 편의시설 등)"""
    return await search_places_tool(keyword, region)

@mcp.tool()
async def get_directions(start: str, end: str) -> str:
    """경로 안내 (출발지 → 목적지)"""
    return await get_directions_tool(start, end)

@mcp.tool()
async def convert_address(address: str) -> str:
    """주소 → 좌표 변환"""
    return await geocode_tool(address)

@mcp.tool()
async def convert_coordinates(x: str, y: str) -> str:
    """좌표 → 주소 변환"""
    return await reverse_geocode_tool(x, y)

if __name__ == "__main__":

    mcp.run(port=8765, host="0.0.0.0")