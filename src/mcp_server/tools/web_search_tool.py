import os
import json
import httpx
from typing import Dict, Any
from ..cloudflare_client import cf_client

class FreeWebSearch:
    """
    무료 웹 검색 API 통합
    - Brave Search API 우선 사용
    - DuckDuckGo API fallback
    """
    
    def __init__(self):
        self.brave_api_key = os.getenv("BRAVE_API_KEY", "")
    
    async def _search_brave(self, query: str, count: int = 5) -> Dict[str, Any]:
        """Brave Search API 호출"""
        if not self.brave_api_key:
            return {"error": "Brave API key not configured"}
        
        headers = {"X-Subscription-Token": self.brave_api_key}
        params = {"q": query, "count": count}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """웹 검색 실행 (Brave → DuckDuckGo)"""
        # Brave Search 시도
        result = await self._search_brave(query, max_results)
        
        if "error" in result or "web" not in result:
            # DuckDuckGo fallback
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        "https://duckduckgo-api.vercel.app/api",
                        params={"q": query, "max_results": max_results}
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                return {"error": f"Both search APIs failed: {str(e)}"}
        
        # 결과 포맷팅
        formatted_results = []
        for item in result["web"]["results"][:max_results]:
            formatted_results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "source": "brave"
            })
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }

# MCP 도구용 인터페이스
web_searcher = FreeWebSearch()

async def web_search_tool(query: str, max_results: int = 5) -> str:
    """웹 검색 MCP 도구"""
    result = await web_searcher.search(query, max_results)
    return json.dumps(result, ensure_ascii=False)