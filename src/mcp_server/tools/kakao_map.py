import os
import json
import httpx
from urllib.parse import quote
from ..cloudflare_client import cf_client

class KakaoTools:
    """
    카카오 API 통합
    - 장소 검색
    - 경로 안내  
    - 주소/좌표 변환
    """
    
    def __init__(self):
        self.api_key = os.getenv("KAKAO_REST_API_KEY")
        if not self.api_key:
            print("Warning: KAKAO_REST_API_KEY not set")
        
        self.base_headers = {"Authorization": f"KakaoAK {self.api_key}"}
        self.search_url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        self.geocode_url = "https://dapi.kakao.com/v2/local/search/address.json"
        self.reverse_geocode_url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
        self.route_url = "https://apis-navi.kakaomobility.com/v1/directions"
    
    async def _api_call(self, url: str, params: dict) -> Dict[str, Any]:
        """공통 API 호출 메서드"""
        if not self.api_key:
            return {"error": "Kakao API key not configured"}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    url,
                    headers=self.base_headers,
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def search_places(self, keyword: str, region: str = "") -> Dict[str, Any]:
        """장소 검색"""
        query = f"{region} {keyword}".strip()
        params = {"query": quote(query), "size": 5}
        
        result = await self._api_call(self.search_url, params)
        
        if "error" in result or not result.get("documents"):
            return {"status": "error", "message": result.get("error", "No results")}
        
        # Cloudflare LLM을 이용한 결과 분석
        instruction = (
            "Extract top 3 most relevant places. For each: name, category, address, "
            "phone, and a one-sentence summary in Korean why it's good for students."
        )
        
        processed = await cf_client.process_with_llm(
            raw_data=result,
            instruction=instruction,
            role="analyst"
        )
        
        return processed
    
    async def get_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """경로 안내"""
        if not origin or not destination:
            return {"error": "Both origin and destination required"}
        
        params = {"origin": origin, "destination": destination}
        result = await self._api_call(self.route_url, params)
        
        if "error" in result:
            return result
        
        # 경로 정보 추출
        try:
            route = result["routes"][0]
            summary = route["summary"]
            
            return {
                "status": "success",
                "distance_km": summary["distance"] / 1000,
                "duration_min": summary["duration"] / 60,
                "taxi_fare": summary.get("taxi_fare", 0),
                "traffic_condition": "heavy" if summary["duration"] > summary["distance"] * 0.1 else "normal"
            }
        except Exception as e:
            return {"error": f"Route parsing failed: {str(e)}"}
    
    async def geocode(self, address: str) -> Dict[str, Any]:
        """주소 → 좌표"""
        params = {"query": quote(address)}
        result = await self._api_call(self.geocode_url, params)
        
        if "error" in result or not result.get("documents"):
            return {"error": "Address not found"}
        
        doc = result["documents"][0]
        return {
            "status": "success",
            "address": doc["address_name"],
            "road_address": doc.get("road_address_name", ""),
            "x": doc["x"],
            "y": doc["y"]
        }
    
    async def reverse_geocode(self, x: str, y: str) -> Dict[str, Any]:
        """좌표 → 주소"""
        params = {"x": x, "y": y}
        result = await self._api_call(self.reverse_geocode_url, params)
        
        if "error" in result or not result.get("documents"):
            return {"error": "Coordinates not found"}
        
        doc = result["documents"][0]
        return {
            "status": "success",
            "address": doc["address"]["address_name"],
            "road_address": doc.get("road_address", {}).get("address_name", "")
        }

# 전역 인스턴스
kakao = KakaoTools()

# MCP 도구 인터페이스
async def search_places_tool(keyword: str, region: str = "") -> str:
    result = await kakao.search_places(keyword, region)
    return json.dumps(result, ensure_ascii=False)

async def get_directions_tool(start: str, end: str) -> str:
    result = await kakao.get_route(start, end)
    return json.dumps(result, ensure_ascii=False)

async def geocode_tool(address: str) -> str:
    result = await kakao.geocode(address)
    return json.dumps(result, ensure_ascii=False)

async def reverse_geocode_tool(x: str, y: str) -> str:
    result = await kakao.reverse_geocode(x, y)
    return json.dumps(result, ensure_ascii=False)