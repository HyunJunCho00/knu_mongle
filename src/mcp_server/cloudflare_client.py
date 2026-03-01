import os
import json
import httpx
from typing import Any, Dict, Union

class CloudflareAIClient:
    """
    Cloudflare Workers AI API 클라이언트
    - 모델 선택 및 API 호출 관리
    - 응답 파싱 및 에러 처리
    """
    
    MODEL_MAPPING = {
        "analyst": "@cf/meta/llama-3.3-8b-instruct",
        "summarizer": "@cf/qwen/qwen-1.5-7b-chat-awq",
        "lightweight": "@cf/google/gemma-2b-it"
    }
    
    def __init__(self):
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def process_with_llm(self, raw_data: Union[Dict, str], instruction: str, role: str = "analyst") -> Dict[str, Any]:
        """
        Cloudflare LLM에 작업 요청 및 결과 반환
        
        Args:
            raw_ 처리할 원본 데이터
            instruction: LLM에게 줄 작업 지시사항
            role: 모델 역할 ("analyst", "summarizer", "lightweight")
        
        Returns:
            Dict: 파싱된 JSON 결과 또는 에러 정보
        """
        if not self.account_id or not self.api_token:
            return {"error": "Cloudflare credentials not configured"}
        
        model = self.MODEL_MAPPING.get(role, self.MODEL_MAPPING["analyst"])
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model}"
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # 데이터 크기 제한 (비용 최적화)
        data_str = json.dumps(raw_data, ensure_ascii=False)
        if len(data_str) > 3000:
            data_str = data_str[:3000] + "... (truncated)"
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data processing assistant. Return ONLY valid JSON without markdown."
                },
                {
                    "role": "user", 
                    "content": f"Data: {data_str}\n\nTask: {instruction}\n\nOutput JSON:"
                }
            ]
        }
        
        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if not result.get("success"):
                return {"error": "AI processing failed", "details": result}
            
            ai_text = result["result"]["response"]
            # 마크다운 제거 및 JSON 파싱
            clean_text = ai_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
            
        except Exception as e:
            return {"error": str(e), "raw_response": result.get("result", {}).get("response", "")}

# 싱글톤 인스턴스
cf_client = CloudflareAIClient()