import requests
from src.core.config import settings

class KakaoMapTool:
    def __init__(self):
        self.api_key = settings.KAKAO_API_KEY
        self.base_url = "https://dapi.kakao.com/v2/local/search/keyword.json"

    def search_places(self, query: str, x: str = None, y: str = None, radius: int = 2000):
        """
        Search for places using Kakao Local API.
        x: Longitude (optional, for location-based search)
        y: Latitude (optional, for location-based search)
        radius: Search radius in meters (default 2000m)
        """
        if not self.api_key:
            return {"error": "KAKAO_API_KEY is not set"}

        headers = {"Authorization": f"KakaoAK {self.api_key}"}
        params = {"query": query}
        
        if x and y:
            params.update({"x": x, "y": y, "radius": radius})

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_navigation_link(self, destination_name: str, destination_x: str, destination_y: str):
        """
        Generate a Kakao Map navigation URL.
        """
        # Kakao Map URL Scheme
        return f"https://map.kakao.com/link/to/{destination_name},{destination_y},{destination_x}"
