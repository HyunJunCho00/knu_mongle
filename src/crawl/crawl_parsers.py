import base64
import os
import uuid
from urllib.parse import urljoin
from knu_chatbot.crawling.config import CONFIG # 설정값 불러오기

def save_base64_image(base64_str):
    """
    Base64 문자열을 이미지 파일로 저장하고 상대 경로를 반환합니다.
    """
    try:
        if ";base64," not in base64_str:
            return None
        
        header, data = base64_str.split(";base64,")
        
        file_ext = "png"
        if "image/" in header:
            file_ext = header.split("image/")[1]
            if file_ext == "jpeg": file_ext = "jpg"
        
        filename = f"{uuid.uuid4()}.{file_ext}"
        save_path = os.path.join(CONFIG["image_dir"], filename)
        
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(data))
            
        return f"./data/images/{filename}"
        
    except Exception as e:
        print(f"[Image Error] Base64 decoding failed: {e}")
        return None

def parse_post_content(soup, url):
    if not soup: return "", [], []

    # 1. 노이즈 제거
    for tag in soup(["script", "style", "iframe", "header", "footer", 
                     ".gnb", ".snb", ".lnb", "#header", "#footer", "#top", "#bottom", 
                     ".leftmenu", ".pagetitle", ".btn_area", ".prev_next", 
                     ".comment", "#comment", ".totalsearch"]):
        tag.decompose()

    # 2. 본문 영역 찾기
    candidates = [
        "td.contentview", ".board_view", "#bo_v_con", ".board_view_con", 
        "div.cont", ".view_content", "div[id^='bo_v_con']"
    ]

    content_area = None
    content = ""

    for selector in candidates:
        element = soup.select_one(selector)
        if element:
            content_area = element
            for junk in element.select(".addfile, .file_list, .sns_area, .cmt_btn"):
                junk.decompose()
            
            content = element.get_text("\n", strip=True)
            
            # [보강] 표 내용 추가 추출
            if len(content) < 10:
                tables = element.select("table")
                for tbl in tables:
                    content += "\n" + tbl.get_text("\n", strip=True)
            
            if len(content) > 5: break
    
    if not content:
        article = soup.select_one("article")
        if article:
            content = article.get_text("\n", strip=True)
            content_area = article

    # 3. 이미지 추출 (Base64 -> 파일 저장 로직 적용)
    images = []
    
    # 본문 + 첨부 영역 통합 검색
    targets = []
    if content_area: targets.append(content_area)
    targets.extend(soup.select("#bo_v_img, .view_image, .attached_image, .file_list"))

    seen_src = set()
    for target in targets:
        for img in target.select("img"):
            src = img.get("src")
            if not src: continue
            
            final_url = None
            
            # [A] Base64 이미지인 경우 -> 파일로 저장
            if src.startswith("data:image"):
                saved_path = save_base64_image(src)
                if saved_path:
                    final_url = saved_path
            
            # [B] 일반 URL 이미지인 경우
            else:
                final_url = urljoin(url, src)

            if final_url and final_url not in seen_src:
                images.append(final_url)
                seen_src.add(final_url)

    # 4. 첨부파일 추출
    attachments = []
    file_selectors = [".addfile a", ".file a", ".bo_v_file a", "a[href*='download']", "a[href*='down']", ".board_view_file a"]
    
    seen_urls = set()
    for f_sel in file_selectors:
        for a in soup.select(f_sel):
            href = a.get('href')
            if not href or "javascript" in href: continue
            
            full_url = urljoin(url, href)
            if full_url in seen_urls: continue
            
            name = a.get_text(strip=True)
            if not name: continue
            
            attachments.append({"name": name, "url": full_url})
            seen_urls.add(full_url)

    return content, images, attachments