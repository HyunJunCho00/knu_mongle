import os
from urllib.parse import urljoin
from crawl.crawl_config import CONFIG

    
def parse_post_content(soup, url):
    """HTML 파싱: 본문 텍스트 + 이미지 URL + 첨부파일 URL 수집"""
    if not soup:
        return "", [], []

    for tag in soup(["script", "style", "iframe", "header", "footer", 
                     ".gnb", ".snb", ".lnb", "#header", "#footer", "#top", "#bottom", 
                     ".leftmenu", ".pagetitle", ".btn_area", ".prev_next", 
                     ".comment", "#comment", ".totalsearch"]):
        tag.decompose()

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
            
            if len(content) < 10:
                tables = element.select("table")
                for tbl in tables:
                    content += "\n" + tbl.get_text("\n", strip=True)
            
            if len(content) > 5:
                break
    
    if not content:
        article = soup.select_one("article")
        if article:
            content = article.get_text("\n", strip=True)
            content_area = article

    # 이미지 URL 수집 (다운로드는 안 함)
    targets = []
    if content_area:
        targets.append(content_area)
    targets.extend(soup.select("#bo_v_img, .view_image, .attached_image, .file_list"))

    processed_imgs = set()
    images_to_analyze = []

    for target in targets:
        for img in target.select("img"):
            src = img.get("src")
            if not src:
                continue
            
            is_base64 = src.startswith("data:")
            
            if not is_base64:
                src = urljoin(url, src)

            if src in processed_imgs:
                continue
            processed_imgs.add(src)

            # base64는 스킵, URL만 수집
            if not is_base64:
                images_to_analyze.append({
                    "url": src,
                    "alt": img.get("alt", "")
                })

    # 첨부파일 수집
    attachments = []
    file_selectors = [
        ".addfile a", ".file a", ".bo_v_file a", 
        "a[href*='download']", "a[href*='down']", 
        ".board_view_file a", "#bo_v_file a"
    ]
    
    seen_urls = set()
    for f_sel in file_selectors:
        for a in soup.select(f_sel):
            href = a.get('href')
            if not href or "javascript" in href or href.startswith("data:"):
                continue

            full_url = urljoin(url, href)
            if full_url in seen_urls:
                continue
            
            name = a.get_text(strip=True)
            if not name:
                continue
            
            ext = os.path.splitext(name)[1].lower()
            if ext in CONFIG.get("download_file_exts", []):
                attachments.append({"name": name, "url": full_url})
                seen_urls.add(full_url)

    return content, images_to_analyze, attachments