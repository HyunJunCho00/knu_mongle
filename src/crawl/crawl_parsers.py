import os
from urllib.parse import urljoin
from groq import Groq

try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except:
    groq_client = None

def analyze_image(image_src):
    if not groq_client: return ""
    
    if image_src.startswith("data:image") and len(image_src) < 1000:
        return ""

    try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and extract all text and key visual information in Korean. Be concise."},
                        {"type": "image_url", "image_url": {"url": image_src}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=800
        )
        return completion.choices[0].message.content
    except Exception:
        return ""

def parse_post_content(soup, url):
    if not soup: return "", [], []

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
            
            if len(content) > 5: break
    
    if not content:
        article = soup.select_one("article")
        if article:
            content = article.get_text("\n", strip=True)
            content_area = article

    targets = []
    if content_area: targets.append(content_area)
    targets.extend(soup.select("#bo_v_img, .view_image, .attached_image, .file_list"))

    processed_imgs = set()
    vlm_results = []

    for target in targets:
        for img in target.select("img"):
            src = img.get("src")
            if not src: continue
            
            if not src.startswith("data:"):
                src = urljoin(url, src)

            if src in processed_imgs: continue
            processed_imgs.add(src)

            description = analyze_image(src)
            if description:
                vlm_results.append(f"\n[이미지 설명: {description}]")

    if vlm_results:
        content += "\n" + "\n".join(vlm_results)

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

    return content, [], attachments