import json
import requests
import time
import urllib3
import os
import re
import threading
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

import datetime

# 모듈 불러오기
from crawl_config import CONFIG
from crawl_parsers import parse_post_content
from crawl_image import sanitize_filename

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 전역 락
file_lock = threading.Lock()
vlm_lock = threading.Lock()

class BaseCrawler:
    def __init__(self, target):
        self.dept = target['dept']
        self.detail = target['detail']
        self.base_url = target['url']
        
        safe_dept_name = sanitize_filename(self.dept)
        self.file_path = os.path.join(CONFIG["data_dir"], f"{safe_dept_name}.jsonl")
        
        self.last_crawled_date = CONFIG["cutoff_date"]
        
        self.session = requests.Session()
        self.session.headers.update(CONFIG["headers"])
        
        self.collected_links = set()
        if os.path.exists(self.file_path):
            with file_lock:
                try:
                    with open(self.file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if 'url' in data:
                                    self.collected_links.add(data['url'])
                            except: continue
                except Exception as e:
                    pass
    
    def fetch_page(self, url, referer=None):
        if referer: self.session.headers.update({"Referer": referer})
        try:
            time.sleep(0.5)
            resp = self.session.get(url, verify=False, timeout=30)
            if resp.encoding == 'ISO-8859-1': resp.encoding = resp.apparent_encoding
            return BeautifulSoup(resp.text, 'html.parser')
        except Exception as e:
            print(f"[Error] {self.dept} Fetch Fail: {e}")
            return None
    
    def save_post(self, post_data):
        if post_data['url'] in self.collected_links: return
        
        with file_lock:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(post_data, ensure_ascii=False) + "\n")
                self.collected_links.add(post_data['url'])
            except Exception as e:
                print(f"[Save Error] {self.dept}: {e}")
    
    def process_detail_page(self, title, date, link, referer=None, force_save=False):
        # 이미 수집한 링크는 건너뜀 (중복 방지는 유지)
        if link in self.collected_links: return True
        
        soup = self.fetch_page(link, referer)
        if not soup: return True

        # 본문에서 진짜 날짜 확인
        page_text = soup.get_text()
        date_match = re.search(r"20\d{2}[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])", page_text)
        
        if date_match:
            real_date = date_match.group(0).replace('.', '-').replace('/', '-')
            if real_date[:4] != date[:4]:
                date = real_date

        # [수정] 강제 저장 모드(force_save)가 아닐 때만 날짜를 체크하여 스킵
        if not force_save:
            if date < self.last_crawled_date:
                print(f"   ㄴ [Stop] 기준 날짜({self.last_crawled_date}) 미달: {date}")
                return False
        else:
            # 강제 저장 모드일 경우 로그만 찍고 통과
            if date < self.last_crawled_date:
                print(f"   ㄴ [Force] 날짜 미달({date})이지만 2페이지 이내라 강제 수집")

        content, images_to_save, atts_to_save = parse_post_content(soup, link)
        
        safe_dept = sanitize_filename(self.dept)
        base_dir = Path(CONFIG["attachments_dir"]) / safe_dept / date
        
        processed_images = []
        img_dir = base_dir / "images"
        
        for img_info in images_to_save:
            url = img_info['url']
            name = sanitize_filename(os.path.basename(url.split('?')[0]) or "image.jpg")
            save_path = img_dir / name
            
            meta = _download_file(self.session, url, save_path, referer=link)
            
            processed_images.append({
                "url": url,
                "saved_path": str(save_path),
                "alt": img_info['alt'],
                "description": img_info['description'], 
                **meta
            })

        processed_atts = []
        extracted_texts = [] 
        att_dir = base_dir / "files"
        
        for att in atts_to_save:
            url = att['url']
            name = sanitize_filename(att['name'])
            ext = os.path.splitext(name)[1]
            save_path = att_dir / name
            
            meta = _download_file(self.session, url, save_path, referer=link)
            
            att_data = {
                "name": name,
                "url": url,
                "saved_path": str(save_path),
                **meta
            }
            
            if meta.get("status") == "success" and ext in CONFIG["extract_text_exts"]:
                extracted = _extract_text_from_file(save_path, ext)
                if extracted:
                    att_data["extracted_text"] = extracted
                    extracted_texts.append(f"\n\n[첨부파일 내용: {name}]\n{extracted}")
            
            processed_atts.append(att_data)
        
        if extracted_texts:
            content += "".join(extracted_texts)   
            
        self.save_post({
            "dept": self.dept,
            "detail": self.detail,
            "title": title,
            "date": date,
            "url": link,
            "content": content,
            "images": processed_images,
            "attachments": processed_atts
        })
        return True
    
# ==========================================
# Type A: 일반형 (기존과 동일)
# ==========================================
class TypeACrawler(BaseCrawler):
    def crawl(self):
        page = 1
        while True:
            print(f"[{self.dept}_{self.detail}] Page {page}...")
            if '?' in self.base_url: url = f"{self.base_url}&page={page}"
            else: url = f"{self.base_url}?page={page}"
            
            soup = self.fetch_page(url)
            if not soup: break
            
            rows = soup.select(".board_body tbody tr")
            if not rows: 
                print("   ㄴ 더 이상 게시글이 없습니다.")
                break

            found_regular_post = False
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 4: continue
                
                is_notice = not cols[0].get_text(strip=True).isdigit()
                if page > 1 and is_notice: continue 

                found_regular_post = True
                title_elem = row.select_one("td.left a")
                if not title_elem: continue
                
                link = urljoin(self.base_url, title_elem['href'])
                for span in title_elem.find_all('span'): span.decompose()
                title = title_elem.get_text(strip=True)
                date = cols[3].get_text(strip=True)

                if page > 1 and date < self.last_crawled_date:
                    print(f"   ㄴ [Stop] 기준 날짜({self.last_crawled_date}) 도달: {date}")
                    return

                self.process_detail_page(title, date, link,page)
            
            if page > 1 and not found_regular_post:
                print("   ㄴ [End] 일반 게시글 없음. 종료합니다.")
                break
            page += 1
import datetime  # 파일 최상단에 없다면 추가 필요

# ==========================================
# Type B: 그누보드 
# ==========================================
class TypeBCrawler(BaseCrawler):
    def crawl(self):
        page = 1
        while True:
            print(f"[{self.dept}_{self.detail}] Page {page}...")
            if '?' in self.base_url: url = f"{self.base_url}&page={page}"
            else: url = f"{self.base_url}?page={page}"

            soup = self.fetch_page(url)
            if not soup: break
            
            rows = soup.select(".tbl_head01 tbody tr, .basic_tbl_head tbody tr")
            is_list_style = False
            
            if not rows:
                rows = soup.select(".max_board li, .list_board li, .webzine li")
                if rows: is_list_style = True
            
            if not rows: 
                print("   ㄴ 더 이상 게시글이 없습니다.")
                break
            
            found_regular_post = False
            
            # [수정] 2페이지까지는 강제 수집 모드 활성화
            is_force_mode = (page <= 2)

            for row in rows:
                is_notice = False
                
                if is_list_style:
                    if row.select_one(".notice_icon"): is_notice = True
                    link_tag = row.find("a")
                    if not link_tag: continue
                    link = urljoin(self.base_url, link_tag['href'])
                    title_tag = row.select_one("h2") or row.select_one(".subject")
                    title = title_tag.get_text(strip=True) if title_tag else "No Title"
                    date_tag = row.select_one(".date")
                    date = date_tag.get_text(strip=True) if date_tag else "2025-01-01"
                
                else:
                    subj_div = row.select_one(".bo_tit a") or row.select_one(".td_subject a")
                    if not subj_div: continue
                    
                    num_elem = row.select_one(".td_num2") or row.select_one(".td_num")
                    if num_elem and not num_elem.get_text(strip=True).isdigit(): is_notice = True
                    if "bo_notice" in row.get("class", []): is_notice = True

                    link = urljoin(self.base_url, subj_div['href'])
                    title = subj_div.get_text(strip=True)
                    
                    date_elem = row.select_one(".td_datetime")
                    if not date_elem: continue
                    date = date_elem.get_text(strip=True)
                    
                    if len(date) == 5 and date[2] == '-':
                        today = datetime.date.today()
                        try:
                            post_month = int(date[:2])
                            year = today.year - 1 if post_month > today.month else today.year
                            date = f"{year}-{date}"
                        except:
                            date = f"{today.year}-{date}"
                    
                    elif len(date) == 8 and date[2] == '-':
                        date = "20" + date

                if page > 1 and is_notice: continue
                found_regular_post = True

                # [수정] 3페이지부터만 목록 날짜 체크로 조기 중단
                if not is_force_mode and page > 2 and date < self.last_crawled_date:
                    print(f"   ㄴ [Stop] 기준 날짜({self.last_crawled_date}) 도달: {date}")
                    return

                success = self.process_detail_page(title, date, link, referer=url, force_save=is_force_mode)

                if not is_force_mode and not success:
                    return
            
            if page > 1 and not found_regular_post:
                print("   ㄴ [End] 일반 게시글 없음. 종료합니다.")
                break
            page += 1

# ==========================================
# Type C: 통합홈페이지 (일반/Q&A 범용 호환)
# ==========================================
class TypeCCrawler(BaseCrawler):
    def crawl(self):
        current_url = self.base_url
        page_cnt = 0
        
        while True:
            page_cnt += 1
            print(f"[{self.dept}_{self.detail}] Page {page_cnt} reading...")
            
            soup = self.fetch_page(current_url)
            if not soup: break
            
            rows = soup.select(".board_list tbody tr")
            if not rows: 
                print("   ㄴ 더 이상 게시글이 없습니다.")
                break

            found_regular_post = False
            
            # [추가] 2페이지까지는 강제 수집 모드
            is_force_mode = (page_cnt <= 2)

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 3: continue 
                
                # 공지글 판단
                first_col_text = cols[0].get_text(strip=True)
                is_notice = not first_col_text.isdigit()
                
                if page_cnt > 1 and is_notice: continue 

                # 비밀글 스킵
                if row.select("img[src*='secret'], img[alt*='비밀'], img[src*='lock']"):
                    continue

                # 제목 및 링크 찾기
                title_tag = None
                for col in cols[1:]:
                    a = col.find("a")
                    if a and a.get_text(strip=True):
                        title_tag = a
                        break
                
                if not title_tag: continue
                
                title = title_tag.get_text(strip=True)
                link = urljoin(self.base_url, title_tag['href'])

                # 날짜 찾기
                date = ""
                for col in cols:
                    txt = col.get_text(strip=True)
                    match = re.search(r"20\d{2}[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])", txt)
                    if match:
                        date = match.group(0).replace('/', '-').replace('.', '-')
                        break
                
                if not date: date = "2025-01-01"

                # [수정] 3페이지부터만 목록 날짜 체크로 중단
                if not is_force_mode and page_cnt > 2 and date < self.last_crawled_date:
                    print(f"   ㄴ [Stop] 기준 날짜({self.last_crawled_date}) 도달: {date}")
                    return
                
                found_regular_post = True
                
                # [수정] process_detail_page 호출 시 force_save 전달 및 반환값 체크
                success = self.process_detail_page(title, date, link, referer=current_url, force_save=is_force_mode)
                
                # 강제 모드가 아닌데(3페이지 이상) 상세 페이지 날짜가 옛날이면 종료
                if not is_force_mode and not success:
                    return

            # 페이지네이션 처리
            if page_cnt > 1 and not found_regular_post:
                print("   ㄴ [End] 유효한 게시글 없음. 종료.")
                break

            paging = soup.select_one(".paging")
            if paging:
                current_strong = paging.find("strong")
                if current_strong:
                    next_tag = current_strong.find_next_sibling("a")
                    if next_tag:
                        current_url = urljoin(self.base_url, next_tag['href'])
                    else: break
                else: break
            else: break

# ==========================================
# 메인 실행부
# ==========================================
def get_crawler(target):
    url = target['url']
    if "home.knu.ac.kr" in url: return TypeCCrawler(target)
    elif "board" in url: return TypeBCrawler(target)
    else: return TypeACrawler(target)

def process(target):
    try:
        crawler = get_crawler(target)
        crawler.crawl()
    except Exception as e:
        print(f"[Fatal] {target['dept']} Error: {e}")

if __name__ == "__main__":
    targets = []
    master_file = "src\crawl\knu.jsonl"
    
    if os.path.exists(master_file):
        with open(master_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "dept" in data and "url" in data:
                        if "detail" not in data:
                            data["detail"] = "공지"
                        targets.append(data)
                except Exception as e:
                    print(f"[Skip] JSON Load Error: {e}")
        print(f"Loaded {len(targets)} targets from {master_file}")
    else:
        print(f"[Error] {master_file} not found!")

    if targets:
        print(f"Starting crawler with {CONFIG['max_workers']} workers...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            executor.map(process, targets)
            
        end_time = time.time()
        print(f"\nAll tasks completed in {end_time - start_time:.2f} seconds.")
