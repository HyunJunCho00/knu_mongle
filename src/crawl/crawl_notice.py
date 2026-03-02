import datetime
import io
import json
import os
import re
import threading
import time
import urllib3
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from crawl_config import CONFIG
    from crawl_parsers import parse_post_content
    from crawl_image import (
        _download_file,
        _download_image_to_memory,
        extract_text_with_meta,
        analyze_image_from_memory,
        sanitize_filename,
    )
except ImportError:
    from src.crawl.crawl_config import CONFIG
    from src.crawl.crawl_parsers import parse_post_content
    from src.crawl.crawl_image import (
        _download_file,
        _download_image_to_memory,
        extract_text_with_meta,
        analyze_image_from_memory,
        sanitize_filename,
    )

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

file_lock = threading.Lock()


def _normalize_id(raw_value: str, fallback_seed: str, prefix: str, fallback_label: str = "") -> str:
    base = (raw_value or "").strip().lower()
    base = re.sub(r"[^a-z0-9_-]+", "-", base)
    base = re.sub(r"-{2,}", "-", base).strip("-_")
    if base:
        return base
    if fallback_label:
        readable = sanitize_filename(fallback_label).strip().lower()
        readable = re.sub(r"\s+", "-", readable)
        readable = re.sub(r"-{2,}", "-", readable).strip("-_")
        if readable:
            return readable
    digest = hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _normalize_program_level(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower()
    if not value:
        return "all"
    if value in {"undergrad", "undergraduate", "ug"}:
        return "undergrad"
    if value in {"graduate", "grad", "ms", "phd", "master", "doctorate"}:
        return "graduate"
    if value in {"all", "common", "general"}:
        return "all"
    if any(token in value for token in ["?숈궗", "?숇?", "under"]):
        return "undergrad"
    if any(token in value for token in ["??숈썝", "?앹궗", "諛뺤궗", "grad"]):
        return "graduate"
    if any(token in value for token in ["?꾩껜", "怨듯넻", "怨듭?", "all"]):
        return "all"
    return "all"


def _normalize_source_type(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower()
    return re.sub(r"[^a-z0-9_/-]+", "", value)


def _parse_target_json(line: str) -> Optional[Dict]:
    raw = (line or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = raw
        repaired = re.sub(r'"dept":"([^"]*),"detail":', r'"dept":"\1","detail":', repaired)
        repaired = re.sub(r'"detail":"([^"]*),"url":', r'"detail":"\1","url":', repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def _default_school_name(school_id: str) -> str:
    known = {
        "knu": "Kyungpook National University",
        "snu": "Seoul National University",
    }
    return known.get(school_id, school_id.upper())


def _normalize_target(raw: Dict, fallback_school_id: str, fallback_school_name: str) -> Optional[Dict]:
    url = str(raw.get("url", "")).strip()
    if not url:
        return None

    school_id = _normalize_id(str(raw.get("school_id", fallback_school_id)), url, "school")
    school_name = str(raw.get("school_name", fallback_school_name or _default_school_name(school_id)))
    dept_name = str(raw.get("dept_name", raw.get("dept", "")) or "").strip()
    dept_id = _normalize_id(
        str(raw.get("dept_id", "")),
        fallback_seed=f"{school_id}|{dept_name}|{url}",
        prefix="dept",
        fallback_label=dept_name,
    )
    program_level = _normalize_program_level(raw.get("program_level", raw.get("detail", "all")))
    detail_label = str(raw.get("detail", program_level)).strip() or program_level
    source_type = _normalize_source_type(raw.get("source_type", ""))

    return {
        "school_id": school_id,
        "school_name": school_name,
        "dept_id": dept_id,
        "dept_name": dept_name or dept_id,
        "dept": dept_name or dept_id,
        "detail": detail_label,
        "program_level": program_level,
        "source_type": source_type,
        "url": url,
    }


def _load_targets_from_file(path: Path, fallback_school_id: str, fallback_school_name: str) -> List[Dict]:
    targets: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_target_json(line)
            if not parsed:
                continue
            normalized = _normalize_target(parsed, fallback_school_id, fallback_school_name)
            if normalized:
                targets.append(normalized)
    return targets


class BaseCrawler:
    def __init__(self, target):
        self.dept = str(target.get("dept", target.get("dept_name", target.get("dept_id", "unknown"))))
        self.detail = str(target.get("detail", target.get("program_level", "all")))
        self.base_url = target["url"]
        self.school_id = _normalize_id(
            str(target.get("school_id", "knu")),
            fallback_seed=self.base_url,
            prefix="school",
        )
        self.school_name = str(target.get("school_name", "Kyungpook National University"))
        self.dept_id = _normalize_id(
            str(target.get("dept_id", "")),
            fallback_seed=f"{self.school_id}|{self.dept}|{self.base_url}",
            prefix="dept",
            fallback_label=self.dept,
        )
        self.dept_name = self.dept
        self.program_level = _normalize_program_level(target.get("program_level", self.detail))
        self.source_type = _normalize_source_type(target.get("source_type", ""))

        school_dir = Path(CONFIG["data_dir"]) / self.school_id
        school_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = school_dir / f"{self.dept_id}.jsonl"
        self.last_crawled_date = CONFIG["cutoff_date"]

        self.session = requests.Session()
        self.session.headers.update(CONFIG["headers"])
        self.collected_links = set()
        if self.file_path.exists():
            with file_lock:
                try:
                    with self.file_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if "url" in data:
                                    self.collected_links.add(data["url"])
                            except Exception:
                                continue
                except Exception:
                    pass

    def fetch_page(self, url, referer=None):
        if referer:
            self.session.headers.update({"Referer": referer})
        try:
            time.sleep(CONFIG.get("request_delay", 0.05))
            resp = self.session.get(url, verify=False, timeout=30)
            if resp.encoding == "ISO-8859-1":
                resp.encoding = resp.apparent_encoding
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[Error] {self.dept} fetch fail: {e}")
            return None

    def save_post(self, post_data):
        if post_data["url"] in self.collected_links:
            return
        with file_lock:
            try:
                with self.file_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(post_data, ensure_ascii=False) + "\n")
                self.collected_links.add(post_data["url"])
            except Exception as e:
                print(f"[Save Error] {self.dept}: {e}")

    def _process_image(self, img_info, link, img_dir: Path):
        url = img_info["url"]
        alt = img_info.get("alt", "")
        name = sanitize_filename(os.path.basename(url.split("?")[0]) or "image.jpg")
        save_path = img_dir / name
        meta = _download_file(self.session, url, save_path, referer=link)

        description = ""
        if meta.get("status") == "success":
            image_bytes = _download_image_to_memory(self.session, url, referer=link)
            if image_bytes:
                description = analyze_image_from_memory(image_bytes, alt_text=alt)

        return {
            "url": url,
            "saved_path": str(save_path),
            "alt": alt,
            "description": description,
            **meta,
        }

    def _process_attachment(self, att, link, att_dir: Path):
        url = att["url"]
        name = sanitize_filename(att["name"])
        ext = os.path.splitext(name)[1].lower()
        save_path = att_dir / name
        meta = _download_file(self.session, url, save_path, referer=link)

        att_data = {
            "name": name,
            "url": url,
            "saved_path": str(save_path),
            **meta,
        }
        extracted_text = ""
        if meta.get("status") == "success":
            if ext in CONFIG["extract_text_exts"]:
                parsed = extract_text_with_meta(save_path, ext)
                extracted_text = str(parsed.get("text", "") or "")
                att_data.update(
                    {
                        "parser_name": parsed.get("parser_name", "none"),
                        "parser_version": parsed.get("parser_version", "unknown"),
                        "parse_confidence": parsed.get("parse_confidence", 0.0),
                        "parse_error": parsed.get("parse_error", ""),
                        "extraction_method": parsed.get("extraction_method", "text_parser"),
                    }
                )
                if extracted_text:
                    att_data["extracted_text"] = extracted_text
            elif ext in CONFIG.get("image_exts", []):
                try:
                    with open(save_path, "rb") as fp:
                        image_bytes = io.BytesIO(fp.read())
                    extracted_text = analyze_image_from_memory(image_bytes, alt_text=name)
                    att_data.update(
                        {
                            "parser_name": "vlm-fallback",
                            "parser_version": "runtime",
                            "parse_confidence": 0.75 if extracted_text else 0.0,
                            "parse_error": "" if extracted_text else "vlm_empty",
                            "extraction_method": "vlm_fallback",
                        }
                    )
                    if extracted_text:
                        att_data["extracted_text"] = extracted_text
                except Exception as exc:
                    att_data.update(
                        {
                            "parser_name": "vlm-fallback",
                            "parser_version": "runtime",
                            "parse_confidence": 0.0,
                            "parse_error": str(exc),
                            "extraction_method": "vlm_fallback",
                        }
                    )
        return att_data, extracted_text

    def process_detail_page(self, title, date, link, referer=None, force_save=False):
        if link in self.collected_links:
            return True

        soup = self.fetch_page(link, referer)
        if not soup:
            return True

        page_text = soup.get_text()
        date_match = re.search(r"20\d{2}[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])", page_text)
        if date_match:
            real_date = date_match.group(0).replace(".", "-").replace("/", "-")
            if real_date[:4] != date[:4]:
                date = real_date

        if not force_save and date < self.last_crawled_date:
            print(f"   [Stop] cutoff={self.last_crawled_date}, post={date}")
            return False

        content, images_to_save, atts_to_save = parse_post_content(soup, link)
        base_dir = Path(CONFIG["attachments_dir"]) / self.school_id / self.dept_id / date
        img_dir = base_dir / "images"
        att_dir = base_dir / "files"

        processed_images = []
        if images_to_save:
            with ThreadPoolExecutor(max_workers=CONFIG["max_image_workers"]) as executor:
                futures = [executor.submit(self._process_image, img, link, img_dir) for img in images_to_save]
                for future in as_completed(futures):
                    try:
                        image_data = future.result()
                        processed_images.append(image_data)
                    except Exception:
                        continue

        processed_atts = []
        extracted_texts = []
        if atts_to_save:
            with ThreadPoolExecutor(max_workers=CONFIG["max_file_workers"]) as executor:
                futures = [executor.submit(self._process_attachment, att, link, att_dir) for att in atts_to_save]
                for future in as_completed(futures):
                    try:
                        att_data, extracted = future.result()
                        processed_atts.append(att_data)
                        if extracted:
                            extracted_texts.append(f"\n\n[ATTACHMENT_TEXT: {att_data['name']}]\n{extracted}")
                    except Exception:
                        continue

        image_texts = []
        for image in processed_images:
            desc = (image.get("description") or "").strip()
            if desc:
                image_texts.append(f"\n\n[IMAGE_ANALYSIS]\n{desc}")

        if extracted_texts:
            content += "".join(extracted_texts)
        if image_texts:
            content += "".join(image_texts)

        self.save_post(
            {
                "school_id": self.school_id,
                "school_name": self.school_name,
                "dept_id": self.dept_id,
                "dept_name": self.dept_name,
                "dept": self.dept,
                "detail": self.detail,
                "program_level": self.program_level,
                "source_type": self.source_type,
                "title": title,
                "date": date,
                "url": link,
                "content": content,
                "images": processed_images,
                "attachments": processed_atts,
            }
        )
        return True


class TypeACrawler(BaseCrawler):
    def crawl(self):
        page = 1
        while True:
            print(f"[{self.dept}_{self.detail}] Page {page}...")
            url = f"{self.base_url}&page={page}" if "?" in self.base_url else f"{self.base_url}?page={page}"
            soup = self.fetch_page(url)
            if not soup:
                break

            rows = soup.select(".board_body tbody tr")
            if not rows:
                break

            found_regular_post = False
            is_force_mode = page <= 2
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                is_notice = not cols[0].get_text(strip=True).isdigit()
                if (not is_force_mode) and page > 1 and is_notice:
                    continue
                found_regular_post = True

                title_elem = row.select_one("td.left a")
                if not title_elem:
                    continue
                link = urljoin(self.base_url, title_elem["href"])
                for span in title_elem.find_all("span"):
                    span.decompose()
                title = title_elem.get_text(strip=True)
                date = cols[3].get_text(strip=True)

                if (not is_force_mode) and page > 2 and date < self.last_crawled_date:
                    return
                ok = self.process_detail_page(title, date, link, referer=url, force_save=is_force_mode)
                if (not is_force_mode) and not ok:
                    return

            if page > 1 and not found_regular_post:
                break
            page += 1


class TypeBCrawler(BaseCrawler):
    def crawl(self):
        page = 1
        while True:
            print(f"[{self.dept}_{self.detail}] Page {page}...")
            url = f"{self.base_url}&page={page}" if "?" in self.base_url else f"{self.base_url}?page={page}"
            soup = self.fetch_page(url)
            if not soup:
                break

            rows = soup.select(".tbl_head01 tbody tr, .basic_tbl_head tbody tr")
            is_list_style = False
            if not rows:
                rows = soup.select(".max_board li, .list_board li, .webzine li")
                is_list_style = bool(rows)
            if not rows:
                break

            found_regular_post = False
            is_force_mode = page <= 2

            for row in rows:
                is_notice = False
                if is_list_style:
                    if row.select_one(".notice_icon"):
                        is_notice = True
                    link_tag = row.find("a")
                    if not link_tag:
                        continue
                    link = urljoin(self.base_url, link_tag["href"])
                    title_tag = row.select_one("h2") or row.select_one(".subject")
                    title = title_tag.get_text(strip=True) if title_tag else "No Title"
                    date_tag = row.select_one(".date")
                    date = date_tag.get_text(strip=True) if date_tag else "2025-01-01"
                else:
                    subj_div = row.select_one(".bo_tit a") or row.select_one(".td_subject a")
                    if not subj_div:
                        continue
                    num_elem = row.select_one(".td_num2") or row.select_one(".td_num")
                    if num_elem and not num_elem.get_text(strip=True).isdigit():
                        is_notice = True
                    if "bo_notice" in row.get("class", []):
                        is_notice = True

                    link = urljoin(self.base_url, subj_div["href"])
                    title = subj_div.get_text(strip=True)
                    date_elem = row.select_one(".td_datetime")
                    if not date_elem:
                        continue
                    date = date_elem.get_text(strip=True)

                    if len(date) == 5 and date[2] == "-":
                        today = datetime.date.today()
                        try:
                            post_month = int(date[:2])
                            year = today.year - 1 if post_month > today.month else today.year
                            date = f"{year}-{date}"
                        except Exception:
                            date = f"{today.year}-{date}"
                    elif len(date) == 8 and date[2] == "-":
                        date = "20" + date

                if (not is_force_mode) and page > 1 and is_notice:
                    continue
                found_regular_post = True

                if not is_force_mode and page > 2 and date < self.last_crawled_date:
                    return

                ok = self.process_detail_page(title, date, link, referer=url, force_save=is_force_mode)
                if not is_force_mode and not ok:
                    return

            if page > 1 and not found_regular_post:
                break
            page += 1


class TypeCCrawler(BaseCrawler):
    def crawl(self):
        current_url = self.base_url
        page_cnt = 0
        while True:
            page_cnt += 1
            print(f"[{self.dept}_{self.detail}] Page {page_cnt} reading...")
            soup = self.fetch_page(current_url)
            if not soup:
                break

            rows = soup.select(".board_list tbody tr")
            if not rows:
                break
            found_regular_post = False
            is_force_mode = page_cnt <= 2

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue
                is_notice = not cols[0].get_text(strip=True).isdigit()
                if (not is_force_mode) and page_cnt > 1 and is_notice:
                    continue
                if row.select("img[src*='secret'], img[alt*='??쑬?'], img[src*='lock']"):
                    continue

                title_tag = None
                for col in cols[1:]:
                    a = col.find("a")
                    if a and a.get_text(strip=True):
                        title_tag = a
                        break
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                link = urljoin(self.base_url, title_tag["href"])

                date = ""
                for col in cols:
                    txt = col.get_text(strip=True)
                    match = re.search(r"20\d{2}[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])", txt)
                    if match:
                        date = match.group(0).replace("/", "-").replace(".", "-")
                        break
                if not date:
                    date = "2025-01-01"

                if not is_force_mode and page_cnt > 2 and date < self.last_crawled_date:
                    return

                found_regular_post = True
                ok = self.process_detail_page(title, date, link, referer=current_url, force_save=is_force_mode)
                if not is_force_mode and not ok:
                    return

            if page_cnt > 1 and not found_regular_post:
                break

            paging = soup.select_one(".paging")
            if not paging:
                break
            current_strong = paging.find("strong")
            if not current_strong:
                break
            next_tag = current_strong.find_next_sibling("a")
            if not next_tag:
                break
            current_url = urljoin(self.base_url, next_tag["href"])


def get_crawler(target):
    source_type = _normalize_source_type(target.get("source_type", ""))
    if source_type:
        type_b_sources = {"gnuboard_php", "cms_board"}
        type_c_sources = {"knu_home_sub"}
        type_a_sources = {"other", "type_a"}

        if source_type in type_c_sources:
            return TypeCCrawler(target)
        if source_type in type_b_sources:
            return TypeBCrawler(target)
        if source_type in type_a_sources:
            return TypeACrawler(target)

    url = target["url"]
    if "home.knu.ac.kr" in url:
        return TypeCCrawler(target)
    if "board" in url:
        return TypeBCrawler(target)
    return TypeACrawler(target)


def process(target):
    try:
        crawler = get_crawler(target)
        crawler.crawl()
    except Exception as e:
        name = target.get("dept_name") or target.get("dept") or target.get("dept_id") or "unknown"
        print(f"[Fatal] {name} error: {e}")


if __name__ == "__main__":
    targets = []
    schools_dir = Path("src") / "crawl" / "schools"
    if schools_dir.exists():
        for school_file in sorted(schools_dir.glob("*.jsonl")):
            school_id = _normalize_id(school_file.stem, school_file.stem, "school")
            school_name = _default_school_name(school_id)
            loaded = _load_targets_from_file(school_file, school_id, school_name)
            targets.extend(loaded)
            print(f"Loaded {len(loaded)} targets from {school_file}")

    if not targets:
        master_file = Path("src") / "crawl" / "knu_master.jsonl"
        if master_file.exists():
            loaded = _load_targets_from_file(
                master_file,
                fallback_school_id="knu",
                fallback_school_name=_default_school_name("knu"),
            )
            targets.extend(loaded)
            print(f"Loaded {len(loaded)} targets from {master_file} (fallback)")
        else:
            print(f"[Error] {master_file} not found")

    if targets:
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            executor.map(process, targets)
        end_time = time.time()
        print(f"All tasks completed in {end_time - start_time:.2f}s")

