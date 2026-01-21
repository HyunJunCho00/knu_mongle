import time
import json
import hashlib
import pandas as pd
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

class KnuCurriculumScraper:
    def __init__(self):
        self.url = "https://knuin.knu.ac.kr/public/stddm/edu.knu"
        self.guidelines = []
        self.roadmaps = []
        self.target_years = ["2025", "2026"] 
        self.alert_triggered = False

    def get_data_hash(self, data_list):
        if not data_list: return None
        return hashlib.md5(json.dumps(data_list, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def parse_grid(self, html, grid_id, meta_info):
        soup = BeautifulSoup(html, 'html.parser')
        rows = soup.select(f"#{grid_id}_body_table tbody tr")
        extracted_data = []
        current_grade = "" 

        for row in rows:
            if "조회된 내역이 없습니다" in row.get_text(): continue
            try:
                if grid_id == 'grid01': 
                    cat_cell = row.select_one("td[col_id='complMnulSubjt']")
                    cont_cell = row.select_one("td[col_id='cntns']")
                    if cat_cell and cont_cell:
                        cat_text = cat_cell.get_text(strip=True)
                        cont_text = cont_cell.get_text(strip=True)
                        if cat_text:
                            data = meta_info.copy()
                            data.update({"구분": cat_text, "내용": cont_text})
                            extracted_data.append(data)
                elif grid_id == 'grid03': 
                    grade_cell = row.select_one("td[col_id='estblGrade']")
                    grade_text = grade_cell.get_text(strip=True) if grade_cell else ""
                    if grade_text: current_grade = grade_text
                    if not current_grade: continue

                    sub1_nm = row.select_one("td[col_id='sbjetNm1']")
                    sub1_cr = row.select_one("td[col_id='crditSystem1']")
                    sub1_code = row.select_one("td[col_id='sbjetCd1']")
                    if sub1_nm and sub1_nm.get_text(strip=True):
                        data = meta_info.copy()
                        data.update({
                            "학년": current_grade, "학기": "1학기",
                            "교과목명": sub1_nm.get_text(strip=True),
                            "학점": sub1_cr.get_text(strip=True) if sub1_cr else "",
                            "과목코드": sub1_code.get_text(strip=True) if sub1_code else ""
                        })
                        extracted_data.append(data)
                    sub2_nm = row.select_one("td[col_id='sbjetNm2']")
                    sub2_cr = row.select_one("td[col_id='crditSystem2']")
                    sub2_code = row.select_one("td[col_id='sbjetCd2']")
                    if sub2_nm and sub2_nm.get_text(strip=True):
                        data = meta_info.copy()
                        data.update({
                            "학년": current_grade, "학기": "2학기",
                            "교과목명": sub2_nm.get_text(strip=True),
                            "학점": sub2_cr.get_text(strip=True) if sub2_cr else "",
                            "과목코드": sub2_code.get_text(strip=True) if sub2_code else ""
                        })
                        extracted_data.append(data)
            except Exception:
                continue
        return extracted_data

    async def handle_dialog(self, dialog):
        self.alert_triggered = True
        try:
            await dialog.accept()
        except: pass

    async def select_option_safely(self, page, selector, value, retries=3):
        for i in range(retries):
            try:
                await page.select_option(selector, value, force=True)
                await asyncio.sleep(0.3)
                current_val = await page.input_value(selector)
                if str(current_val).strip() == str(value).strip():
                    await page.evaluate(f"document.querySelector('{selector}').dispatchEvent(new Event('change', {{bubbles:true}}))")
                    return True
                
                # 실패 시 JS 주입
                await page.evaluate(f"""(arg) => {{
                    let s = document.querySelector(arg.sel);
                    s.value = arg.val;
                    s.dispatchEvent(new Event('change', {{bubbles:true}}));
                }}""", {'sel': selector, 'val': value})
                await asyncio.sleep(0.3)
                
                current_val = await page.input_value(selector)
                if str(current_val).strip() == str(value).strip():
                    return True
            except: pass
            await asyncio.sleep(0.5)
        return False

    async def fetch_year_data(self, page, year, compare_hash=None):
        self.alert_triggered = False
        
        # 1. 연도 입력
        try:
            await page.fill("#schTrgtYrsf___input", year)
            await page.press("#schTrgtYrsf___input", "Enter")
            await asyncio.sleep(0.5)
        except: return False, None, None

        # 2. 조회 클릭
        try:
            await page.click("#udcBtns_btnSearch", force=True)
        except: pass

        # 3. 데이터 확인
        for _ in range(3):
            if self.alert_triggered: return False, None, None

            try:
                await page.click("#tabControl1_tab_tabs2_tabHTML", force=True)
                await asyncio.sleep(0.3)
                html = await page.inner_html("#tabControl1_contents_content2_body")
                
                if "조회된 내역이 없습니다" in html: return False, None, None

                temp_data = self.parse_grid(html, 'grid01', {})
                current_hash = self.get_data_hash(temp_data)

                if current_hash and current_hash != compare_hash:
                    # 탭3 수집
                    await page.click("#tabControl1_tab_tabs3_tabHTML", force=True)
                    await asyncio.sleep(0.5)
                    html3 = await page.inner_html("#tabControl1_contents_content3_body")
                    
                    return True, {
                        'year': year,
                        'guidelines': temp_data,
                        'roadmaps': self.parse_grid(html3, 'grid03', {})
                    }, current_hash
                
                await asyncio.sleep(1.0)
            except: pass
            
        return False, None, None

    async def run(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = await browser.new_context(viewport={'width': 1280, 'height': 1024})
            page = await context.new_page()
            page.on("dialog", self.handle_dialog)

            print("🚀 경북대 교육과정 크롤링 시작 (학과별 완전 초기화 모드)...")
            
            # 1. 메타데이터 수집을 위한 최초 접속
            await page.goto(self.url)
            await page.wait_for_selector("#schSbjetCd1", state="attached", timeout=60000)
            
            # 대학 선택 (학부)
            await self.select_option_safely(page, "#schSbjetCd1", "") # 초기화 트리거
            await page.evaluate("""() => {
                const opts = document.querySelectorAll('#schSbjetCd1 option');
                for (let opt of opts) {
                    if (opt.text.includes('대학') && !opt.text.includes('대학원')) {
                        document.querySelector('#schSbjetCd1').value = opt.value;
                        document.querySelector('#schSbjetCd1').dispatchEvent(new Event('change'));
                        break;
                    }
                }
            }""")
            await asyncio.sleep(1.0)

            # [중요] 단과대 목록과 학과 목록을 미리 다 수집해놓습니다.
            # (새로고침을 계속 할 것이므로, 구조를 미리 파악해야 함)
            structure = []
            
            college_options = await page.evaluate("""() => {
                const opts = Array.from(document.querySelectorAll('#schSbjetCd2 option'));
                return opts.filter(o => o.value && o.text !== '선택').map(o => ({text: o.text, value: o.value}));
            }""")

            for col in college_options:
                await self.select_option_safely(page, "#schSbjetCd2", col['value'])
                await asyncio.sleep(0.5)
                
                depts = await page.evaluate("""() => {
                    const opts = Array.from(document.querySelectorAll('#schSbjetCd3 option'));
                    return opts.filter(o => o.value && !o.text.includes('선택')).map(o => ({text: o.text, value: o.value}));
                }""")
                structure.append({'college': col, 'depts': depts})
            
            print(f"📋 구조 파악 완료. 총 {len(structure)}개 단과대 순회 시작.")

            # =================================================================
            # 본격적인 크롤링 루프 (구조 정보 기반)
            # =================================================================
            for group in structure:
                college = group['college']
                print(f"\n▶ [{college['text']}] 순회 시작")

                for dept in group['depts']:
                    # [핵심] 학과가 바뀔 때마다 새로고침 -> 백지 상태로 시작
                    await page.reload()
                    await page.wait_for_selector("#schSbjetCd1", state="attached")
                    await asyncio.sleep(0.5)

                    # 1. 대학 재선택
                    await page.evaluate("""() => {
                        const opts = document.querySelectorAll('#schSbjetCd1 option');
                        for (let opt of opts) {
                            if (opt.text.includes('대학') && !opt.text.includes('대학원')) {
                                document.querySelector('#schSbjetCd1').value = opt.value;
                                document.querySelector('#schSbjetCd1').dispatchEvent(new Event('change'));
                                break;
                            }
                        }
                    }""")
                    await asyncio.sleep(0.5)

                    # 2. 단과대 재선택
                    await self.select_option_safely(page, "#schSbjetCd2", college['value'])
                    await asyncio.sleep(0.5)

                    # 3. 학과 선택
                    print(f"  - [{dept['text']}] 처리 중...", end=" ")
                    if not await self.select_option_safely(page, "#schSbjetCd3", dept['value']):
                        print("❌ 학과 선택 실패")
                        continue
                    
                    await asyncio.sleep(0.8) # 전공 로딩 대기

                    # 4. 세부전공 확인
                    major_opts = await page.evaluate("""() => {
                        const select4 = document.querySelector('#schSbjetCd4');
                        if (!select4 || select4.disabled || select4.offsetParent === null) return [];
                        const opts = Array.from(select4.querySelectorAll('option'));
                        return opts.filter(o => o.value && !o.text.includes('선택')).map(o => ({
                            text: o.text.trim(), value: o.value
                        }));
                    }""")

                    # 5. 타겟 설정 (세부전공 있으면 Loop, 없으면 단일)
                    targets = []
                    if major_opts:
                        # 세부전공이 있으면 현재 페이지 상태에서 전공만 바꿔가며 조회
                        # (단, 전공 간 데이터 오염 방지를 위해 각 전공 조회 전 '선택'으로 돌리는 게 안전하지만
                        # 여기서는 비교 로직이 있으므로 전공 loop는 그냥 진행)
                        for m in major_opts:
                            targets.append({'name': f"{dept['text']} {m['text']}", 'val': m['value'], 'is_major': True})
                    else:
                        targets.append({'name': dept['text'], 'val': None, 'is_major': False})

                    # 6. 실제 데이터 조회
                    for target in targets:
                        if target['is_major']:
                            print(f"    👉 [{target['name'].split()[-1]}]...", end=" ")
                            await self.select_option_safely(page, "#schSbjetCd4", target['val'])
                            await asyncio.sleep(0.5)
                        
                        final_data = None
                        
                        # 2025 조회 (Baseline)
                        # 여기서는 화면이 깨끗하므로 compare_hash = None
                        ok_25, data_25, hash_25 = await self.fetch_year_data(page, "2025", None)
                        if ok_25: final_data = data_25
                        
                        # 2026 조회 (Override)
                        # 2025년 데이터가 있으면 그것과 달라야 함
                        compare = hash_25 if ok_25 else None
                        ok_26, data_26, hash_26 = await self.fetch_year_data(page, "2026", compare)
                        if ok_26: final_data = data_26

                        # 저장
                        if final_data:
                            prev_txt = final_data['guidelines'][0]['구분'] if final_data['guidelines'] else (
                                final_data['roadmaps'][0]['교과목명'] if final_data['roadmaps'] else "내용없음"
                            )
                            print(f"✅ {final_data['year']}년 확정 (내용: {prev_txt})")
                            
                            meta = {"대학": college['text'], "학과": target['name'], "연도": final_data['year']}
                            for item in final_data['guidelines']:
                                item.update(meta)
                                self.guidelines.append(item)
                            for item in final_data['roadmaps']:
                                item.update(meta)
                                self.roadmaps.append(item)
                        else:
                            print("⏭️ 데이터 없음")

            await browser.close()

        # 저장
        pd.DataFrame(self.guidelines).to_csv("knu_guide_final.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.roadmaps).to_csv("knu_road_final.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    scraper = KnuCurriculumScraper()
    scraper.run()