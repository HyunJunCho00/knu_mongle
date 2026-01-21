import pandas as pd
from playwright.async_api import async_playwright
import asyncio
import os

# ==============================================================================
# 1. 컬럼 매핑 정의 (WebSquare 내부 변수명 -> 한글 헤더)
# ==============================================================================
# 이전 HTML 분석을 통해 확인된 ID 값들입니다.
COLUMN_MAPPING = {
    # [메타 데이터]
    "Category1": "대분류",
    "Category2": "중분류",
    "Category3": "소분류",
    
    "estblYear": "개설연도",
    "estblSmstrSctnm": "개설학기",
    "estblGrade": "학년",
    "sbjetSctnm": "교과구분",
    "estblUnivNm": "개설대학",
    "estblDprtnNm": "개설학과",
    "crseNo": "강좌번호", 
    "sbjetNm": "교과목명",
    "crdit": "학점",
    "thryTime": "강의시수",
    "prctsTime": "실습시수",
    "totalPrfssNm": "담당교수",
    "lssnsTimeInfo": "강의시간",
    "lssnsRealTimeInfo": "강의시간(실제)",
    "lctrmInfo": "강의실",
    "rmnmCd": "호실번호",
    "attlcPrscpCnt": "수강정원",
    "appcrCnt": "수강신청인원",
    "pckgeRqstCnt": "꾸러미신청인원",
    "pckgeRqstPssblYn": "꾸러미가능여부",
    "grdscCmmnnSbjetYn": "대학원공통여부",
    "rmrk": "비고"
}

# ==============================================================================
# 2. 헬퍼 함수 (기존과 동일)
# ==============================================================================

async def wait_for_loading(page):
    try:
        await page.wait_for_timeout(300)
        if await page.locator("#__progressModal").is_visible():
            await page.locator("#__progressModal").wait_for(state="hidden", timeout=5000)
        await page.wait_for_load_state("networkidle", timeout=3000)
    except:
        await page.wait_for_timeout(500)

async def force_select(page, selector, value):
    try:
        await page.select_option(selector, value=value)
        await page.evaluate(f"""
            var select = document.querySelector('{selector}');
            select.dispatchEvent(new Event('change', {{ bubbles: true }}));
            select.dispatchEvent(new Event('blur', {{ bubbles: true }}));
        """)
        await wait_for_loading(page)
    except: pass

async def get_options(page, selector):
    if not await page.is_visible(selector): return []
    options = await page.eval_on_selector_all(
        f"{selector} option", 
        "options => options.map(o => ({ text: o.innerText.trim(), value: o.value }))"
    )
    return [o for o in options if o['value'] and "선택" not in o['text']]

async def setup_semester(page, year, semester):
    print(f" 검색 조건 설정: {year}년 {semester}...")
    try:
        await page.fill("#schEstblYear___input", str(year))
        await page.press("#schEstblYear___input", "Enter")
        await wait_for_loading(page)
    except: pass

    semester_options = await page.eval_on_selector_all(
        "#schEstblSmstrSctcd option",
        "opts => opts.map(o => ({ text: o.innerText.trim(), value: o.value }))"
    )
    target_val = next((o['value'] for o in semester_options if semester in o['text']), None)
    
    if target_val:
        await force_select(page, "#schEstblSmstrSctcd", target_val)
    else:
        print(f"  ❌ 학기 옵션 없음: {semester}")
    await page.wait_for_timeout(1000)

# ==============================================================================
# 3. 데이터 추출 (모든 컬럼 수집)
# ==============================================================================

async def extract_all_columns_json(page, cat1, cat2, cat3):
    try:
        await page.click("input#btnSearch")
        await wait_for_loading(page)

        if await page.locator("#grid01_noresult").is_visible():
            return []

        # ⚡ WebSquare 원본 JSON 통째로 가져오기
        raw_data = await page.evaluate("""
            () => {
                try {
                    if (typeof grid01 !== 'undefined') {
                        return grid01.getAllJSON();
                    }
                    return null;
                } catch(e) { return null; }
            }
        """)

        if not raw_data: return []

        processed_data = []
        for row in raw_data:
            # 필수 데이터 확인
            if not row.get('crseNo') or not row.get('sbjetNm'): continue

            # 분류 정보 추가
            row['Category1'] = cat1
            row['Category2'] = cat2
            row['Category3'] = cat3
            
            # 원본 행 그대로 리스트에 추가 (나중에 Pandas에서 컬럼 정리)
            processed_data.append(row)

        if processed_data:
            # 예시 출력 (첫 번째 과목명)
            ex_name = processed_data[0].get('sbjetNm', 'Unknown')
            print(f"  ✅ 수집: {cat2} > {cat3} | {len(processed_data)}건 (예: {ex_name})")
        
        return processed_data

    except Exception:
        try: await page.keyboard.press("Enter")
        except: pass
        return []

# ==============================================================================
# 4. 메인 실행 및 CSV 저장 (컬럼 매핑 적용)
# ==============================================================================

async def scrape_knu_full_mode(target_year="2025", target_semester="1학기"):
    all_courses = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(10000)
        
        print(f"KNU 수강편람 [모든 컬럼] 수집 시작...")
        await page.goto("https://sy.knu.ac.kr/_make/lect/lect_list.php")
        await page.wait_for_load_state("networkidle")

        await setup_semester(page, target_year, target_semester)

        level1_options = await get_options(page, "select#schSbjetCd1")
        
        for l1 in level1_options:
            l1_name = l1['text']
            # 테스트 시에는 '대학', '교양'만 주석 해제해서 확인 권장
            # if "대학" not in l1_name and "교양" not in l1_name: continue
            
            print(f"\n📂 [대분류] {l1_name}")
            await force_select(page, "select#schSbjetCd1", l1['value'])

            level2_options = await get_options(page, "select#schSbjetCd2")
            if not level2_options:
                data = await extract_all_columns_json(page, l1_name, "N/A", "N/A")
                all_courses.extend(data)
                continue

            for l2 in level2_options:
                l2_name = l2['text']
                await force_select(page, "select#schSbjetCd2", l2['value'])

                level3_options = await get_options(page, "select#schSbjetCd3")
                if not level3_options:
                    data = await extract_all_columns_json(page, l1_name, l2_name, "N/A")
                    all_courses.extend(data)
                    continue

                for l3 in level3_options:
                    await force_select(page, "select#schSbjetCd3", l3['value'])
                    data = await extract_all_columns_json(page, l1_name, l2_name, l3['text'])
                    all_courses.extend(data)

        await browser.close()

    # 데이터 저장 처리 (Pandas Magic)
    if all_courses:
        df = pd.DataFrame(all_courses)
        
        # 1. 중복 제거
        df = df.drop_duplicates(subset=['crseNo']) # 강좌번호 기준
        
        # 2. 필요한 컬럼만 선택하고 이름 바꾸기 (Rename)
        # 매핑 딕셔너리에 있는 컬럼들만 남기고, 한국어 이름으로 변경
        available_columns = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
        df_final = df[available_columns].rename(columns=COLUMN_MAPPING)
        
        print("-" * 50)
        print(f"총 강좌 수: {len(df_final)}")
        print(f"수집된 컬럼: {list(df_final.columns)}")
        
        filename = f"knu_full_data_{target_year}_{target_semester}.csv"
        df_final.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {os.path.abspath(filename)}")
    else:
        print("데이터 없음")

if __name__ == "__main__":
    # 필요한 학기로 설정
    asyncio.run(scrape_knu_full_mode("2026", "1학기"))

