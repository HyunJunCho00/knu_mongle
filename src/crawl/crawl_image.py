from io import BytesIO
from PIL import Image
import subprocess
import hashlib
from core.config import Settings
from groq import Groq
from pathlib import Path
# ==========================================
# VLM 모델 로딩 
# ==========================================
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  
print(f"Loading VLM model: {MODEL_ID}")


groq_client = Groq(api_key=Settings.GROQ_API_KEY)

def analyze_image(image_src):
    """이미지 VLM 분석 (URL 또는 base64 data URL)"""
    if not groq_client: 
        return ""
    
    # base64 이미지가 너무 짧으면(1KB 미만) 아이콘/블릿이므로 스킵
    if image_src.startswith("data:image") and len(image_src) < 1000:
        return ""
    
    try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 이미지에서 모든 텍스트와 표, 도표의 내용을 한국어로 정확하게 추출하세요. 표는 마크다운 형식으로 변환하세요. 불필요한 설명은 생략하고 내용만 출력하세요."},
                        {"type": "image_url", "image_url": {"url": image_src}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=1500  # 표나 긴 텍스트를 대비해 토큰 수 확보
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"   [VLM Error] {e}")
        return ""
    
try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256*28*28,  # 해상도 낮춰서 속도 향상
        max_pixels=1024*28*28
    )
    print(f"✓ Model loaded successfully on {model.device}")
except Exception as e:
    print(f"[Model Load Error] {e}")
    model = None
    processor = None



def sanitize_filename(name):
    """파일명 특수문자 제거"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def _download_file(session, url, save_path: Path, referer=None):
    """파일 다운로드 """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {}
    if referer:
        headers["Referer"] = referer
    
    try:
        resp = session.get(url, headers=headers, verify=False, timeout=30, stream=True)
        resp.raise_for_status()
        
        h = hashlib.sha256()
        size = 0
        
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                h.update(chunk)
                size += len(chunk)
        
        return {
            "size": size,
            "sha256": h.hexdigest(),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _download_image_to_memory(session, url, referer=None):
    """이미지를 메모리로만 다운로드 """
    headers = {}
    if referer:
        headers["Referer"] = referer
    
    try:
        resp = session.get(url, headers=headers, verify=False, timeout=15, stream=True)
        resp.raise_for_status()
        
        # 메모리에 바이트로 저장
        image_bytes = BytesIO()
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                image_bytes.write(chunk)
        
        image_bytes.seek(0)
        return image_bytes
    except Exception as e:
        print(f"   [Image Download Error] {url}: {e}")
        return None

def analyze_image_from_memory(image_bytes: BytesIO, alt_text: str = "") -> str:
    """메모리의 이미지를 VLM으로 분석 """
    if not model or not processor or not image_bytes:
        return ""
    
    try:
        # PIL Image로 변환
        img = Image.open(image_bytes).convert('RGB')
        
        # 너무 작은 이미지는 스킵 (노이즈 이미지 필터링)
        if img.width < 100 or img.height < 100:
            return ""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "이미지의 모든 텍스트와 표를 한국어로 추출해. 표는 마크다운으로 변환해."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # GPU 락으로 병렬 충돌 방지
        with vlm_lock:
            generated_ids = model.generate(**inputs, max_new_tokens=1024)  # 토큰 수 줄여서 속도 향상
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # 메모리 정리
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result

    except Exception as e:
        print(f"   [VLM Error] {e}")
        return ""


def _extract_text_from_file(file_path: Path, ext: str) -> str:
    """PDF, DOCX, HWP 텍스트 추출"""
    text = ""
    ext = ext.lower()
    
    try:
        if ext == ".pdf":
            try:
                import fitz
                # MuPDF 에러 메시지 억제
                fitz.TOOLS.mupdf_display_errors(False)
                
                with fitz.open(str(file_path)) as doc:
                    texts = [page.get_text().strip() for page in doc if page.get_text().strip()]
                    text = "\n\n".join(texts)
            except Exception as e:
                # 에러 메시지 간소화
                if "FT_New_Memory_Face" not in str(e):
                    print(f"[PDF Extract Error] {file_path.name}: {e}")
            
        elif ext == ".docx":
            try:
                import docx
                from docx.oxml.text.paragraph import CT_P
                from docx.oxml.table import CT_Tbl
                from docx.table import _Cell, Table
                from docx.text.paragraph import Paragraph
                
                doc = docx.Document(str(file_path))
                texts = []
                
                # 문서 순서대로 모든 요소 추출
                for element in doc.element.body:
                    if isinstance(element, CT_P):  # 단락
                        p = Paragraph(element, doc)
                        if p.text.strip():
                            texts.append(p.text.strip())
                    
                    elif isinstance(element, CT_Tbl):  # 표
                        table = Table(element, doc)
                        table_text = []
                        for row in table.rows:
                            row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                            if row_text:
                                table_text.append(row_text)
                        if table_text:
                            texts.append("\n[표]\n" + "\n".join(table_text))
                
                text = "\n\n".join(texts)
                
            except Exception as e:
                print(f"[DOCX Extract Error] {file_path.name}: {e}")
                    
        elif ext == ".hwp":
            try:
                result = subprocess.run(
                    ["hwp5txt", str(file_path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout.strip()
                else:
                    raise Exception("hwp5txt returned empty or error")
                    
            except Exception as hwp_e:
                print(f"[HWP] hwp5txt 실패({hwp_e}), olefile 시도: {file_path.name}")
                try:
                    import olefile
                    if olefile.isOleFile(str(file_path)):
                        ole = olefile.OleFileIO(str(file_path))
                        texts = []
                        for entry in ole.listdir():
                            if entry[0] == "BodyText":
                                stream = ole.openstream(entry)
                                data = stream.read()
                                try:
                                    decoded = data.decode('utf-16le', errors='ignore')
                                    cleaned = ''.join(c for c in decoded if c.isprintable() or c in '\n\t ')
                                    if cleaned.strip(): 
                                        texts.append(cleaned.strip())
                                except: 
                                    pass
                        ole.close()
                        text = "\n\n".join(texts)
                except Exception as ole_e:
                    print(f"[HWP Extract Error] {file_path.name}: {ole_e}")
            
    except Exception as e:
        print(f"[Text Extract Error] {file_path.name}: {e}")
        
    return text