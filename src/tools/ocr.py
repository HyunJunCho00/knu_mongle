import re
import os

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[Warning] easyocr not installed. OCR functionality will be limited.")

class OCRTool:
    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
            except Exception as e:
                print(f"[Warning] Failed to initialize EasyOCR: {e}")

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        """
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        if not self.reader:
            return "Error: OCR reader not initialized. Please install easyocr."
        
        try:
            print(f"Processing image: {image_path}")
            results = self.reader.readtext(image_path)
            
            extracted_lines = []
            for detection in results:
                text = detection[1]
                confidence = detection[2]
                if confidence > 0.3:
                    extracted_lines.append(text)
            
            return "\n".join(extracted_lines)
        
        except Exception as e:
            return f"Error during OCR processing: {str(e)}"

    def parse_visa_info(self, text: str) -> dict:
        """
        Parse extracted text to identify visa-related information.
        """
        info = {
            "name": None,
            "passport_number": None,
            "visa_type": None,
            "nationality": None,
            "date_of_birth": None,
            "issue_date": None,
            "expiry_date": None
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line_upper = line.upper()
            
            if "NAME" in line_upper or "이름" in line:
                name_match = re.search(r'[:：]\s*([A-Z\s]+)', line_upper)
                if name_match:
                    info["name"] = name_match.group(1).strip()
            
            if "PASSPORT" in line_upper or "여권" in line:
                passport_match = re.search(r'[A-Z]\d{8}', line)
                if passport_match:
                    info["passport_number"] = passport_match.group(0)
            
            if "VISA" in line_upper or "사증" in line:
                visa_types = ["D-2", "D-4", "F-2", "F-4", "E-7", "H-1"]
                for vtype in visa_types:
                    if vtype in line_upper:
                        info["visa_type"] = vtype
                        break
            
            if "NATIONALITY" in line_upper or "국적" in line:
                nat_match = re.search(r'[:：]\s*([A-Za-z]+)', line)
                if nat_match:
                    info["nationality"] = nat_match.group(1).strip()
            
            date_pattern = r'\d{4}[-/.]\d{2}[-/.]\d{2}'
            dates = re.findall(date_pattern, line)
            
            if "BIRTH" in line_upper or "생년월일" in line:
                if dates:
                    info["date_of_birth"] = dates[0]
            
            if "ISSUE" in line_upper or "발급" in line:
                if dates:
                    info["issue_date"] = dates[0]
            
            if "EXPIR" in line_upper or "만료" in line or "유효" in line:
                if dates:
                    info["expiry_date"] = dates[0]
        
        return {k: v for k, v in info.items() if v is not None}

    def extract_form_fields(self, text: str) -> dict:
        """
        Extract common form fields from OCR text.
        """
        fields = {}
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            field_patterns = {
                "student_id": r'\d{10}',
                "phone": r'\d{3}[-\s]?\d{3,4}[-\s]?\d{4}',
                "email": r'[\w\.-]+@[\w\.-]+\.\w+',
                "address": r'(서울|대구|부산|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주).+',
            }
            
            for field_name, pattern in field_patterns.items():
                matches = re.findall(pattern, line_clean)
                if matches and field_name not in fields:
                    fields[field_name] = matches[0]
        
        return fields