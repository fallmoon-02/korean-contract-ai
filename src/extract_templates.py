import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

# 1) 경로 설정
TEMPLATE_DIR = os.path.join(os.getcwd(), "templates")
OUTPUT_CSV = os.path.join(os.getcwd(), "data", "templates_raw.csv")

# 2) PDF 텍스트 추출 함수
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

# 3) DOCX 텍스트 추출 함수
def extract_text_from_docx(path):
    doc = Document(path)
    paragraphs = [para.text for para in doc.paragraphs]
    return "\n".join(paragraphs)

# 4) 전체 템플릿 디렉터리 순회하며 CSV로 저장
def main():
    rows = []
    for fname in os.listdir(TEMPLATE_DIR):
        file_path = os.path.join(TEMPLATE_DIR, fname)
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif fname.lower().endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            continue
        template_id = os.path.splitext(fname)[0]
        rows.append({"template_id": template_id, "full_text": text})

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved raw templates to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
