import os
import re
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import time

# --- CONFIG ---
# <<< QUAN TRỌNG: SỬA LẠI ĐƯỜNG DẪN NÀY CHO ĐÚNG >>>
DATA_SOURCE_DIR = "data/raw/Corpus" 
VECTOR_STORE_PATH = "data/processed/faiss_index_medical"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# Danh sách bệnh tim mạch mục tiêu
TARGET_DISEASES = [
    'tang-huyet-ap.html', 'benh-mach-vanh.html', 'xo-vua-dong-mach-vanh.html', 'thieu-mau-co-tim.html',
    'thieu-mau-co-tim-cuc-bo-man-tinh.html', 'roi-loan-lipid-mau.html', 'suy-tim.html',
    'suy-tim-man-tinh.html', 'suy-tim-phai.html', 'suy-tim-trai.html', 'rung-nhi.html', 'dot-quy.html',
    'dot-quy-thieu-mau-cuc-bo.html', 'dai-thao-duong.html', 'dai-thao-duong-thai-ky.html'
]

def clean_text(text):
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_content_from_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
    except Exception as e:
        print(f"  Lỗi khi đọc file {filepath}: {e}")
        return None, []

    disease_name_tag = soup.find('h1')
    disease_name = disease_name_tag.get_text(strip=True) if disease_name_tag else "Không rõ tên bệnh"
    
    content_area = soup.find('body') 
    if not content_area:
        return disease_name, []

    structured_content = []
    current_section_title = "Tổng quan"
    current_section_text = []
    
    start_element = disease_name_tag.find_next_sibling() if disease_name_tag else content_area.find()
    if not start_element: return disease_name, []
    
    for element in start_element.find_next_siblings():
        if element.name in ['h2', 'h3']:
            if current_section_text:
                full_text = "\n".join(current_section_text)
                structured_content.append({"section": current_section_title, "content": clean_text(full_text)})
            current_section_title = element.get_text(strip=True)
            current_section_text = []
        elif element.name in ['p', 'ul', 'ol', 'div']:
            if element.name == 'div' and (not element.get_text(strip=True) or element.find(['script', 'style', 'nav', 'iframe'])):
                continue
            text = element.get_text(strip=True, separator='\n')
            if text and 'Mục lục' not in element.find_previous(['h3', 'h2'],"").get_text():
                 current_section_text.append(text)

    if current_section_text:
        full_text = "\n".join(current_section_text)
        structured_content.append({"section": current_section_title, "content": clean_text(full_text)})

    return disease_name, structured_content

def load_and_process_data():
    all_chunks = []
    print(f"Bắt đầu quét thư mục: '{DATA_SOURCE_DIR}'")
    if not os.path.exists(DATA_SOURCE_DIR):
        print(f"LỖI: Thư mục '{DATA_SOURCE_DIR}' không tồn tại.")
        return []

    for disease_filename in TARGET_DISEASES:
        filepath = os.path.join(DATA_SOURCE_DIR, disease_filename)
        if os.path.exists(filepath):
            print(f"-> Đang xử lý file: {disease_filename}")
            disease_name, sections = extract_content_from_html(filepath)
            if not sections:
                print(f"  Cảnh báo: Không trích xuất được nội dung từ file {filepath}")
                continue
            for section_data in sections:
                if not section_data.get('content') or not section_data.get('section'): continue
                content = f"Thông tin về bệnh {disease_name}, mục {section_data['section']}: {section_data['content']}"
                doc = Document(page_content=content, metadata={"source": disease_name, "section": section_data['section']})
                all_chunks.append(doc)
        else:
            print(f"  Cảnh báo: Không tìm thấy file '{disease_filename}'")
    
    print(f"\n=> Đã tạo tổng cộng {len(all_chunks)} chunks.")
    return all_chunks

def create_vector_store(chunks):
    print("\nBắt đầu quá trình embedding với mô hình local...")
    start_time = time.time()
    
    # Sử dụng mô hình embedding local
    model_kwargs = {'device': 'cpu'} # Chạy trên CPU, đổi thành 'cuda' nếu có GPU
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
    end_time = time.time()
    print(f"=> Embedding hoàn tất trong {end_time - start_time:.2f} giây.")
    
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"=> Đã lưu Vector Store vào thư mục: {VECTOR_STORE_PATH}")

if __name__ == '__main__':
    processed_chunks = load_and_process_data()
    if processed_chunks:
        create_vector_store(processed_chunks)
    else:
        print("\nKhông có chunk nào được tạo. Vui lòng kiểm tra lại cấu trúc thư mục và code.")