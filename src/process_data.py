import os
import re
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import time
import shutil

# --- CONFIG (Giữ nguyên) ---
DATA_SOURCE_DIR = "data/raw/Corpus"
VECTOR_STORE_PATH = "data/processed/faiss_index_medical"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
TARGET_DISEASES = [
    'tang-huyet-ap.html', 'benh-mach-vanh.html', 'xo-vua-dong-mach-vanh.html',
    'thieu-mau-co-tim.html', 'thieu-mau-co-tim-cuc-bo-man-tinh.html',
    'roi-loan-lipid-mau.html', 'suy-tim.html', 'suy-tim-man-tinh.html',
    'suy-tim-phai.html', 'suy-tim-trai.html', 'rung-nhi.html', 'dot-quy.html',
    'dot-quy-thieu-mau-cuc-bo.html', 'dai-thao-duong.html', 'dai-thao-duong-thai-ky.html'
]

def clean_text(text):
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# ==============================================================================
# HÀM EXTRACT HOÀN TOÀN MỚI - CHIẾN LƯỢC SPLIT
# ==============================================================================
def extract_content_from_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
    except Exception as e:
        print(f"  Lỗi khi đọc file {filepath}: {e}")
        return None, []

    disease_name_tag = soup.find('h1')
    disease_name = disease_name_tag.get_text(strip=True) if disease_name_tag else os.path.basename(filepath)
    
    article_content = soup.find('div', class_='entry-content')
    if not article_content:
        article_content = soup.find('body')
        if not article_content: return disease_name, []
    
    # Dọn dẹp các thẻ không cần thiết
    for element_to_remove in article_content.find_all(['script', 'style', 'iframe', 'nav', 'b']):
        element_to_remove.decompose()
    for toc_title in article_content.find_all(lambda tag: tag.name == 'h3' and 'mục lục' in tag.get_text(strip=True).lower()):
        toc_div = toc_title.find_next_sibling('div', class_='lwptoc_i')
        if toc_div: toc_div.decompose()
        toc_title.decompose()
    
    # "Làm phẳng" HTML: Thay thế H2, H3 bằng một ký tự phân tách đặc biệt
    # Dùng ký tự đặc biệt khó trùng lặp
    SEPARATOR = "§SECTION_BREAK§"
    for h_tag in article_content.find_all(['h2', 'h3']):
        h_tag.replace_with(f"{SEPARATOR}{h_tag.get_text(strip=True)}{SEPARATOR}")
        
    # Lấy toàn bộ văn bản đã được "làm phẳng"
    full_text = article_content.get_text(separator='\n')
    
    # Tách văn bản thành các chunks dựa trên ký tự phân tách
    sections = full_text.split(SEPARATOR)
    
    structured_content = []
    # Bỏ qua phần đầu tiên nếu nó rỗng (thường là nội dung trước tiêu đề đầu tiên)
    for section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue
        
        # Tách tiêu đề và nội dung
        lines = section_text.split('\n')
        section_title = lines[0].strip()
        section_content = "\n".join(lines[1:]).strip()
        
        # Chỉ thêm nếu cả tiêu đề và nội dung đều có
        if section_title and section_content:
            structured_content.append({
                "section": section_title,
                "content": clean_text(section_content)
            })
            
    # Thêm phần "Tổng quan" (nội dung trước tiêu đề đầu tiên)
    overview_text = sections[0] if sections else ''
    if overview_text and not structured_content: # Nếu chỉ có 1 chunk lớn
         structured_content.append({"section": "Tổng quan", "content": clean_text(overview_text)})
    
    return disease_name, structured_content

# ==============================================================================
# CÁC HÀM CÒN LẠI GIỮ NGUYÊN
# ==============================================================================
def load_and_process_data():
    all_chunks = []
    print(f"Bắt đầu quét thư mục: '{DATA_SOURCE_DIR}'")
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
                content = f"Thông tin về bệnh {disease_name}, mục '{section_data['section']}': {section_data['content']}"
                doc = Document(page_content=content, metadata={"source": disease_name, "section": section_data['section']})
                all_chunks.append(doc)
        else:
            print(f"  Cảnh báo: Không tìm thấy file '{disease_filename}'")
    
    print(f"\n=> Đã tạo tổng cộng {len(all_chunks)} chunks.")
    return all_chunks

def create_vector_store(chunks):
    print("\nBắt đầu quá trình embedding với mô hình local...")
    start_time = time.time()
    model_kwargs = {'device': 'cpu'}
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
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Đang xóa Vector DB cũ tại: {VECTOR_STORE_PATH}")
        shutil.rmtree(VECTOR_STORE_PATH)

    processed_chunks = load_and_process_data()
    if processed_chunks:
        print("\n" + "="*25 + " KIỂM TRA NỘI DUNG CHUNKS " + "="*25)
        for i, chunk in enumerate(processed_chunks):
            print(f"\n----------- CHUNK #{i+1} -----------")
            print(f"METADATA: {chunk.metadata}")
            print("---------------------------------")
            print(f"NỘI DUNG:\n{chunk.page_content}\n")
        print("="*75)
        create_vector_store(processed_chunks)
    else:
        print("\nKhông có chunk nào được tạo.")