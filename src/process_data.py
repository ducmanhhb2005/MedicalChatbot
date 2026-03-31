import os
import re
import time
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# THAY ĐỔI 1: Bỏ RecursiveCharacterTextSplitter và thêm SemanticChunker
from langchain_experimental.text_splitter import SemanticChunker

# ==============================================================================
# I. CẤU HÌNH & ĐƯỜNG DẪN
# ==============================================================================
# CHUNK_SIZE và CHUNK_OVERLAP không còn tác dụng khi dùng SemanticChunker
# CHUNK_SIZE = 512
# CHUNK_OVERLAP = 100

# Lấy đường dẫn tuyệt đối của thư mục chứa script hiện tại (src/process_data.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Đi ngược lên 1 cấp để đến thư mục gốc của dự án
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Corpus")
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "faiss_index_medical_semantic") # Đổi tên để không ghi đè file cũ
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# ==============================================================================
# II. DỮ LIỆU ĐẦU VÀO (Không đổi)
# ==============================================================================
TARGET_DISEASES = [
    'benh-ho-van-tim.html', 'benh-lao-phoi.html', 'khan-tieng.html', 'nhoi-mau-co-tim-khong-st-chenh-len.html',
    'nhoi-mau-co-tim-that-phai.html', 'nhoi-mau-nao.html', 'suy-gian-tinh-mach-chi-duoi.html',
    'suy-gian-tinh-mach-sau-chi-duoi.html', 'suy-ho-hap.html', 'suy-tim-giai-doan-cuoi.html',
    'suy-tim-man-tinh.html', 'suy-tim-mat-bu.html', 'suy-tim-phai.html', 'suy-tim-sung-huyet.html',
    'suy-tim-trai.html', 'suy-tim.html', 'suy-tinh-mach-man-tinh.html', 'thieu-mau-co-tim-cuc-bo-man-tinh.html',
    'thieu-mau-co-tim.html', 'tim-dap-nhanh.html', 'ung-thu-phoi-khong-te-bao-nho-giai-doan-1.html',
    'ung-thu-phoi-khong-te-bao-nho-giai-doan-2.html', 'ung-thu-phoi-khong-te-bao-nho-giai-doan-3.html',
    'ung-thu-phoi.html', 'ung-thu-thanh-quan.html', 'ung-thu-thuc-quan.html', 'ung-thu-vom-hong-giai-doan-0.html',
    'ung-thu-vom-hong-giai-doan-1.html', 'ung-thu-vom-hong-giai-doan-2.html', 'ung-thu-vom-hong-giai-doan-3.html',
    'ung-thu-vom-hong-giai-doan-dau.html', 'ung-thu-vom-hong.html', 'viem-amidan-hoc-mu.html', 'viem-amidan-man-tinh.html',
    'viem-amidan.html', 'viem-phe-quan-phoi.html', 'viem-phoi-do-metapneumovirus.html', 'viem-phoi.html',
    'viem-thanh-quan.html', 'xo-phoi.html', 'xo-vua-dong-mach-vanh.html', 'xo-vua-dong-mach.html',
]


def clean_text(text):
    """Hàm này không đổi, vẫn rất hữu dụng."""
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_content_from_html(filepath):
    """HÀM NÀY CỦA CHỊ VẪN GIỮ NGUYÊN 100% - KHÔNG THAY ĐỔI GÌ CẢ."""
    CUTOFF_STRING = "HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH"
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
            for tag in soup.find_all(['strong', 'em', 'b']):
                tag.unwrap()
    except Exception as e:
        print(f"  Lỗi khi đọc file {filepath}: {e}")
        return None, []

    disease_name_tag = soup.find('h1')
    if not disease_name_tag:
        return "Không rõ tên bệnh", []
    disease_name = disease_name_tag.get_text(strip=True).split(':')[0].strip()

    structured_content = []
    # Xử lý phần Tổng quan (nội dung giữa H1 và H2 đầu tiên)
    current_node = disease_name_tag.find_next_sibling()
    overview_content = []
    while current_node and current_node.name != 'h2':
        if isinstance(current_node, Tag) and 'mục lục' in current_node.get_text(strip=True).lower():
            break
        text = current_node.get_text(separator=' ', strip=True)
        if text:
            overview_content.append(text)
        current_node = current_node.find_next_sibling()
    
    if overview_content:
        structured_content.append({
            'section': 'Tổng quan',
            'content': clean_text('\n'.join(overview_content))
        })
        
    # Xử lý các section H2
    h2_tags = soup.find_all('h2')
    for h2 in h2_tags:
        section_title = h2.get_text(strip=True)
        section_content = []
        for sibling in h2.find_next_siblings():
            if sibling.name == 'h2' or (isinstance(sibling, Tag) and CUTOFF_STRING in sibling.get_text()):
                break
            text = sibling.get_text(separator=' ', strip=True)
            if text:
                section_content.append(text)
        
        content_text = clean_text('\n'.join(section_content).strip())
        if content_text:
            structured_content.append({
                'section': section_title,
                'content': content_text
            })

    return disease_name, structured_content

def load_and_process_data():
    """
    HÀM NÀY ĐƯỢC NÂNG CẤP LÕI CHUNKING
    """
    # Giai đoạn 1: Trích xuất cấu trúc từ HTML (giữ nguyên code của chị)
    all_docs = []
    print(f"Bắt đầu Giai đoạn 1: Trích xuất cấu trúc từ HTML...")
    for disease_filename in TARGET_DISEASES:
        filepath = os.path.join(DATA_SOURCE_DIR, disease_filename)
        if os.path.exists(filepath):
            print(f"- Đang xử lý file: {disease_filename}")
            disease_name, sections = extract_content_from_html(filepath)
            if not sections: continue
            for section_data in sections:
                if section_data.get('content') and section_data.get('section'):
                    doc = Document(
                        page_content=section_data['content'],
                        metadata={"source": disease_name, "section": section_data['section']}
                    )
                    all_docs.append(doc)

    print(f"=> Đã trích xuất được {len(all_docs)} sections lớn.")

    # THAY ĐỔI 2: Nâng cấp lõi chunking tại đây
    # --------------------------------------------------------------------------
    print(f"\nBắt đầu Giai đoạn 2: Chia nhỏ ngữ nghĩa bằng Semantic Chunker...")
    
    # [CÚ PHÁP TỔNG QUÁT]
    # 1. Khởi tạo model embedding mà chunker sẽ dùng để "hiểu" văn bản.
    # 2. Khởi tạo SemanticChunker với model embedding đó.
    # 3. Áp dụng chunker lên các document đã có.

    # [ÁP DỤNG CHO DỰ ÁN]
    # 1. Khởi tạo model embedding tiếng Việt
    print("- Đang tải model embedding cho chunker...")
    model_kwargs = {'device': 'cuda'}  # Dùng GPU cho nhanh nha chị
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs
    )
    
    # 2. Khởi tạo Semantic Chunker
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="standard_deviation"
    )

    # 3. Tiến hành chia nhỏ các document lớn đã trích xuất ở Giai đoạn 1
    print("- Bắt đầu quá trình chia nhỏ theo ngữ nghĩa...")
    all_chunks = text_splitter.split_documents(all_docs)
    # --------------------------------------------------------------------------

    print(f"\n=> ĐÃ HOÀN TẤT: Tạo ra tổng cộng {len(all_chunks)} chunks chất lượng cao.")
    return all_chunks

def create_vector_store(chunks):
    """Hàm này gần như không đổi, chỉ cần đảm bảo device là 'cuda' cho nhất quán."""
    if not chunks:
        print("\nKhông có chunk nào được tạo.")
        return

    print("\nBắt đầu quá trình embedding và lưu trữ VectorDB...")
    start_time = time.time()
    
    model_kwargs = {'device': 'cuda'} # Chạy trên GPU cho nhanh chị nhé
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
    # THAY ĐỔI 3: Cần cài thêm thư viện mới
    print("Lưu ý: Chị nhớ chạy 'pip install langchain_experimental' nếu chưa có nhé!")
    
    processed_chunks = load_and_process_data()
    if processed_chunks:
        create_vector_store(processed_chunks)
    else:
        print("\nKhông có chunk nào được tạo. Vui lòng kiểm tra lại cấu trúc thư mục và code.")