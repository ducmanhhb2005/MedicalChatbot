# file: check_chunks.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CẤU HÌNH ---
# Đường dẫn này phải khớp với đường dẫn chị đã lưu VectorDB
VECTOR_STORE_PATH = "data/processed/faiss_index_medical_semantic"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
# -----------------

print(f"Đang tải VectorDB từ: {VECTOR_STORE_PATH}")
print("Vui lòng chờ, quá trình này có thể mất một chút thời gian để tải model embedding...")

# 1. Khởi tạo model embedding (phải giống hệt lúc tạo DB)
model_kwargs = {'device': 'cuda'} # Đổi thành 'cpu' nếu chị dùng CPU
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs
)

# 2. Tải lại VectorDB từ ổ cứng
try:
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True # Cần thiết cho phiên bản LangChain mới
    )
    print("✅ Tải VectorDB thành công!")
except Exception as e:
    print(f"💥 Lỗi khi tải VectorDB: {e}")
    print("Vui lòng kiểm tra lại đường dẫn VECTOR_STORE_PATH có chính xác không.")
    exit()

# 3. Lấy toàn bộ dữ liệu từ VectorDB
# VectorDB của FAISS lưu trữ dữ liệu trong một dictionary
# key 'docstore' chứa thông tin của tất cả các chunk
retriever = vector_store.as_retriever(search_kwargs={'k': 1000}) # Lấy tất cả
all_documents = retriever.get_relevant_documents(query="*")


print(f"\nTổng số chunks trong VectorDB: {len(all_documents)}")
print("-" * 50)

# 4. In ra 5 chunks đầu tiên để xem thử
print("🔎 Dưới đây là 5 chunks đầu tiên để xem thử:")
for i, doc in enumerate(all_documents[:5]):
    print(f"\n=============== CHUNK {i+1} ================")
    print(f"NỘI DUNG CHUNK:\n---\n{doc.page_content}\n---")
    print(f"METADATA:\n  - Nguồn (Bệnh): {doc.metadata.get('source')}")
    print(f"  - Mục (Section): {doc.metadata.get('section')}")
    print("=" * 35)