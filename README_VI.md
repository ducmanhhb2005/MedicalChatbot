# 🏥 Cẩm Nang Y Khoa - Medical Chatbot

Một hệ thống chatbot y khoa thông minh chạy hoàn toàn trên máy tính cục bộ, sử dụng công nghệ Retrieval-Augmented Generation (RAG) để trả lời các câu hỏi về bệnh tim mạch với độ chính xác cao.

## ✨ Tính Năng Chính

- **🔐 Hoàn toàn cục bộ**: Không cần kết nối internet, dữ liệu y tế được xử lý trên máy của bạn
- **🧠 AI thông minh**: Sử dụng Vinallama 7B LLM (Large Language Model) để trả lời câu hỏi tiếng Việt
- **📚 Cơ sở dữ liệu vector**: FAISS indexing để tìm kiếm thông tin bệnh học nhanh chóng
- **🎯 Chuyên biệt**: Tập trung vào các bệnh về tim mạch với dữ liệu y tế được xác thực
- **🌐 Giao diện web**: Streamlit UI thân thiện, dễ sử dụng
- **💬 Hỗ trợ tiếng Việt**: Xử lý tiếng Việt tự nhiên (NLP) với mô hình nhúng tối ưu

## 📋 Yêu Cầu Hệ Thống

### Cấu Hình Tối Thiểu
- **CPU**: Intel/AMD 64-bit (tốt nhất là i5 trở lên)
- **RAM**: 8GB (tối thiểu), 16GB (khuyến nghị)
- **Ổ cứng**: 20GB dung lượng trống
- **Python**: Version 3.8 - 3.11

### Cấu Hình Khuyến Nghị
- **CPU**: CPU mạnh (i7/Ryzen 7 trở lên)
- **RAM**: 16GB+
- **GPU**: NVIDIA CUDA (tùy chọn, để tăng tốc độ)
- **Ổ cứng**: SSD 30GB+

## 🚀 Cài Đặt & Chạy

### 1. Clone/Copy Project
```bash
# Tải project vào máy
git clone <repository-url>
cd Medical_Chatbot
```

### 2. Tạo Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

⏱️ **Lưu ý**: Lần đầu cài đặt có thể mất 10-15 phút vì cần tải các model lớn (LLM ~3.4GB, embedding model ~500MB)

### 4. Chạy Ứng Dụng
```bash
python -m streamlit run .\src\app.py
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

## 📁 Cấu Trúc Dự Án

```
Medical_Chatbot/
├── src/
│   ├── app.py                      # Giao diện Streamlit chính
│   ├── core_logic.py               # Logic RAG pipeline
│   ├── core_logic_2.py             # Logic thay thế (R&D)
│   └── process_data.py             # Xử lý dữ liệu
├── data/
│   ├── raw/
│   │   ├── Corpus/                 # Dữ liệu bệnh HTML thô
│   │   └── Corpus_Redone/          # Dữ liệu bệnh đã xử lý
│   └── processed/
│       ├── medical_data.json       # Dữ liệu JSON
│       ├── faiss_index_medical/    # Vector store (indexing)
│       └── faiss_index_medical_semantic/  # Vector store (semantic)
├── models/
│   ├── vinallama-7b-chat-Q3_K_M.gguf    # LLM (~3.4GB)
│   ├── vi-gemma-2b-rag-q4_k_s.gguf      # LLM thay thế (~2GB)
│   └── llama-2-7b-chat.Q3_K_M.gguf      # LLM thay thế (~3.8GB)
├── vietnamese-bi-encoder/          # Embedding model
├── scripts/
│   └── convert_html_to_json.py     # Convert HTML sang JSON
├── evaluation/
│   ├── evaluate.py                 # Script đánh giá
│   └── evaluation_dataset.csv      # Tập đánh giá
├── check_chunks.py                 # Debug chunks
├── Modelfile                       # Ollama config
├── requirements.txt                # Dependencies
└── README.md                        # Documentation
```

## 🔧 Công Nghệ Sử Dụng

| Thành Phần | Công Nghệ | Lý Do Chọn |
|-----------|-----------|-----------|
| **LLM** | Vinallama 7B (GGUF) | Tối ưu tiếng Việt, chạy CPU nhanh |
| **Embedding** | Vietnamese Bi-Encoder | Nhúng tiếng Việt chính xác |
| **Vector DB** | FAISS | Tìm kiếm vector cực nhanh |
| **RAG Framework** | LangChain (v1.2.13) | Framework RAG mạnh mẽ |
| **UI** | Streamlit | Deploy web nhanh, thân thiện |
| **NLP** | BeautifulSoup4, LXML | Parse HTML, xử lý dữ liệu |

## 📊 Cách Hoạt Động

### Quy Trình RAG (Retrieval-Augmented Generation)

```
Câu hỏi của người dùng
        ↓
   Embedding (Vector hóa)
        ↓
   Tìm kiếm FAISS → Lấy 4 chunks liên quan nhất
        ↓
   Format ngữ cảnh
        ↓
   LLM xử lý (Prompt + Context)
        ↓
   Trả lời tự nhiên bằng tiếng Việt
```

### Ví Dụ
**Người dùng hỏi**: "Tôi thấy hơi khó thở ở ngực. Tôi có bị sao không?"

**Hệ thống**:
1. Vector hóa câu hỏi
2. Tìm kiếm trong FAISS indexing → Tìm bệnh liên quan (khó thở, đau ngực)
3. Lấy thông tin về: suy tim, nhồi máu cơ tim, viêm cơ tim...
4. LLM tạo câu trả lời từ thông tin y tế + ngữ cảnh

**Trả lời**: "Khó thở kèm đau ngực có thể là dấu hiệu của các bệnh tim mạch nghiêm trọng như..."

## ⚙️ Cấu Hình & Tuning

### Thay đổi Model LLM
Chỉnh sửa trong `src/core_logic.py`:
```python
MODEL_GGUF_PATH = os.path.join(PROJECT_ROOT, "models", "vinallama-7b-chat-Q3_K_M.gguf")
```

**Các model khác có thể dùng**:
- `vi-gemma-2b-rag-q4_k_s.gguf` (nhỏ hơn, nhanh hơn)
- `llama-2-7b-chat.Q3_K_M.gguf` (khác)

### Tuning Số Lượng Chunks
Trong `src/core_logic.py`:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # Thay 4 thành số khác
```
- `k=3`: Tìm kiếm 3 chunk (nhanh, nhưng ít context)
- `k=4`: Mặc định (cân bằng)
- `k=5`: Find 5 chunks (chậm hơn, nhiều context)

### Tuning Temperature (LLM)
```python
llm = LlamaCpp(
    model_path=MODEL_GGUF_PATH,
    temperature=0.1  # 0.0-1.0. Cao hơn = sáng tạo hơn
)
```

## 🐛 Troubleshooting

### ❌ Lỗi: `ModuleNotFoundError: No module named 'transformers'`
**Giải pháp**:
```bash
pip install transformers
```

### ❌ Lỗi: `Could not import llama-cpp-python`
**Giải pháp**:
```bash
pip install llama-cpp-python
```

### ❌ Ứng dụng chạy chậm
**Nguyên nhân**: Model đang load, hoặc máy không đủ RAM
**Giải pháp**:
- Nâng cấp RAM
- Đóng các ứng dụng khác
- Dùng model nhỏ hơn (vi-gemma-2b)

### ❌ Lỗi: `FileNotFoundError: Could not find faiss index`
**Giải pháp**: Đảm bảo đường dẫn `data/processed/faiss_index_medical/` tồn tại

### ⚠️ Warnings về `torchvision`
**Nguyên nhân**: `transformers` cố load modules image processing (không cần dùng)
**Giải pháp**: Bỏ qua unsafe, app vẫn chạy bình thường. Nếu muốn tắt:
```bash
pip install torchvision
```

## 📦 Dependency Versions

Dự án đã test với các phiên bản sau:
```
Python 3.10.x
streamlit==1.55.0
langchain==1.2.13
langchain-community==0.4.1
langchain-core==1.2.23
pydantic==2.12.5
faiss-cpu==1.13.2
llama-cpp-python==0.3.19
sentence-transformers==5.3.0
```

## 🎯 Kế Hoạch Phát Triển

- [ ] Hỗ trợ multi-language (tiếng Anh, Hàn...)
- [ ] Thêm các chuyên khoa y tế khác (phổi, thần kinh...)
- [ ] Integration với Electronic Health Records (EHR)
- [ ] Mobile app version
- [ ] Real-time knowledge update
- [ ] User feedback loop để cải thiện model

## 📝 License

MIT License - Tự do sử dụng, sửa đổi, phân phối

## 👥 Đóng Góp

Chào mừng các đóng góp! Vui lòng:
1. Fork project
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## ⚠️ Công Pháp Pháp Lý & Trách Nhiệm

**Lưu ý quan trọng**: 
- Chatbot này là **phương tiện tham khảo duy nhất** và **KHÔNG thay thế** lời khuyên y tế từ bác sĩ
- Không được xem là chẩn đoán y tế chính thức
- Người dùng phải tự chịu trách nhiệm về quyết định y tế của mình
- Luôn tham vấn bác sĩ chuyên khoa trước khi quyết định điều trị

## 📧 Liên Hệ & Support

- **Issues**: Report bug tại GitHub Issues
- **Discussion**: Thảo luận tại GitHub Discussions
- **Email**: [your-email@example.com]

## 🙏 Cảm Ơn

- Data từ [Nguồn dữ liệu y tế]
- Base model: Vinallama team
- Framework: LangChain & Streamlit communities

---

**Phiên bản**: 1.0.0  
**Cập nhật lần cuối**: 31/03/2026  
**Trạng thái**: Production Ready ✅
