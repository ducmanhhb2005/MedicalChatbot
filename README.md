# 🏥 Medical Chatbot - Cẩm Nang Y Khoa

An intelligent medical chatbot system running entirely locally on your personal computer, using Retrieval-Augmented Generation (RAG) technology to answer questions about cardiovascular diseases with high accuracy.

Link demo: https://youtu.be/pw3MoaDh6gA?si=MEVO6N2AQXqod3O4

## ✨ Key Features

- **🔐 Completely Local**: No internet required, medical data processed on your machine
- **🧠 Intelligent AI**: Vietnamese-optimized Vinallama 7B LLM (Large Language Model)
- **📚 Vector Database**: FAISS indexing for fast medical information retrieval
- **🎯 Specialized**: Focused on cardiovascular diseases with verified medical data
- **🌐 Web Interface**: User-friendly Streamlit UI
- **💬 Vietnamese Support**: Natural Language Processing with optimized embedding models

## 📋 System Requirements

### Minimum Configuration
- **CPU**: 64-bit Intel/AMD
- **RAM**: 8GB (minimum), 16GB (recommended)
- **Storage**: 20GB free space
- **Python**: Version 3.8 - 3.11

### Recommended Configuration
- **CPU**: High-performance (i7/Ryzen 7+)
- **RAM**: 16GB+
- **GPU**: NVIDIA CUDA (optional for acceleration)
- **Storage**: 30GB+ SSD

## 🚀 Installation & Usage

### 1. Clone/Download Project
```bash
git clone <repository-url>
cd Medical_Chatbot
```

### 2. Create Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

⏱️ **Note**: Initial installation may take 10-15 minutes due to large model files (~4GB total)

### 4. Run Application
```bash
python -m streamlit run .\src\app.py
```

Application opens at: **http://localhost:8501**

## 📁 Project Structure

```
Medical_Chatbot/
├── src/
│   ├── app.py                      # Main Streamlit interface
│   ├── core_logic.py               # RAG pipeline implementation
│   ├── core_logic_2.py             # Alternative logic (R&D)
│   └── process_data.py             # Data processing
├── data/
│   ├── raw/
│   │   ├── Corpus/                 # Raw disease HTML data
│   │   └── Corpus_Redone/          # Processed disease data
│   └── processed/
│       ├── medical_data.json       # JSON format data
│       ├── faiss_index_medical/    # Vector store index
│       └── faiss_index_medical_semantic/  # Semantic index
├── models/
│   ├── vinallama-7b-chat-Q3_K_M.gguf    # Main LLM (~3.4GB)
│   ├── vi-gemma-2b-rag-q4_k_s.gguf      # Alternative LLM (~2GB)
│   └── llama-2-7b-chat.Q3_K_M.gguf      # Another alternative (~3.8GB)
├── vietnamese-bi-encoder/          # Embedding model
├── scripts/
│   └── convert_html_to_json.py     # HTML to JSON converter
├── evaluation/
│   ├── evaluate.py                 # Evaluation script
│   └── evaluation_dataset.csv      # Evaluation dataset
├── requirements.txt                # Dependencies
└── README.md                        # This file
```

## 🔧 Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| **LLM** | Vinallama 7B (GGUF) | Vietnamese-optimized, CPU-efficient |
| **Embedding** | Vietnamese Bi-Encoder | Accurate Vietnamese embeddings |
| **Vector DB** | FAISS | Extremely fast vector search |
| **RAG Framework** | LangChain (v1.2.13) | Robust RAG implementation |
| **UI** | Streamlit | Quick web deployment, user-friendly |
| **NLP** | BeautifulSoup4, LXML | HTML parsing, data processing |

## 📊 How It Works

### RAG (Retrieval-Augmented Generation) Process

```
User Question
    ↓
Vectorization
    ↓
FAISS Search → Retrieve top 4 relevant chunks
    ↓
Format Context
    ↓
LLM Processing (Prompt + Context)
    ↓
Natural Vietnamese Answer
```

### Example
**User asks**: "I feel a bit breathless in my chest. Is something wrong?"

**System**:
1. Vectorizes the question
2. Searches FAISS index → Finds related diseases (shortness of breath, chest pain)
3. Retrieves info: heart failure, myocardial infarction, myocarditis...
4. LLM generates answer from medical info + context

**Response**: "Shortness of breath with chest pain could indicate serious cardiovascular conditions like..."

## ⚙️ Configuration & Tuning

### Change LLM Model
Edit `src/core_logic.py`:
```python
MODEL_GGUF_PATH = os.path.join(PROJECT_ROOT, "models", "vinallama-7b-chat-Q3_K_M.gguf")
```

**Alternative models**:
- `vi-gemma-2b-rag-q4_k_s.gguf` (smaller, faster)
- `llama-2-7b-chat.Q3_K_M.gguf` (different)

### Adjust Number of Retrieved Chunks
In `src/core_logic.py`:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # Change 4 to another value
```
- `k=3`: Faster, less context
- `k=4`: Default (balanced)
- `k=5`: More context, slower

### Adjust LLM Temperature
```python
llm = LlamaCpp(
    model_path=MODEL_GGUF_PATH,
    temperature=0.1  # 0.0-1.0. Higher = more creative
)
```

## 🐛 Troubleshooting

### ❌ Error: `ModuleNotFoundError: No module named 'transformers'`
**Solution**:
```bash
pip install transformers
```

### ❌ Error: `Could not import llama-cpp-python`
**Solution**:
```bash
pip install llama-cpp-python
```

### ❌ Application runs slowly
**Cause**: Model loading or insufficient RAM  
**Solution**:
- Upgrade RAM
- Close other applications
- Use smaller model (vi-gemma-2b)

### ❌ Error: `FileNotFoundError: Could not find faiss index`
**Solution**: Ensure `data/processed/faiss_index_medical/` exists

### ⚠️ Warnings about `torchvision`
**Cause**: `transformers` tries loading image processing modules (not needed)  
**Solution**: Safe to ignore, app works fine. To remove:
```bash
pip install torchvision
```

## 📦 Tested Dependency Versions

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

## 🎯 Development Roadmap

- [ ] Multi-language support (English, Korean...)
- [ ] Add other medical specialties (pulmonology, neurology...)
- [ ] Integration with Electronic Health Records (EHR)
- [ ] Mobile app version
- [ ] Real-time knowledge updates
- [ ] User feedback loop for model improvement

## 📝 License

MIT License - Free to use, modify, distribute

## 👥 Contributing

Contributions welcome! Please:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Legal Disclaimer & Responsibility

**Important Note**:
- This chatbot is a **reference tool only** and **DOES NOT REPLACE** medical advice from licensed doctors
- NOT intended as official medical diagnosis
- Users are responsible for their own medical decisions
- Always consult with healthcare professionals before treatment decisions

## 📧 Contact & Support

- **Issues**: Report bugs on GitHub Issues
- **Discussion**: GitHub Discussions
- **Email**: [your-email@example.com]

## 🙏 Acknowledgments

- Data source: [Medical data sources]
- Base models: Vinallama team
- Frameworks: LangChain & Streamlit communities

---

**Version**: 1.0.0  
**Last Updated**: March 31, 2026  
**Status**: Production Ready ✅

---

**[Vietnamese Version](README_VI.md)**
