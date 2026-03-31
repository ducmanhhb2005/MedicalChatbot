# -*- coding: utf-8 -*-
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, '..')
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "faiss_index_medical")


# VECTOR_STORE_PATH = "data/processed/faiss_index_medical"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
LOCAL_LLM_MODEL = "vinallama:7b-chat-local"
GGUF_MODEL_PATH = f"src/models/vinallama-7b-chat_q5_0.gguf"  # 🔄 Đường dẫn file gguf

@st.cache_resource
def load_rag_pipeline():
    """Tải Vector DB, Embedding model và LLM local. Hàm này được cache để tăng tốc."""
    print("Đang tải pipeline RAG...")
    # Tải embedding model
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Tải Vector DB
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Khởi tạo LLM local
    llm = LlamaCpp(
        model_path=GGUF_MODEL_PATH,
        n_ctx=4096,
        temperature=0.3,
        max_tokens=1024,
        verbose=True,
        n_threads=3,  # Điều chỉnh theo máy bạn
        n_gpu_layers=0  # 0 nếu không dùng GPU
    )
    
    # Tạo Prompt Template
    prompt_template = """
    Bạn là một trợ lý y khoa hữu ích. Chỉ sử dụng những thông tin được cung cấp trong phần ngữ cảnh dưới đây để trả lời câu hỏi.
    Tuyệt đối không tự bịa đặt thông tin. Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rằng bạn không có đủ thông tin để trả lời.
    Luôn nhắc nhở người dùng rằng thông tin chỉ để tham khảo và họ nên tham vấn ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác.

    Ngữ cảnh:
    {context}

    Câu hỏi:
    {question}

    Câu trả lời bằng tiếng Việt:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    
    # Tạo chuỗi RAG bằng LCEL
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = (
        {"context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
         "question": RunnablePassthrough()}
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    print("Pipeline RAG đã sẵn sàng.")
    return qa_chain

def get_rag_response(qa_chain, query: str):
    """Nhận query và trả về kết quả từ RAG chain."""
    try:
        result = qa_chain.invoke({"query": query})
        return result
    except Exception as e:
        print(f"Lỗi khi thực thi RAG chain: {e}")
        return {"result": "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại.", "source_documents": []}