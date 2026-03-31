# -*- coding: utf-8 -*-
import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "faiss_index_medical")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "vietnamese-bi-encoder")
MODEL_GGUF_PATH = os.path.join(PROJECT_ROOT, "models", "vinallama-7b-chat-Q3_K_M.gguf") # 3.4 GB
@st.cache_resource
def load_rag_pipeline():
    print(f"Đang tải RAG pipeline với LLM: Vinallama 7B (Q3)...")
    
    # Chạy Embedding trên CPU để giải phóng VRAM cho LLM
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs=model_kwargs)
    
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = LlamaCpp(
        model_path=MODEL_GGUF_PATH,
        n_gpu_layers=-1, # Dành toàn bộ GPU cho LLM
        n_ctx=3072,      # Cho nó đủ không gian ngữ cảnh
        verbose=False,
        temperature=0.1 
    )

    
    prompt_template = """Bạn là một robot trích xuất thông tin y khoa. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách trích xuất và kết hợp các câu chữ có sẵn trong ngữ cảnh.

QUY TẮC TUYỆT ĐỐI:
1.  ĐỌC KỸ CÂU HỎI.
2.  TÌM CÁC CÂU TRẢ LỜI TRỰC TIẾP TRONG NGỮ CẢNH.
3.  TỔNG HỢP LẠI CÁC CÂU ĐÓ MÀ KHÔNG THAY ĐỔI Ý NGHĨA.
4.  NẾU NGỮ CẢNH KHÔNG CÓ THÔNG TIN, TRẢ LỜI: "Tôi không tìm thấy thông tin cụ thể về vấn đề này trong tài liệu."
5.  KHÔNG SUY DIỄN. KHÔNG BÌNH LUẬN. KHÔNG THÊM BẤT KỲ THÔNG TIN NÀO TỪ BÊN NGOÀI.
6.  LUÔN KẾT THÚC BẰNG CÂU SAU TRÊN DÒNG RIÊNG: "Lưu ý: Thông tin này chỉ mang tính tham khảo, vui lòng tham vấn ý kiến bác sĩ."

Ngữ cảnh:
{context}
Câu hỏi:
{question}
Câu trả lời được trích xuất:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    
    # Sử dụng LCEL để xây dựng chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Format context từ documents
    def format_docs(docs):
        if isinstance(docs, list):
            return "\n\n".join([doc.page_content for doc in docs])
        return str(docs)
    
    # Chain input sẽ là string (question/query)
    qa_chain = (
        {
            "context": (lambda x: x) | retriever | RunnableLambda(format_docs),
            "question": lambda x: x
        }
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    
    print("RAG pipeline đã sẵn sàng.")
    return qa_chain
def get_rag_response(qa_chain, query: str):
    """
    Nhận một câu hỏi, thực thi RAG chain và trả về một string kết quả.
    """
    try:
        # qa_chain.invoke nhận query string trực tiếp
        result = qa_chain.invoke(query)
        
        # Đảm bảo kết quả là string
        if isinstance(result, dict):
            result = result.get("result", str(result))
        
        return str(result)
    except Exception as e:
        print(f"Lỗi khi thực thi RAG chain: {e}")
        import traceback
        traceback.print_exc()
        # Trả về một error message
        return f"Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý: {str(e)}"