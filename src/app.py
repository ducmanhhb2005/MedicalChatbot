import streamlit as st
from core_logic import load_rag_pipeline, get_rag_response

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Cẩm Nang Y Khoa Local", layout="wide")
st.title("⚕️ Cẩm Nang Y Khoa (Local & Privacy-focused)")
st.info("Chào mừng bạn! Hệ thống này chạy hoàn toàn trên máy tính của bạn, đảm bảo an toàn và bảo mật. Hãy đặt câu hỏi về các bệnh tim mạch.")

# Tải pipeline RAG (chỉ tải 1 lần nhờ caching)
try:
    qa_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"Không thể khởi tạo hệ thống. Vui lòng kiểm tra lại. Lỗi: {e}")
    st.stop()

# Khởi tạo session state để lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
if prompt := st.chat_input("Hỏi tôi về triệu chứng, nguyên nhân của bệnh tim..."):
    # Hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lấy câu trả lời từ bot và hiển thị
    with st.chat_message("assistant"):
        with st.spinner("Bác sĩ AI đang suy nghĩ..."):
            result = get_rag_response(qa_chain, prompt)
            response = result.get("result", "Không có câu trả lời.")
            st.markdown(response)
            
            # Hiển thị nguồn tham khảo
            source_docs = result.get("source_documents", [])
            if source_docs:
                with st.expander("Xem nguồn tham khảo"):
                    for doc in source_docs:
                        st.info(f"**Nguồn:** Bệnh {doc.metadata.get('source')} - Mục: {doc.metadata.get('section')}")

    st.session_state.messages.append({"role": "assistant", "content": response})