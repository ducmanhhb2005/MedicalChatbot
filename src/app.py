# -*- coding: utf-8 -*-
import streamlit as st
from core_logic import load_rag_pipeline, get_rag_response

st.set_page_config(page_title="Cẩm Nang Y Khoa Local", layout="wide")
st.title("⚕️ Cẩm Nang Y Khoa (Hybrid LLM)")
st.info("Chào mừng bạn! Hệ thống này chạy hoàn toàn trên máy tính của bạn, đảm bảo an toàn và bảo mật. Hãy đặt câu hỏi về các bệnh tim mạch")

try:
    # Hàm load bây giờ chỉ tải retriever
    retriever = load_rag_pipeline()
except Exception as e:
    st.error(f"Không thể khởi tạo hệ thống. Lỗi: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi tôi về bệnh tim mạch..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Bác sĩ AI đang suy nghĩ..."):
            # Gọi hàm get_rag_response
            response = get_rag_response(retriever, prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})