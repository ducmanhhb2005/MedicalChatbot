import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import sys
import os
import time

# Thêm thư mục src vào path để import được core_logic
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core_logic import load_rag_pipeline, get_rag_response

def run_evaluation():
    print("--- BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ HỆ THỐNG RAG ---")
    
    # 1. Tải pipeline RAG
    try:
        qa_chain = load_rag_pipeline()
    except Exception as e:
        print(f"Lỗi khi tải pipeline: {e}")
        return

    # 2. Tải bộ dữ liệu đánh giá
    eval_df = pd.read_csv("evaluation/evaluation_dataset.csv")
    print(f"Đã tải {len(eval_df)} câu hỏi từ bộ dữ liệu đánh giá.")

    # 3. Chạy RAG trên từng câu hỏi để thu thập kết quả
    results = []
    total_latency = 0
    print("\nĐang chạy RAG trên bộ dữ liệu...")
    for index, row in eval_df.iterrows():
        start_time = time.time()
        result = get_rag_response(qa_chain, row["question"])
        end_time = time.time()
        
        latency = end_time - start_time
        total_latency += latency
        
        results.append({
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "answer": result.get("result"),
            "contexts": [doc.page_content for doc in result.get("source_documents", [])]
        })
        print(f"  - Hoàn thành câu hỏi {index + 1}/{len(eval_df)} trong {latency:.2f}s")
        
    avg_latency = total_latency / len(eval_df)
    print(f"\nThời gian phản hồi trung bình: {avg_latency:.2f} giây/câu hỏi.")

    # 4. Chuẩn bị dataset cho RAGAs
    eval_results_df = pd.DataFrame(results)
    ragas_dataset = Dataset.from_pandas(eval_results_df)

    # 5. Thực hiện đánh giá với RAGAs
    print("\nĐang tính toán các chỉ số RAGAs...")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    # Lưu ý: RAGAs có thể cần OpenAI để đánh giá, đây là một điểm cần nêu trong báo cáo.
    # Để chạy hoàn toàn local, cần cấu hình RAGAs với LLM và Embedding local, phức tạp hơn.
    # Tạm thời, ta có thể dùng OpenAI cho bước đánh giá này.
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Cần OPENAI_API_KEY trong file .env để chạy RAGAs evaluation.")

        result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
        )
        print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
        print(result)

    except Exception as e:
        print(f"\nLỗi khi chạy RAGAs: {e}")
        print("Mẹo: Đảm bảo bạn có file .env với OPENAI_API_KEY hợp lệ để RAGAs hoạt động mặc định.")
        print("Việc cấu hình RAGAs chạy local hoàn toàn là một bước nâng cao.")
    
    print("\n--- KẾT QUẢ HIỆU NĂNG ---")
    print(f"Thời gian phản hồi trung bình: {avg_latency:.2f} giây")
    # Các thông số khác như RAM, Storage có thể cần đo thủ công và ghi lại.

if __name__ == '__main__':
    run_evaluation()