# -*- coding: utf-8 -*-
"""
Tên file: convert_html_to_json.py
Mục tiêu: Đọc tất cả các file HTML trong thư mục Corpus, trích xuất nội dung
         có cấu trúc (tên bệnh, các section) và lưu vào một file JSON duy nhất.
Cách chạy: Chạy file này một lần duy nhất từ thư mục gốc của dự án.
Lệnh: python scripts/convert_html_to_json.py
"""
import os
import json
from bs4 import BeautifulSoup, NavigableString

# ==============================================================================
# I. CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
# Lấy đường dẫn thư mục gốc của dự án (giả sử script này nằm trong thư mục scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn đến thư mục chứa các file HTML thô
HTML_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Corpus")

# Đường dẫn tới file JSON đầu ra
JSON_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "medical_data.json")

# Danh sách các file HTML cần xử lý
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

def parse_html_to_structured_data(filepath, filename):
    """
    Hàm này nhận vào một file HTML, phân tích và trả về một dictionary
    có cấu trúc chứa thông tin về bệnh.
    """
    print(f"  -> Đang xử lý file: {filename}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')

        # 1. Trích xuất tên bệnh từ thẻ H1
        h1_tag = soup.find('h1')
        disease_name = h1_tag.get_text(strip=True) if h1_tag else "Không rõ tiêu đề"

        # 2. Tìm vùng chứa nội dung chính để loại bỏ nhiễu (header, footer, sidebar)
        #    Dựa trên file mẫu, nội dung chính nằm trong div có class 'entry-content'
        content_area = soup.find('div', class_='entry-content')
        if not content_area:
            print(f"    [Cảnh báo] Không tìm thấy vùng nội dung ('entry-content') trong file {filename}.")
            return None
        
        # 3. Trích xuất các sections dựa trên thẻ H2
        sections_data = []
        all_h2s = content_area.find_all('h2')

        for i, h2_tag in enumerate(all_h2s):
            section_title = h2_tag.get_text(strip=True)
            content_parts = []
            
            # Lấy tất cả các thẻ nằm giữa H2 này và H2 tiếp theo
            for element in h2_tag.find_next_siblings():
                if element.name == 'h2':
                    break # Dừng lại khi gặp section tiếp theo
                
                # Bỏ qua các chuỗi rỗng hoặc chỉ có khoảng trắng
                if isinstance(element, NavigableString) and not element.strip():
                    continue
                
                # Lấy text và làm sạch
                text = element.get_text(separator='\n', strip=True) # Dùng \n để giữ các đoạn văn
                if text:
                    content_parts.append(text)
            
            # Kết hợp các phần nội dung thành một chuỗi duy nhất và thêm vào danh sách
            if content_parts:
                full_content = "\n".join(content_parts)
                sections_data.append({"title": section_title, "content": full_content})

        # 4. Tạo đối tượng cuối cùng cho bệnh này
        disease_object = {
            "disease_name": disease_name,
            "source_file": filename, # Lưu lại tên file gốc để truy vết
            "sections": sections_data
        }
        return disease_object

    except Exception as e:
        print(f"    [Lỗi] Xảy ra lỗi khi xử lý file {filename}: {e}")
        return None

def main():
    """
    Hàm chính điều khiển toàn bộ quá trình:
    1. Lặp qua danh sách file HTML.
    2. Gọi hàm parse cho từng file.
    3. Gom kết quả và ghi ra file JSON.
    """
    all_diseases_data = []
    print("Bắt đầu quá trình chuyển đổi HTML sang JSON...")
    print(f"Đang quét thư mục: {HTML_SOURCE_DIR}")
    
    for filename in TARGET_DISEASES:
        filepath = os.path.join(HTML_SOURCE_DIR, filename)
        if os.path.exists(filepath):
            structured_data = parse_html_to_structured_data(filepath, filename)
            if structured_data:
                all_diseases_data.append(structured_data)
        else:
            print(f"  [Cảnh báo] Bỏ qua file không tồn tại: {filename}")

    # Ghi toàn bộ dữ liệu đã thu thập được vào file JSON
    print("\nĐang ghi dữ liệu ra file JSON...")
    try:
        with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            # ensure_ascii=False để hiển thị tiếng Việt đúng
            # indent=2 để file JSON dễ đọc hơn
            json.dump(all_diseases_data, f, ensure_ascii=False, indent=2)
        print(f"✅ THÀNH CÔNG! Đã tạo file JSON tại: {JSON_OUTPUT_PATH}")
    except Exception as e:
        print(f"💥 THẤT BẠI! Không thể ghi file JSON. Lỗi: {e}")

if __name__ == '__main__':
    main()