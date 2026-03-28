import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ... (Giữ nguyên các hàm order_points, perspective_transform, scan_process) ...
# Lưu ý: Sửa hàm scan_process một chút để nhận vào object file thay vì đường dẫn

def scan_process_streamlit(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        warped = perspective_transform(orig, screenCnt.reshape(4, 2))
    else:
        warped = orig

    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)

# Giao diện Streamlit thay cho Tkinter
st.title("📄 App Quét Ảnh Sang PDF")

uploaded_files = st.file_uploader("Chọn các ảnh muốn scan", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if st.button("Bắt đầu Scan và Tạo PDF"):
        scanned_images = []
        for uploaded_file in uploaded_files:
            img = scan_process_streamlit(uploaded_file)
            scanned_images.append(img)
        
        if scanned_images:
            # Lưu PDF vào bộ nhớ đệm để người dùng tải về
            pdf_buffer = io.BytesIO()
            scanned_images[0].save(pdf_buffer, format="PDF", save_all=True, append_images=scanned_images[1:])
            
            st.success("Đã xử lý xong!")
            st.download_button(
                label="📥 Tải file PDF về máy",
                data=pdf_buffer.getvalue(),
                file_name="scanned_document.pdf",
                mime="application/pdf"
            )
