import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import gc # Garbage Collector để giải phóng bộ nhớ

# --- Cấu hình trang ---
st.set_page_config(page_title="App Quét Ảnh Chuyên Nghiệp", page_icon="📄", layout="centered")

# --- Các hàm xử lý ảnh OpenCV (Giữ nguyên từ code trước) ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def apply_scan_filter(img):
    """Tạo hiệu ứng giống máy scan: Trắng đen và tăng độ tương phản"""
    # Chuyển sang ảnh xám
    img = img.convert('L')
    
    # Chuyển sang mảng numpy để OpenCV xử lý (cho nhanh)
    img_np = np.array(img)
    
    # Áp dụng Adaptive Thresholding để làm trắng nền, làm rõ chữ
    # Đây là kỹ thuật CamScanner hay dùng
    thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Chuyển ngược lại PIL Image
    return Image.fromarray(thresh).convert('RGB')

# --- Hàm xử lý ảnh chính (Dùng cho Web) ---
def process_single_image_web(uploaded_file):
    """Xử lý từng ảnh một để tiết kiệm RAM"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig = image.copy()
    
    # Tiền xử lý
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Tìm viền
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

    # Chuyển sang PIL
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(warped_rgb)
    
    # Áp dụng bộ lọc scan để làm đẹp
    scanned_pil_img = apply_scan_filter(pil_img)
    
    # Giải phóng bộ nhớ OpenCV
    del image
    del orig
    del gray
    del edged
    
    return scanned_pil_img

# --- Giao diện Web ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📄 App Quét Ảnh Chuyên Nghiệp</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Chụp ảnh tài liệu, quét viền và ghép thành một file PDF duy nhất.</p>", unsafe_allow_html=True)

# 1. Khu vực tải file
# accept_multiple_files=True cho phép chọn nhiều ảnh
uploaded_files = st.file_uploader("Chọn hoặc Chụp ảnh tài liệu (Chọn nhiều cùng lúc)", 
                                  type=['jpg', 'jpeg', 'png'], 
                                  accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    st.write(f"Đã chọn **{num_files}** ảnh.")

    # Hiển thị xem trước ảnh (giới hạn 3 ảnh đầu để đỡ đơ máy)
    st.write("### Xem trước (3 ảnh đầu):")
    cols = st.columns(3)
    for i, file in enumerate(uploaded_files[:3]):
        with cols[i]:
            st.image(file, use_column_width=True, caption=file.name)

    # 2. Nút bấm xử lý
    if st.button("🚀 Bắt đầu Scan và Gom thành 1 file PDF", key="scan_btn"):
        # Progress bar để người dùng không sốt ruột
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        scanned_pil_images = []
        errors = []

        # Xử lý TUẦN TỰ từng ảnh một
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Đang xử lý ảnh {i+1}/{num_files}: {uploaded_file.name}...")
                
                # Quét và nắn thẳng
                final_pil_img = process_single_image_web(uploaded_file)
                scanned_pil_images.append(final_pil_img)
                
                # Cập nhật progress bar
                progress_bar.progress(int((i + 1) / num_files * 100))
                
                # Ép Python giải phóng bộ nhớ ngay lập tức
                gc.collect() 

            except Exception as e:
                errors.append(f"Lỗi ảnh {uploaded_file.name}: {str(e)}")

        status_text.text("Xử lý hoàn tất!")
        progress_bar.empty()

        # Hiển thị lỗi nếu có
        if errors:
            st.warning("Có một số lỗi xảy ra:")
            for err in errors:
                st.error(err)

        # 3. Gom và cho tải về
        if scanned_pil_images:
            st.write("---")
            st.success("Tất cả ảnh đã được quét thành công!")
            
            # Lưu PDF vào bộ nhớ tạm
            pdf_buffer = io.BytesIO()
            # Bức ảnh đầu tiên làm trang bìa, các bức sau append vào
            scanned_pil_images[0].save(
                pdf_buffer, 
                format="PDF", 
                save_all=True, 
                append_images=scanned_pil_images[1:]
            )
            
            st.info(f"Tổng dung lượng file PDF: **{len(pdf_buffer.getvalue()) / 1024 / 1024:.2f} MB**")
            
            st.download_button(
                label="📥 Tải file PDF của bạn về máy",
                data=pdf_buffer.getvalue(),
                file_name="scanned_document.pdf",
                mime="application/pdf"
            )

# --- Chú ý ở chân trang ---
st.write("---")
st.markdown("<p style='text-align: center; color: gray;'>Chú ý: Do giới hạn của server online, vui lòng không tải lên quá 50 ảnh độ phân giải cao một lúc để tránh app bị đơ.</p>", unsafe_allow_html=True)
