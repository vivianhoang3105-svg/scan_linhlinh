import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import gc

# --- Cấu hình trang ---
st.set_page_config(page_title="Scanner Pro Max", page_icon="📄")

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
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def clean_and_enhance(img_np):
    """Bộ lọc làm nét chữ, sạch nền và khử hạt"""
    # 1. Khử nhiễu (Denoising) - Giúp ảnh hết bị hạt
    dst = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
    
    # 2. Chuyển sang PIL để tăng độ tương phản (Contrast)
    pil_img = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.8) # Tăng tương phản để chữ đen hơn, nền trắng hơn
    
    # 3. Làm nét (Sharpness)
    sharpness = ImageEnhance.Sharpness(pil_img)
    pil_img = sharpness.enhance(2.0)
    
    return pil_img

def process_image(uploaded_file):
    # Đọc ảnh với độ phân giải gốc
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Tìm viền
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # Cắt và nắn thẳng
    if screenCnt is not None:
        warped = perspective_transform(image, screenCnt.reshape(4, 2))
    else:
        warped = image

    # TỰ ĐỘNG XOAY: Nếu chiều ngang > chiều dọc -> Xoay đứng lại
    h, w = warped.shape[:2]
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # Làm đẹp ảnh (Khử nhiễu + Tăng nét)
    final_img = clean_and_enhance(warped)
    
    del image, gray, blurred, edged, warped
    gc.collect()
    return final_img

# --- Giao diện Streamlit ---
st.title("🚀 Scan Linh Linh Pro Max")
st.write("Chụp ảnh nét, tự động nắn đứng, gom 1 file PDF duy nhất.")

uploaded_files = st.file_uploader("Tải ảnh lên (không giới hạn số lượng)", 
                                  type=['jpg', 'jpeg', 'png'], 
                                  accept_multiple_files=True)

if uploaded_files:
    if st.button("Bắt đầu xử lý"):
        progress = st.progress(0)
        all_scanned = []
        
        for i, file in enumerate(uploaded_files):
            with st.spinner(f"Đang xử lý trang {i+1}..."):
                scanned_img = process_image(file)
                all_scanned.append(scanned_img)
                progress.progress((i + 1) / len(uploaded_files))

        if all_scanned:
            # Lưu PDF với chất lượng cao (DPI 300)
            pdf_path = io.BytesIO()
            all_scanned[0].save(
                pdf_path, 
                format="PDF", 
                save_all=True, 
                append_images=all_scanned[1:], 
                resolution=300.0, # Giữ độ phân giải cao
                quality=95
            )
            
            st.success("Đã hoàn thành!")
            st.download_button(
                label="📥 Tải PDF Scan siêu nét",
                data=pdf_path.getvalue(),
                file_name="LinhLinh_Scan.pdf",
                mime="application/pdf"
            )
   
            
