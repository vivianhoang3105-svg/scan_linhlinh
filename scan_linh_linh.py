import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import gc
from streamlit_cropper import st_cropper # Nhập thư viện cắt ảnh xịn

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Scanner Linh Linh Pro Max", page_icon="📑", layout="wide")

# --- HÀM XỬ LÝ LÕI ---

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

def apply_filters(warped_img, mode):
    """Sửa màu ảnh: Trắng đen siêu nét vs Giữ màu"""
    if mode == "Quét Trắng Đen (B&W)":
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
        return Image.fromarray(thresh).convert('RGB')
    else: # Giữ màu
        img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        return pil_img

def auto_flatten(file_bytes):
    """Bước quét và nắn thẳng TỰ ĐỘNG"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
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

    warped = perspective_transform(image, screenCnt.reshape(4, 2)) if screenCnt is not None else image
    return warped

# --- GIAO DIỆN CHÍNH ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📑 Scanner Linh Linh Super Pro</h1>", unsafe_allow_html=True)
st.write("---")

uploaded_files = st.file_uploader("📤 Chọn ảnh tài liệu của bà Linh", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    final_pages = []
    
    st.subheader("🛠️ Cắt, Xoay & Duyệt từng trang")

    # Bà kiểm tra từng tấm một trong cái "thẻ" Expander này
    for i, file in enumerate(uploaded_files):
        with st.expander(f"Trang {i+1} - {file.name}", expanded=(i == 0)):
            # Tải ảnh gốc bằng PIL (streamlit-cropper yêu cầu)
            pil_original = Image.open(file)
            
            st.write("👉 **Bước 1: Cắt ảnh (Crop).** Bà dùng chuột kéo cái khung trắng để chọn vùng tờ giấy nhé.")
            
            # CÔNG CỤ CẮT ẢNH XỊN
            # Nó sẽ hiện ra cái khung cho bà kéo thả. Ảnh bà cắt xong sẽ lưu vào `pil_cropped`.
            pil_cropped = st_cropper(pil_original, realtime_update=True, box_color='#FF0000', aspect_ratio=None, key=f"crop_{i}")
            
            # Giải phóng RAM ảnh gốc
            del pil_original 

            col_img, col_ctrl = st.columns([2, 1])
            
            with col_ctrl:
                st.write("👉 **Bước 2: Chỉnh sửa nâng cao.**")
                
                flatten_mode = st.radio(f"Làm phẳng tờ giấy (Trang {i+1})", 
                                       ["Tự động nắn thẳng (OpenCV)", "Cắt sao giữ vậy"], 
                                       key=f"flat_{i}")
                
                color_mode = st.radio(f"Chọn màu (Trang {i+1})", 
                                     ["Quét Trắng Đen (B&W)", "Giữ màu gốc (Color)"], 
                                     key=f"mode_{i}")
                
                rotate_extra = st.selectbox(f"Xoay lại hướng nếu máy đoán sai (Trang {i+1})", 
                                           ["Không xoay", "90°", "180°", "270°"], 
                                           key=f"rot_{i}")

            # 1. Chuyển ảnh đã cắt (PIL) sang OpenCV để làm phẳng (nếu bà chọn)
            img_to_process = np.array(pil_cropped.convert('RGB')) # PIL -> Numpy RGB
            img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_RGB2BGR) # RGB -> BGR (OpenCV)

            # 2. Làm phẳng / Nắn thẳng
            if flatten_mode == "Tự động nắn thẳng (OpenCV)":
                # Chuyển ngược lại file bytes tạm thời (OpenCV nắn thẳng từ file bytes)
                _, buffer = cv2.imencode('.jpg', img_to_process)
                bytes_temp = buffer.tobytes()
                warped_raw = auto_flatten(bytes_temp)
                
                # Tự động xoay đứng nếu máy phát hiện ảnh nằm ngang
                h, w = warped_raw.shape[:2]
                if w > h:
                    warped_raw = cv2.rotate(warped_raw, cv2.ROTATE_90_CLOCKWISE)
            else:
                # Nếu "Cắt sao giữ vậy", app sẽ giữ nguyên góc bà cắt, ko nắn thẳng.
                warped_raw = img_to_process

            # 3. Áp dụng filter màu (Dựa trên ảnh đã nắn thẳng hoặc ảnh cắt)
            final_pil = apply_filters(warped_raw, color_mode)
            
            # 4. Áp dụng xoay thủ công nếu bà chọn
            if rotate_extra == "90°": final_pil = final_pil.rotate(-90, expand=True)
            elif rotate_extra == "180°": final_pil = final_pil.rotate(180)
            elif rotate_extra == "270°": final_pil = final_pil.rotate(90, expand=True)

            with col_img:
                st.image(final_pil, use_column_width=True, caption=f"Ảnh chốt Trang {i+1}")
            
            final_pages.append(final_pil)
            del pil_cropped, img_to_process, warped_raw
            gc.collect()

    # --- XUẤT PDF ---
    st.write("---")
    if st.button("🚀 GOM TẤT CẢ VÀ TẠO PDF PRO MAX", use_container_width=True, type="primary"):
        if final_pages:
            with st.spinner("Đang đóng gói..."):
                pdf_io = io.BytesIO()
                # Resolution 300 DPI cho siêu nét
                final_pages[0].save(pdf_io, format="PDF", save_all=True, append_images=final_pages[1:], resolution=300.0, quality=95)
                
                st.balloons()
                st.download_button(
                    label="📥 TẢI FILE PDF CỦA BÀ TẠI ĐÂY",
                    data=pdf_io.getvalue(),
                    file_name="LinhLinh_Pro_Document.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                
