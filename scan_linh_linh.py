import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import gc
from streamlit_cropper import st_cropper

# --- CẤU HÌNH ---
st.set_page_config(page_title="Linh Linh Smart Scanner", page_icon="📑", layout="wide")

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
    """Giữ nguyên bộ lọc sửa màu trắng đen bà ưng ý"""
    if mode == "Quét Trắng Đen (B&W)":
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
        return Image.fromarray(thresh).convert('RGB')
    else:
        img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.4)
        return pil_img

def auto_process(file_bytes):
    """Hàm tự động quét và nắn thẳng không cần hỏi"""
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
    
    # Tự động xoay đứng
    h, w = warped.shape[:2]
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

# --- GIAO DIỆN ---
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>📄 Scanner Linh Linh - Tự động & Thông minh</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📤 Thêm ảnh vào đây bà ơi", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    final_pages = []
    
    st.subheader("📸 Xem trước & Chỉnh sửa (Nếu cần)")

    for i, file in enumerate(uploaded_files):
        with st.expander(f"Trang {i+1}: {file.name}", expanded=(i == 0)):
            
            # Ô tích chọn: Có muốn cắt tay không?
            use_manual_crop = st.checkbox("Tự cắt vùng ảnh (Chỉ dùng khi máy quét sai)", key=f"check_{i}")
            
            col_img, col_ctrl = st.columns([2, 1])
            
            with col_ctrl:
                color_mode = st.radio(f"Màu sắc", ["Quét Trắng Đen (B&W)", "Giữ màu gốc"], key=f"m_{i}")
                rotate_extra = st.selectbox(f"Xoay thêm", ["Không", "90°", "180°", "270°"], key=f"r_{i}")

            if use_manual_crop:
                # Nếu bà muốn cắt tay
                st.write("📍 *Bà kéo khung để chọn vùng nhé:*")
                pil_raw = Image.open(file)
                cropped_img = st_cropper(pil_raw, realtime_update=True, box_color='#FF0000', aspect_ratio=None, key=f"crop_{i}")
                # Chuyển sang OpenCV để áp dụng filter
                img_cv = cv2.cvtColor(np.array(cropped_img.convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                # MẶC ĐỊNH: Tự động hoàn toàn
                img_cv = auto_process(file.getvalue())

            # Áp dụng màu và xoay
            final_pil = apply_filters(img_cv, color_mode)
            if rotate_extra == "90°": final_pil = final_pil.rotate(-90, expand=True)
            elif rotate_extra == "180°": final_pil = final_pil.rotate(180)
            elif rotate_extra == "270°": final_pil = final_pil.rotate(90, expand=True)

            with col_img:
                st.image(final_pil, use_column_width=True, caption="Kết quả trang này")
            
            final_pages.append(final_pil)

    # --- NÚT GOM PDF ---
    st.write("---")
    if st.button("🚀 XUẤT FILE PDF NGAY", use_container_width=True, type="primary"):
        if final_pages:
            pdf_io = io.BytesIO()
            final_pages[0].save(pdf_io, format="PDF", save_all=True, append_images=final_pages[1:], resolution=300.0)
            st.balloons()
            st.download_button(label="📥 TẢI FILE PDF VỀ MÁY", data=pdf_io.getvalue(), file_name="Scanner_LinhLinh.pdf", mime="application/pdf", use_container_width=True)

gc.collect()
