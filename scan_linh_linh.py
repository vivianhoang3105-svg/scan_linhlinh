import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import gc

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
    """Chế độ sửa màu ảnh theo ý bà"""
    if mode == "Quét Trắng Đen (B&W)":
        # Bộ lọc Adaptive Threshold thần thánh cho ảnh chụp màn hình
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
        return Image.fromarray(thresh).convert('RGB')
    
    else: # Chế độ Giữ màu
        img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        # Tăng tương phản và độ nét
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        return pil_img

@st.cache_data(show_spinner=False)
def initial_scan(file_bytes):
    """Bước quét và nắn thẳng ban đầu"""
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
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📑 Scanner Linh Linh Pro Max</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📤 Chọn ảnh tài liệu (Bao nhiêu cũng được)", 
                                  type=['jpg', 'jpeg', 'png'], 
                                  accept_multiple_files=True)

if uploaded_files:
    final_pages = []
    
    st.write("---")
    st.subheader("🛠️ Chỉnh sửa & Duyệt từng trang")

    for i, file in enumerate(uploaded_files):
        with st.expander(f"Trang {i+1}: {file.name}", expanded=(i == 0)):
            col_img, col_ctrl = st.columns([2, 1])
            
            # 1. Quét ban đầu (nắn thẳng + xoay)
            warped_raw = initial_scan(file.getvalue())
            
            with col_ctrl:
                st.write("🎨 **Chế độ sửa màu:**")
                color_mode = st.radio(f"Chọn màu (Trang {i+1})", 
                                     ["Quét Trắng Đen (B&W)", "Giữ màu gốc (Color)"], 
                                     key=f"mode_{i}")
                
                st.write("🔄 **Xoay thủ công:**")
                rotate_extra = st.selectbox(f"Xoay thêm (Trang {i+1})", 
                                           ["Không xoay", "90°", "180°", "270°"], 
                                           key=f"rot_{i}")

            # 2. Áp dụng filter màu bà chọn
            final_pil = apply_filters(warped_raw, color_mode)
            
            # 3. Áp dụng xoay thủ công nếu cần
            if rotate_extra == "90°": final_pil = final_pil.rotate(-90, expand=True)
            elif rotate_extra == "180°": final_pil = final_pil.rotate(180)
            elif rotate_extra == "270°": final_pil = final_pil.rotate(90, expand=True)

            with col_img:
                st.image(final_pil, use_column_width=True, caption=f"Xem trước Trang {i+1}")
            
            final_pages.append(final_pil)

    # --- XUẤT PDF ---
    st.write("---")
    if st.button("🚀 GOM TẤT CẢ VÀ TẠO PDF", use_container_width=True, type="primary"):
        if final_pages:
            with st.spinner("Đang gom file..."):
                pdf_io = io.BytesIO()
                final_pages[0].save(pdf_io, format="PDF", save_all=True, append_images=final_pages[1:], resolution=300.0)
                
                st.balloons()
                st.download_button(
                    label="📥 TẢI FILE PDF CỦA LINH TẠI ĐÂY",
                    data=pdf_io.getvalue(),
                    file_name="LinhLinh_Pro_Scan.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
    # Giải phóng RAM
    gc.collect()

st.write("---")
st.caption("Ứng dụng dành riêng cho Linh - Chụp màn hình cũng không sợ mờ! 😉")
   
    
          
