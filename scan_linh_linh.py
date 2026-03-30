import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import gc
from streamlit_cropper import st_cropper
from datetime import datetime

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Scanner VIP Linh Linh", page_icon="👑", layout="wide")

# --- CÁC HÀM XỬ LÝ ẢNH ---

def order_points(pts):
    """Xác định chính xác 4 góc để nắn thẳng"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    """Kéo phẳng tờ giấy, đảm bảo vuông vức ngay ngắn"""
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

def auto_scan_logic(file_bytes):
    """Tự động tìm tờ giấy và nắn thẳng đứng"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (image.shape[0] * image.shape[1] / 15):
            screenCnt = approx
            break

    if screenCnt is not None:
        warped = perspective_transform(image, screenCnt.reshape(4, 2))
    else:
        warped = image

    # Ép buộc xoay đứng (Portrait) để luôn ngay ngắn
    h, w = warped.shape[:2]
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def apply_filters_and_brightness(warped_img, mode, brightness_val):
    """Áp dụng chỉnh sáng và bộ lọc màu"""
    # 1. Chỉnh sáng trước bằng PIL
    img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    if brightness_val != 1.0:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_val)

    # 2. Áp dụng bộ lọc
    if mode == "Quét Trắng Đen (B&W)":
        # Chuyển lại sang OpenCV để xử lý B&W
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
        return Image.fromarray(thresh).convert('RGB')
    else:
        # Tăng nhẹ độ tương phản cho ảnh màu
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
        return pil_img

# --- GIAO DIỆN STREAMLIT ---
st.markdown("<h1 style='text-align: center; color: #E91E63;'>👑 Máy Quét VIP Linh Linh</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Bản nâng cấp: Nắn thẳng tuyệt đối & Tùy chỉnh độ sáng</p>", unsafe_allow_html=True)
st.write("---")

uploaded_files = st.file_uploader("📤 Kéo thả ảnh của bà vào đây", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    final_pages = []
    st.subheader("🛠️ Căn chỉnh & Lên màu từng trang")

    for i, file in enumerate(uploaded_files):
        with st.expander(f"Trang {i+1}: {file.name}", expanded=(i == 0)):
            
            manual_mode = st.checkbox("Cắt thủ công bằng tay (Nếu máy tự quét sai)", key=f"manual_{i}")
            
            col_img, col_ctrl = st.columns([2, 1.5])
            
            with col_ctrl:
                st.write("🎨 **Bảng Điều Khiển:**")
                color_mode = st.radio(f"Màu sắc", ["Quét Trắng Đen (B&W)", "Giữ màu gốc"], key=f"color_{i}")
                
                # Thanh kéo độ sáng thần thánh (0.5 là tối, 2.0 là cực sáng, 1.0 là gốc)
                brightness = st.slider(f"☀️ Độ sáng", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key=f"bright_{i}")
                
                rotate_extra = st.selectbox(f"🔄 Xoay cho ngay ngắn", ["Chuẩn rồi", "Xoay 90°", "Xoay 180°", "Xoay 270°"], key=f"rot_{i}")

            if manual_mode:
                st.write("📍 *Kéo 4 góc khung đỏ để lấy đúng tờ giấy nhé:*")
                pil_raw = Image.open(file)
                cropped_pil = st_cropper(pil_raw, realtime_update=True, box_color='#E91E63', aspect_ratio=None, key=f"cropper_{i}")
                img_cv = cv2.cvtColor(np.array(cropped_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                img_cv = auto_scan_logic(file.getvalue())

            # Chốt bộ lọc, độ sáng và góc xoay
            final_pil = apply_filters_and_brightness(img_cv, color_mode, brightness)
            
            if rotate_extra == "Xoay 90°": final_pil = final_pil.rotate(-90, expand=True)
            elif rotate_extra == "Xoay 180°": final_pil = final_pil.rotate(180)
            elif rotate_extra == "Xoay 270°": final_pil = final_pil.rotate(90, expand=True)

            with col_img:
                st.image(final_pil, use_column_width=True, caption=f"Ảnh chốt Trang {i+1}")
            
            final_pages.append(final_pil)

    st.write("---")
    if st.button("🚀 XUẤT FILE PDF SIÊU NÉT", use_container_width=True, type="primary"):
        if final_pages:
            with st.spinner("Máy đang đóng gói, chờ xíu nha..."):
                now = datetime.now().strftime("%d-%m_%Hh%M")
                file_name_custom = f"VIP_Scan_{now}.pdf"

                pdf_io = io.BytesIO()
                final_pages[0].save(
                    pdf_io, 
                    format="PDF", 
                    save_all=True, 
                    append_images=final_pages[1:], 
                    resolution=300.0, 
                    quality=95
                )
                
                st.balloons()
                st.success(f"Xong xuất sắc! File của bà đây: {file_name_custom}")
                st.download_button(
                    label="📥 TẢI XUỐNG NGAY", 
                    data=pdf_io.getvalue(), 
                    file_name=file_name_custom, 
                    mime="application/pdf", 
                    use_container_width=True
                )

gc.collect()
