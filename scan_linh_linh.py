import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import gc
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_cropper import st_cropper
from datetime import datetime

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Scanner VIP Thống Nhất", page_icon="👑", layout="wide")

# --- CÁC HÀM XỬ LÝ LÕI ---

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

def apply_auto_rotate_stand(image_cv):
    h, w = image_cv.shape[:2]
    if w > h:
        return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    return image_cv

def auto_scan_logic(file_bytes):
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

    return apply_auto_rotate_stand(warped)

def apply_filters_and_brightness(warped_img_cv, mode, brightness_val):
    img_rgb = cv2.cvtColor(warped_img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    if brightness_val != 1.0:
        pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_val)

    if mode == "Quét Trắng Đen (B&W)":
        pil_img = pil_img.convert('L')
        pil_img = ImageEnhance.Contrast(pil_img).enhance(2.2)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
        return pil_img.convert('RGB')
    else:
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        return pil_img

# --- GIAO DIỆN CHÍNH ---
st.markdown("<h1 style='text-align: center; color: #E91E63;'>👑 Scanner VIP Linh Linh</h1>", unsafe_allow_html=True)

device_type = st.radio(
    "👉 **Chọn thiết bị cắt ảnh thủ công:**", 
    ["💻 Máy tính (Click chuột 4 góc)", "📱 Điện thoại (Vuốt kéo khung)"], 
    horizontal=True
)
st.write("---")

uploaded_files = st.file_uploader("📤 Thêm ảnh vào đây", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    final_pages = []
    
    for i, file in enumerate(uploaded_files):
        with st.expander(f"Trang {i+1}: {file.name}", expanded=(i == 0)):
            
            file_key = f"pts_{file.name}_{file.size}"
            if file_key not in st.session_state:
                st.session_state[file_key] = []
            
            col_img, col_ctrl = st.columns([2, 1])
            
            with col_ctrl:
                st.write("🔧 **Căn chỉnh:**")
                manual_mode = st.checkbox(f"📍 Sửa cắt thủ công", key=f"manual_{i}")
                color_mode = st.radio(f"Chọn màu", ["Quét Trắng Đen (B&W)", "Giữ màu gốc"], key=f"color_{i}")
                brightness = st.slider(f"☀️ Độ sáng", 0.5, 2.0, 1.0, 0.1, key=f"bright_{i}")
                rotate_extra = st.selectbox(f"🔄 Xoay thêm", ["Chuẩn rồi", "Xoay 90°", "Xoay 180°", "Xoay 270°"], key=f"rot_{i}")

            with col_img:
                if manual_mode:
                    if "Điện thoại" in device_type:
                        st.info("📲 Đang dùng công cụ Điện thoại: Kéo 4 góc khung đỏ để chọn vùng ảnh.")
                        pil_original = Image.open(file)
                        
                        # Bản xịn: should_resize_box=True cho phép kéo giãn thoải mái
                        cropped_pil = st_cropper(
                            pil_original, 
                            realtime_update=True, 
                            box_color='#FF0000', 
                            aspect_ratio=None,
                            should_resize_box=True, 
                            key=f"cropper_{i}"
                        )
                        
                        img_cv_base = cv2.cvtColor(np.array(cropped_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                        img_cv_final = apply_auto_rotate_stand(img_cv_base)
                        
                    else:
                        st.info("💻 Đang dùng công cụ Máy tính: Click chuột vào 4 góc để ghim.")
                        pil_original = Image.open(file)
                        cv_original = cv2.cvtColor(np.array(pil_original.convert('RGB')), cv2.COLOR_RGB2BGR)
                        
                        for pt in st.session_state[file_key]:
                            cv2.circle(cv_original, pt, max(15, int(cv_original.shape[1]/50)), (0, 0, 255), -1)
                        pil_to_show = Image.fromarray(cv2.cvtColor(cv_original, cv2.COLOR_BGR2RGB))
                        
                        value = streamlit_image_coordinates(pil_to_show, key=f"coord_{i}")
                        
                        if value is not None:
                            clicked_pt = (value['x'], value['y'])
                            if clicked_pt not in st.session_state[file_key] and len(st.session_state[file_key]) < 4:
                                st.session_state[file_key].append(clicked_pt)
                                st.rerun()

                        if st.button("🔄 Xóa điểm để chấm lại", key=f"reset_{i}"):
                            st.session_state[file_key] = []
                            st.rerun()

                        if len(st.session_state[file_key]) == 4:
                            pts = np.array(st.session_state[file_key], dtype="float32")
                            cv_original_real = cv2.cvtColor(np.array(Image.open(file).convert('RGB')), cv2.COLOR_RGB2BGR)
                            warped_manual = perspective_transform(cv_original_real, pts)
                            img_cv_final = apply_auto_rotate_stand(warped_manual)
                            st.success("Đã khóa 4 góc!")
                        else:
                            st.warning(f"Bà đã chấm {len(st.session_state[file_key])}/4 góc.")
                            img_cv_final = apply_auto_rotate_stand(cv_original)
                else:
                    img_cv_final = auto_scan_logic(file.getvalue())

                final_pil = apply_filters_and_brightness(img_cv_final, color_mode, brightness)
                
                if rotate_extra == "Xoay 90°": final_pil = final_pil.rotate(-90, expand=True)
                elif rotate_extra == "Xoay 180°": final_pil = final_pil.rotate(180)
                elif rotate_extra == "Xoay 270°": final_pil = final_pil.rotate(90, expand=True)

                st.write("**Kết quả trang này:**")
                st.image(final_pil, use_column_width=True)
                final_pages.append(final_pil)

    st.write("---")
    if st.button("🚀 GOM TẤT CẢ VÀ TẠO FILE PDF", use_container_width=True, type="primary"):
        if final_pages:
            with st.spinner("Đang đóng gói PDF Pro..."):
                now = datetime.now().strftime("%d-%m_%Hh%M")
                file_name_custom = f"Scanner_Pro_{now}.pdf"

                pdf_io = io.BytesIO()
                final_pages[0].save(pdf_io, format="PDF", save_all=True, append_images=final_pages[1:], resolution=300.0, quality=95)
                
                st.balloons()
                st.success(f"Hoàn tất! Tên file: {file_name_custom}")
                st.download_button(label="📥 TẢI PDF VỀ MÁY", data=pdf_io.getvalue(), file_name=file_name_custom, mime="application/pdf", use_container_width=True)

    gc.collect()
