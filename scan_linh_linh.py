import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import gc
from streamlit_image_coordinates import streamlit_image_coordinates # Dùng cho Desktop
from streamlit_cropper import st_cropper # Dùng cho Mobile
from datetime import datetime

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Scanner VIP Unified", page_icon="📑", layout="wide")

# --- CÁC HÀM XỬ LÝ LÕI OpenCV (GIỮ NGUYÊN) ---

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
    """Nắn thẳng ảnh dựa trên 4 góc"""
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
    """Tự động xoay đứng nếu ảnh ngang (Portrait)"""
    h, w = image_cv.shape[:2]
    if w > h:
        return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    return image_cv

def auto_scan_logic(file_bytes):
    """Tự động tìm viền, nắn, và xoay đứng (Portrait)"""
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

    # Nếu máy đoán đúng thì nắn, không thì để nguyên ảnh gốc cho nét
    if screenCnt is not None:
        warped = perspective_transform(image, screenCnt.reshape(4, 2))
    else:
        warped = image

    # Tự động xoay đứng
    return apply_auto_rotate_stand(warped)

def apply_filters_and_brightness(warped_img_cv, mode, brightness_val):
    """Bộ lọc màu và ánh sáng"""
    img_rgb = cv2.cvtColor(warped_img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Áp dụng độ sáng (slider)
    if brightness_val != 1.0:
        enhancer = ImageEnhance.Brightness(pil_img)
        final_pil = enhancer.enhance(brightness_val)
    else:
        final_pil = pil_img

    if mode == "Quét Trắng Đen (B&W)":
        # Cách siêu nét cứu ảnh màn hình
        img_cv_for_bw = cv2.cvtColor(np.array(final_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv_for_bw, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
        return Image.fromarray(thresh).convert('RGB')
    else:
        # Giữ màu nhưng tăng độ tương phản
        return ImageEnhance.Contrast(final_pil).enhance(1.4)

# --- GIAO DIỆN CHÍNH (THỐNG NHẤT MOBILE & DESKTOP) ---
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>📄 VIP Scanner Thống Nhất - Linh Linh</h1>", unsafe_allow_html=True)
st.info("💡 App sẽ tự nắn. Nếu sai, hãy dùng thanh chỉnh từng trang bên dưới.")

uploaded_files = st.file_uploader("📤 Thêm ảnh vào đây bà ơi", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    final_pages = []
    
    # Hiển thị từng trang ảnh trong expander để sửa
    for i, file in enumerate(uploaded_files):
        with st.expander(f"Trang {i+1}: {file.name}", expanded=(i == 0)):
            
            # Khởi tạo trạng thái chấm điểm cho Desktop nếu chưa có
            file_key = f"pts_{file.name}_{file.size}"
            if file_key not in st.session_state:
                st.session_state[file_key] = []
            
            col_img, col_ctrl = st.columns([2, 1])
            
            # --- CỘT ĐIỀU CHỈNH ---
            with col_ctrl:
                st.write("🔧 **Căn chỉnh:**")
                manual_mode = st.checkbox(f"📍 Sửa thủ công (Kéo góc / Chấm điểm)", key=f"manual_{i}")
                
                color_mode = st.radio(f"Chọn màu", ["Quét Trắng Đen (B&W)", "Giữ màu gốc"], key=f"color_{i}")
                
                brightness = st.slider(f"☀️ Độ sáng", 0.5, 2.0, 1.0, 0.1, key=f"bright_{i}")
                
                rotate_extra = st.selectbox(f"🔄 Xoay thêm cho ngay ngắn", ["Chuẩn rồi", "Xoay 90°", "Xoay 180°", "Xoay 270°"], key=f"rot_{i}")

            # --- CỘT XỬ LÝ VÀ HIỂN THỊ ---
            with col_img:
                if manual_mode:
                    # Kiểm tra nền tảng để dùng công cụ phù hợp
                    # Nếu là Mobile, dùng st_cropper (Cắt kéo V1/V2, hỗ trợ vuốt chạm)
                    if st.platform_is_mobile():
                        st.write("📲 **Đang dùng Điện thoại:** Kéo cái khung đỏ để lấy vùng bà muốn nhé (Vuốt bằng ngón tay).")
                        pil_original = Image.open(file)
                        cropped_pil = st_cropper(pil_original, realtime_update=True, box_color='#FF0000', aspect_ratio=None, key=f"cropper_{i}")
                        img_cv_base = cv2.cvtColor(np.array(cropped_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                        # Cắt kéo thì nắn xoay dọc Portrait
                        img_cv_final = apply_auto_rotate_stand(img_cv_base)
                        st.success("Đã khóa vùng cắt!")
                    
                    # Nếu là Máy tính, dùng streamlit_image_coordinates (Chấm điểm V3)
                    else:
                        st.write("💻 **Đang dùng Máy tính:** Bấm chuột vào 4 góc để ghim (Máy tự sắp xếp tọa độ).")
                        pil_original = Image.open(file)
                        cv_original = cv2.cvtColor(np.array(pil_original.convert('RGB')), cv2.COLOR_RGB2BGR)
                        
                        # Vẽ điểm đỏ
                        for pt in st.session_state[file_key]:
                            cv2.circle(cv_original, pt, max(15, int(cv_original.shape[1]/50)), (0, 0, 255), -1)
                        pil_to_show = Image.fromarray(cv2.cvtColor(cv_original, cv2.COLOR_BGR2RGB))
                        
                        # Nhận click
                        value = streamlit_image_coordinates(pil_to_show, key=f"coord_{i}")
                        
                        if value is not None:
                            clicked_pt = (value['x'], value['y'])
                            if clicked_pt not in st.session_state[file_key] and len(st.session_state[file_key]) < 4:
                                st.session_state[file_key].append(clicked_pt)
                                st.rerun() # Refresh hiện điểm ngay

                        if st.button("🔄 Xóa điểm để chấm lại", key=f"reset_{i}"):
                            st.session_state[file_key] = []
                            st.rerun()

                        if len(st.session_state[file_key]) == 4:
                            pts = np.array(st.session_state[file_key], dtype="float32")
                            cv_original_real = cv2.cvtColor(np.array(Image.open(file).convert('RGB')), cv2.COLOR_RGB2BGR)
                            warped_manual = perspective_transform(cv_original_real, pts)
                            
                            # FIX LỖI 2: Thêm logic tự động xoay dọc Portrait sau khi nắn thủ công trên máy tính
                            img_cv_final = apply_auto_rotate_stand(warped_manual)
                            
                            st.success("Đã khóa 4 góc!")
                        else:
                            st.warning(f"Bà đã chấm {len(st.session_state[file_key])}/4 góc. Tiếp tục nào!")
                            img_cv_final = apply_auto_rotate_stand(cv_original) # Tạm hiện gốc xoay dọc
                else:
                    # MẶC ĐỊNH: Tự động hoàn toàn (OpenCV tìm viền, nắn, xoay)
                    img_cv_final = auto_scan_logic(file.getvalue())

                # Chốt bộ lọc, độ sáng và xoay thêm
                final_pil = apply_filters_and_brightness(img_cv_final, color_mode, brightness)
                
                if rotate_extra == "Xoay 90°": final_pil = final_pil.rotate(-90, expand=True)
                elif rotate_extra == "Xoay 180°": final_pil = final_pil.rotate(180)
                elif rotate_extra == "Xoay 270°": final_pil = final_pil.rotate(90, expand=True)

                st.write("**Kết quả trang này:**")
                st.image(final_pil, use_column_width=True)
                final_pages.append(final_pil)

    # --- NÚT TẠO PDF ---
    st.write("---")
    if st.button("🚀 GOM TẤT CẢ VÀ TẠO FILE PDF", use_container_width=True, type="primary"):
        if final_pages:
            with st.spinner("Đang đóng gói PDF Pro..."):
                now = datetime.now().strftime("%d-%m_%Hh%M")
                file_name_custom = f"Scanner_Pro_{now}.pdf"

                pdf_io = io.BytesIO()
                # Resolution 300 DPI cho siêu nét
                final_pages[0].save(pdf_io, format="PDF", save_all=True, append_images=final_pages[1:], resolution=300.0, quality=95)
                
                st.balloons()
                st.success(f"Hoàn tất! Tên file: {file_name_custom}")
                st.download_button(label="📥 TẢI PDF VỀ MÁY", data=pdf_io.getvalue(), file_name=file_name_custom, mime="application/pdf", use_container_width=True)

    gc.collect()
