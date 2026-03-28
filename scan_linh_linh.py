import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import os

def order_points(pts):
    """Sắp xếp 4 tọa độ theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    """Biến đổi phối cảnh để làm phẳng tờ giấy"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính toán chiều rộng mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Tính toán chiều cao mới
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

def scan_process(image_path):
    """Quy trình nhận diện và cắt ảnh"""
    image = cv2.imread(image_path)
    orig = image.copy()
    
    # Tiền xử lý: Chuyển xám, làm mờ và tìm cạnh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Tìm các đường viền
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4: # Tìm thấy hình có 4 cạnh (tờ giấy)
            screenCnt = approx
            break

    if screenCnt is not None:
        # Nếu tìm thấy viền, thực hiện nắn thẳng
        warped = perspective_transform(orig, screenCnt.reshape(4, 2))
    else:
        # Nếu không tìm thấy, giữ nguyên ảnh gốc nhưng chuyển hệ màu
        warped = orig

    # Chuyển sang định dạng PIL để lưu PDF
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)

def main():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(title="Chọn ảnh scan", 
                                            filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_paths: return

    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", 
                                             filetypes=[("PDF", "*.pdf")])
    if not save_path: return

    scanned_images = []
    for path in file_paths:
        try:
            pilled_img = scan_process(path)
            scanned_images.append(pilled_img)
        except Exception as e:
            print(f"Lỗi xử lý file {path}: {e}")

    if scanned_images:
        scanned_images[0].save(save_path, save_all=True, append_images=scanned_images[1:])
        messagebox.showinfo("Xong!", f"Đã lưu tại: {save_path}")

if __name__ == "__main__":
    main()
