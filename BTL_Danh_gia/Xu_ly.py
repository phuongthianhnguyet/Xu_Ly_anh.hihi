# BTL Xử lý ảnh — Đánh giá chất lượng ảnh sau xử lý
# 1 ảnh gốc | 4 thuật toán | Đánh giá MSE + PSNR

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# TẠO THƯ MỤC LƯU KẾT QUẢ

os.makedirs("results", exist_ok=True)

# PHẦN 1: ĐỌC ẢNH GỐC

img_goc = cv2.imread("images/anh_goc.jpg")

if img_goc is None:
    print("❌ Không tìm thấy ảnh! Hãy đặt ảnh vào: images/anh_goc.jpg")
    exit()

print(f"✅ Đọc ảnh thành công: {img_goc.shape[1]} x {img_goc.shape[0]} px")

# PHẦN 2: TẠO ẢNH NHIỄU GAUSSIAN

def them_nhieu_gaussian(img, std=25):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)

anh_nhieu = them_nhieu_gaussian(img_goc)

cv2.imwrite("results/anh_co_nhieu.png", anh_nhieu)

# PHẦN 3: HÀM TÍNH MSE & PSNR

def tinh_mse(img1, img2):
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return np.mean((a - b) ** 2)

def tinh_psnr(img1, img2):
    mse = tinh_mse(img1, img2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((255 ** 2) / mse)

# PHẦN 4: ÁP DỤNG CÁC THUẬT TOÁN

# Thuật toán 1: Gaussian Blur
kq1 = cv2.GaussianBlur(anh_nhieu, (5, 5), 0)

# Thuật toán 2: Median Filter
kq2 = cv2.medianBlur(anh_nhieu, 5)

# Thuật toán 3: High-pass Filter (lọc thông cao)

kernel_highpass = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

kq3 = cv2.filter2D(anh_nhieu, -1, kernel_highpass)

# Thuật toán 4: Mean Filter (Average Blur)

kq4 = cv2.blur(anh_nhieu, (5, 5))

# LƯU ẢNH KẾT QUẢ

cv2.imwrite("results/kq_gaussian.png", kq1)
cv2.imwrite("results/kq_median.png", kq2)
cv2.imwrite("results/kq_highpass.png", kq3)
cv2.imwrite("results/kq_mean.png", kq4)

print("✅ Đã xử lý xong 4 thuật toán")

# PHẦN 5: TÍNH CHỈ SỐ ĐÁNH GIÁ

mse_nhieu = tinh_mse(img_goc, anh_nhieu)
psnr_nhieu = tinh_psnr(img_goc, anh_nhieu)

mse1 = tinh_mse(img_goc, kq1)
psnr1 = tinh_psnr(img_goc, kq1)

mse2 = tinh_mse(img_goc, kq2)
psnr2 = tinh_psnr(img_goc, kq2)

mse3 = tinh_mse(img_goc, kq3)
psnr3 = tinh_psnr(img_goc, kq3)

mse4 = tinh_mse(img_goc, kq4)
psnr4 = tinh_psnr(img_goc, kq4)

print("\n==============================")
print("So sánh chất lượng ảnh")
print("==============================")

print(f"Ảnh nhiễu      : MSE={mse_nhieu:.2f}  PSNR={psnr_nhieu:.2f}")
print(f"Gaussian Blur  : MSE={mse1:.2f}  PSNR={psnr1:.2f}")
print(f"Median Filter  : MSE={mse2:.2f}  PSNR={psnr2:.2f}")
print(f"High-pass      : MSE={mse3:.2f}  PSNR={psnr3:.2f}")
print(f"Mean Filter    : MSE={mse4:.2f}  PSNR={psnr4:.2f}")

# PHẦN 6: BIỂU ĐỒ SO SÁNH

ten = [
    "Ảnh nhiễu",
    "Gaussian",
    "Median",
    "High-pass",
    "Mean"
]

mse_values = [mse_nhieu, mse1, mse2, mse3, mse4]
psnr_values = [psnr_nhieu, psnr1, psnr2, psnr3, psnr4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

ax1.bar(ten, mse_values)
ax1.set_title("So sánh MSE")

ax2.bar(ten, psnr_values)
ax2.set_title("So sánh PSNR")

plt.tight_layout()
plt.savefig("results/bieu_do_so_sanh.png")
plt.show()

# PHẦN 7: HIỂN THỊ ẢNH

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imgs = [img_goc, anh_nhieu, kq1, kq2, kq3, kq4]

titles = [
    "Ảnh gốc",
    "Ảnh nhiễu",
    "Gaussian",
    "Median",
    "High-pass",
    "Mean"
]

fig, axes = plt.subplots(1, 6, figsize=(20,5))

for ax, im, title in zip(axes, imgs, titles):
    ax.imshow(bgr2rgb(im))
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/so_sanh_truc_quan.png")
plt.show()

print("\n🎉 Hoàn thành! Kết quả nằm trong thư mục results")