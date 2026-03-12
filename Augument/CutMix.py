"""
CutMix Augmentation - Visualization Demo
=========================================
Cắt một vùng hình chữ nhật ngẫu nhiên từ ảnh B và dán vào ảnh A.

Cách dùng:
    1. Sửa IMG_PATH_1 và IMG_PATH_2 bên dưới thành đường dẫn ảnh bạn muốn.
    2. Chạy: python Augument/CutMix.py
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===================== CẤU HÌNH ĐƯỜNG DẪN ẢNH =====================
IMG_PATH_1 = r"d:\IT\Projects\Scale_Error_DC\data\train\MyImage-20250723140003680_jpg.rf.d69126d88ab8801e044c3c92316f89ac.jpg"
IMG_PATH_2 = r"d:\IT\Projects\Scale_Error_DC\data\train\MyImage-20250723140203700_jpg.rf.20cfeff53b7e10b02cfb6595b233ec32.jpg"
BETA = 1.0  # Tham số Beta distribution (càng nhỏ -> vùng cắt càng lớn/nhỏ cực đoan)
# ====================================================================


def cutmix(img1: Image.Image, img2: Image.Image, beta: float = 1.0):
    """
    CutMix: Cắt một vùng hình chữ nhật từ img2 và dán vào img1.

    Returns:
        result_img: Ảnh kết quả (PIL Image)
        lam: Tỉ lệ diện tích giữ lại từ img1
        bbox: (x1, y1, x2, y2) vùng bị cắt
    """
    arr1 = np.array(img1).copy()
    arr2 = np.array(img2)

    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size, Image.BILINEAR)
        arr2 = np.array(img2)

    # Lambda ~ Beta(beta, beta)
    lam = np.random.beta(beta, beta)
    cut_ratio = np.sqrt(1.0 - lam)

    h, w = arr1.shape[:2]
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    # Vị trí trung tâm ngẫu nhiên
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    # Tọa độ vùng cắt
    x1 = int(np.clip(cx - cut_w // 2, 0, w))
    y1 = int(np.clip(cy - cut_h // 2, 0, h))
    x2 = int(np.clip(cx + cut_w // 2, 0, w))
    y2 = int(np.clip(cy + cut_h // 2, 0, h))

    # Dán vùng từ ảnh B vào ảnh A
    arr1[y1:y2, x1:x2] = arr2[y1:y2, x1:x2]

    # Tính lại lambda thực tế (dựa trên diện tích thực sự được cắt)
    actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)

    return Image.fromarray(arr1), actual_lam, (x1, y1, x2, y2)


if __name__ == "__main__":
    print("=" * 60)
    print("  CutMix Augmentation Demo")
    print("=" * 60)

    img1 = Image.open(IMG_PATH_1).convert("RGB")
    img2 = Image.open(IMG_PATH_2).convert("RGB")

    result, lam, (x1, y1, x2, y2) = cutmix(img1, img2, beta=BETA)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ảnh gốc A
    axes[0].imshow(img1)
    axes[0].set_title("Image A (Gốc)", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    # Ảnh gốc B
    axes[1].imshow(img2)
    axes[1].set_title("Image B (Nguồn cắt)", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    # Kết quả CutMix
    axes[2].imshow(result)
    # Vẽ viền đỏ quanh vùng bị cắt
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=3, edgecolor='red', facecolor='none',
                              linestyle='--')
    axes[2].add_patch(rect)
    axes[2].set_title(f"CutMix Result (λ={lam:.2f})", fontsize=14, fontweight='bold', color='red')
    axes[2].axis("off")

    fig.suptitle("CutMix: Cắt vùng chữ nhật từ B dán vào A", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r"d:\IT\Projects\Scale_Error_DC\Augument\cutmix_demo.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Lambda (tỉ lệ giữ lại từ A): {lam:.4f}")
    print(f"  Vùng cắt (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")
    print(f"  Kích thước vùng cắt: {x2-x1} x {y2-y1} pixels")
    print(f"\n  Đã lưu ảnh demo: Augument/cutmix_demo.png")