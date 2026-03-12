"""
MixUp Augmentation - Visualization Demo
=========================================
Trộn tuyến tính hai ảnh: result = λ * img1 + (1-λ) * img2

Cách dùng:
    1. Sửa IMG_PATH_1 và IMG_PATH_2 bên dưới thành đường dẫn ảnh bạn muốn.
    2. Chạy: python Augument/MixUp.py
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===================== CẤU HÌNH ĐƯỜNG DẪN ẢNH =====================
IMG_PATH_1 = r"d:\IT\Projects\Scale_Error_DC\data\train\MyImage-20250723140003680_jpg.rf.d69126d88ab8801e044c3c92316f89ac.jpg"
IMG_PATH_2 = r"d:\IT\Projects\Scale_Error_DC\data\train\MyImage-20250723140203700_jpg.rf.20cfeff53b7e10b02cfb6595b233ec32.jpg"
ALPHA = 0.4  # Tham số Beta distribution (càng nhỏ -> lambda gần 0 hoặc 1)
# ====================================================================


def mixup(img1: Image.Image, img2: Image.Image, alpha: float = 0.4):
    """
    MixUp: Trộn tuyến tính hai ảnh.
    result = λ * img1 + (1 - λ) * img2

    Returns:
        result_img: Ảnh kết quả (PIL Image)
        lam: Tỉ lệ trộn (lambda)
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size, Image.BILINEAR)
        arr2 = np.array(img2).astype(np.float32)

    # Lambda ~ Beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha)

    # Trộn tuyến tính
    mixed = lam * arr1 + (1 - lam) * arr2
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)

    return Image.fromarray(mixed), lam


if __name__ == "__main__":
    print("=" * 60)
    print("  MixUp Augmentation Demo")
    print("=" * 60)

    img1 = Image.open(IMG_PATH_1).convert("RGB")
    img2 = Image.open(IMG_PATH_2).convert("RGB")

    result, lam = mixup(img1, img2, alpha=ALPHA)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ảnh gốc A
    axes[0].imshow(img1)
    axes[0].set_title("Image A (Gốc)", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    # Ảnh gốc B
    axes[1].imshow(img2)
    axes[1].set_title("Image B (Trộn)", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    # Kết quả MixUp
    axes[2].imshow(result)
    axes[2].set_title(f"MixUp Result (λ={lam:.2f})", fontsize=14, fontweight='bold', color='blue')
    axes[2].axis("off")

    fig.suptitle(f"MixUp: result = {lam:.2f} × A + {1-lam:.2f} × B", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r"d:\IT\Projects\Scale_Error_DC\Augument\mixup_demo.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Lambda (tỉ lệ của ảnh A): {lam:.4f}")
    print(f"  Công thức: result = {lam:.2f} × A + {1-lam:.2f} × B")
    print(f"\n  Đã lưu ảnh demo: Augument/mixup_demo.png")