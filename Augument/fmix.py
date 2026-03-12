"""
FMix Augmentation - Visualization Demo
=========================================
Dùng mask Fourier-based (dạng đám mây) để trộn hai ảnh.
Tạo mask có hình dạng tự nhiên, không phải hình chữ nhật.

Cách dùng:
    1. Sửa IMG_PATH_1 và IMG_PATH_2 bên dưới thành đường dẫn ảnh bạn muốn.
    2. Chạy: python Augument/fmix.py
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===================== CẤU HÌNH ĐƯỜNG DẪN ẢNH =====================
IMG_PATH_1 = r"d:\IT\Projects\Scale_Error_DC\data\train\MyImage-20250723140003680_jpg.rf.d69126d88ab8801e044c3c92316f89ac.jpg"
IMG_PATH_2 = r"d:\IT\Projects\Scale_Error_DC\data\train\MyImage-20250723140203700_jpg.rf.20cfeff53b7e10b02cfb6595b233ec32.jpg"
ALPHA = 1.0         # Tham số Beta distribution
DECAY_POWER = 3.0   # Độ suy giảm tần số cao (càng lớn -> mask mượt hơn)
# ====================================================================


def _generate_fourier_mask(shape, alpha=1.0, decay_power=3.0):
    """
    Tạo mask nhị phân dựa trên phổ Fourier ngẫu nhiên.

    Args:
        shape: (H, W) kích thước mask
        alpha: tham số Beta distribution cho lambda
        decay_power: kiểm soát độ mượt của mask (tần số cao bị suy giảm)

    Returns:
        mask: numpy array (H, W) với giá trị 0 hoặc 1
        lam: tỉ lệ diện tích mask = 1
    """
    # 1. Sample lambda từ Beta distribution
    lam = np.random.beta(alpha, alpha)

    # 2. Tạo lưới tần số 2D
    freqs_h = np.fft.fftfreq(shape[0])
    freqs_w = np.fft.fftfreq(shape[1])
    freq_grid = np.sqrt(freqs_h[:, None] ** 2 + freqs_w[None, :] ** 2)

    # 3. Tạo phổ Fourier ngẫu nhiên với suy giảm tần số cao
    #    (tần số cao bị nhân hệ số nhỏ -> mask sẽ mượt, không có nhiễu)
    random_spectrum = np.random.randn(*shape)
    decay = 1.0 / (freq_grid + 1e-6) ** decay_power
    shaped_spectrum = random_spectrum * decay

    # 4. Biến đổi ngược Fourier -> miền không gian
    spatial = np.fft.ifft2(shaped_spectrum).real
    spatial = np.abs(spatial)

    # 5. Threshold để tạo mask nhị phân
    #    Dùng percentile sao cho tỉ lệ vùng = 1 xấp xỉ lambda
    threshold = np.percentile(spatial, (1 - lam) * 100)
    mask = (spatial > threshold).astype(np.float32)

    actual_lam = mask.mean()
    return mask, actual_lam


def fmix(img1: Image.Image, img2: Image.Image,
         alpha: float = 1.0, decay_power: float = 3.0):
    """
    FMix: Trộn hai ảnh bằng mask Fourier-based.
    Vùng mask=1 lấy từ img1, vùng mask=0 lấy từ img2.

    Returns:
        result_img: Ảnh kết quả (PIL Image)
        lam: Tỉ lệ diện tích giữ lại từ img1
        mask: Mask đã dùng (numpy array)
    """
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size, Image.BILINEAR)
        arr2 = np.array(img2)

    H, W = arr1.shape[:2]
    mask, lam = _generate_fourier_mask((H, W), alpha=alpha, decay_power=decay_power)

    # Trộn ảnh theo mask
    mask_3d = mask[..., None]  # (H, W, 1) để broadcast
    mixed = arr1 * mask_3d + arr2 * (1 - mask_3d)
    mixed = mixed.astype(np.uint8)

    return Image.fromarray(mixed), lam, mask


if __name__ == "__main__":
    print("=" * 60)
    print("  FMix Augmentation Demo")
    print("=" * 60)

    img1 = Image.open(IMG_PATH_1).convert("RGB")
    img2 = Image.open(IMG_PATH_2).convert("RGB")

    result, lam, mask = fmix(img1, img2, alpha=ALPHA, decay_power=DECAY_POWER)

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Ảnh gốc A
    axes[0].imshow(img1)
    axes[0].set_title("Image A (Gốc)", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    # Ảnh gốc B
    axes[1].imshow(img2)
    axes[1].set_title("Image B (Trộn)", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    # Mask Fourier
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Fourier Mask", fontsize=14, fontweight='bold', color='green')
    axes[2].axis("off")

    # Kết quả FMix
    axes[3].imshow(result)
    axes[3].set_title(f"FMix Result (λ={lam:.2f})", fontsize=14, fontweight='bold', color='purple')
    axes[3].axis("off")

    fig.suptitle("FMix: Mask Fourier-based (dạng đám mây) trộn A và B",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r"d:\IT\Projects\Scale_Error_DC\Augument\fmix_demo.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Lambda (tỉ lệ giữ lại từ A): {lam:.4f}")
    print(f"  Decay Power: {DECAY_POWER}")
    print(f"  Vùng trắng (mask=1) ~= {lam*100:.1f}% diện tích ảnh")
    print(f"\n  Đã lưu ảnh demo: Augument/fmix_demo.png")