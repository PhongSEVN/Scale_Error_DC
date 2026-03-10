"""
DETR Dataset: Custom Data Loading for YOLO-Format Annotations
===============================================================
Ref: Section 4 of "End-to-End Object Detection with Transformers"

This module implements a custom DETRDataset compatible with the YOLO annotation
format (one .txt file per image with lines: class_id cx cy w h, all normalized).

Key features:
    - Supports YOLO format (.txt) labels with normalized (cx, cy, w, h) boxes.
    - Custom collate_fn pads images to the same size within a batch and
      generates binary masks to indicate padded regions.
    - Applies standard DETR data augmentation (resize, normalize).

Ref: "DETR uses standard data augmentation: horizontal flips, scales, and
      crops." (Section 4)

Mask handling:
    Since images in a batch may have different sizes, we pad them to the
    largest size in the batch. The mask (True = padded) is used by the
    Transformer to ignore padded regions in attention computations.
"""

import os
import glob
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, List, Optional


class DETRDataset(Dataset):
    """
    Custom dataset for DETR training with YOLO-format annotations.

    Directory structure expected:
        root/
            images/
                image1.jpg
                image2.jpg
                ...
            labels/
                image1.txt
                image2.txt
                ...

    Each .txt label file contains one line per object:
        class_id center_x center_y width height
    where all coordinates are normalized to [0, 1].
    """

    def __init__(self, root: str, img_size: int = 800,
                 class_names: Optional[List[str]] = None,
                 augment: bool = True):
        """
        Args:
            root: Path to directory containing 'images/' and 'labels/' subdirs.
            img_size: Target size for the shorter side of the image.
                      DETR paper uses range [480, 800] during training.
            class_names: List of class names (for logging/debugging).
            augment: Whether to apply data augmentation (horizontal flip).
        """
        super().__init__()

        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.class_names = class_names or []

        # Discover image files
        self.img_dir = os.path.join(root, 'images')
        self.lbl_dir = os.path.join(root, 'labels')

        # Support common image formats
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.img_paths = []
        for ext in img_extensions:
            self.img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
        self.img_paths.sort()

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.img_dir}. "
                f"Expected structure: {root}/images/*.jpg"
            )

        print(f"[DETRDataset] Found {len(self.img_paths)} images in {self.img_dir}")

        # ImageNet normalization (standard for pretrained ResNet backbones)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_labels(self, img_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load YOLO-format labels corresponding to an image.

        Args:
            img_path: Path to the image file.

        Returns:
            labels: (M,) tensor of class indices.
            boxes: (M, 4) tensor of normalized (cx, cy, w, h) boxes.
        """
        # Derive label path: images/foo.jpg -> labels/foo.txt
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, img_name + '.txt')

        labels = []
        boxes = []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), \
                                       float(parts[3]), float(parts[4])
                        labels.append(cls_id)
                        boxes.append([cx, cy, w, h])

        if len(labels) == 0:
            return (torch.zeros(0, dtype=torch.long),
                    torch.zeros((0, 4), dtype=torch.float32))

        return (torch.tensor(labels, dtype=torch.long),
                torch.tensor(boxes, dtype=torch.float32))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Load and preprocess an image with its annotations.

        Processing pipeline:
            1. Load image as RGB PIL
            2. Load YOLO labels
            3. Resize image (keeping aspect ratio) so shortest side = img_size
            4. Optionally apply horizontal flip augmentation
            5. Normalize with ImageNet stats
            6. Convert to tensor

        Args:
            idx: Index of the sample.

        Returns:
            image: Tensor of shape (3, H, W), normalized.
            target: Dict with:
                'labels': (M,) class label indices
                'boxes': (M, 4) boxes in (cx, cy, w, h) normalized to [0,1]
                'image_id': tensor with the image index
                'orig_size': tensor [orig_H, orig_W]
        """
        img_path = self.img_paths[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Load labels
        labels, boxes = self._load_labels(img_path)

        # ---- Resize (shortest side to img_size, maintain aspect ratio) ----
        # Ref: DETR uses multi-scale training with shortest side in [480, 800]
        scale = self.img_size / min(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = TF.resize(img, [new_h, new_w])
        # Note: Boxes are normalized, so they remain valid after resize

        # ---- Data Augmentation ----
        if self.augment and torch.rand(1).item() > 0.5:
            # Horizontal flip
            img = TF.hflip(img)
            if boxes.shape[0] > 0:
                # Flip cx: new_cx = 1 - cx
                boxes[:, 0] = 1.0 - boxes[:, 0]

        # ---- Convert to tensor and normalize ----
        img = TF.to_tensor(img)  # (3, H, W), float [0, 1]
        img = TF.normalize(img, mean=self.mean, std=self.std)

        target = {
            'labels': labels,           # (M,)
            'boxes': boxes,             # (M, 4) — (cx, cy, w, h) normalized
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_h, orig_w]),
        }

        return img, target


def detr_collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Custom collate function for DETR that handles variable-size images.

    Since images in a batch may have different dimensions, we:
        1. Find the maximum height and width in the batch.
        2. Pad all images to (max_H, max_W) with zeros.
        3. Create a binary mask: True = padded pixel, False = valid pixel.

    This mask is critical for the Transformer attention mechanism to
    ignore padded regions.

    Ref: "We pad images in the same batch to the same size and create
          a binary mask." (Supplementary Material A.4)

    Args:
        batch: List of (image, target) tuples from DETRDataset.__getitem__.

    Returns:
        images: Padded images, shape (B, 3, max_H, max_W).
        masks: Binary padding masks, shape (B, max_H, max_W).
        targets: List of B target dicts (unchanged).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Find max dimensions in the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    B = len(images)
    padded_images = torch.zeros(B, 3, max_h, max_w)
    masks = torch.ones(B, max_h, max_w, dtype=torch.bool)  # True = padded

    for i, img in enumerate(images):
        _, h, w = img.shape
        padded_images[i, :, :h, :w] = img
        masks[i, :h, :w] = False  # Mark valid (non-padded) pixels

    return padded_images, masks, targets
