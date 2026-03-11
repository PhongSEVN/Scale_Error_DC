"""
DETR COCO Dataset: Loading for COCO JSON Annotations
=====================================================
Ref: Section 4 of "End-to-End Object Detection with Transformers"

This module implements a custom DETRDataset that parses COCO-format JSON annotations.
It handles:
    - Loading image paths and bbox annotations (x, y, w, h) from JSON.
    - Converting COCO (x, y, w, h) to normalized (cx, cy, w, h) for DETR.
    - Multi-scale data augmentation (shortest side 480-800, longest at most 1333).
    - Custom collation for zero-padding and mask generation.
"""

import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List, Optional


class COCODETRDataset(Dataset):
    """
    Custom dataset for DETR training with COCO-format JSON annotations.
    """

    def __init__(self, root: str, ann_file: str, 
                 class_names: List[str],
                 augment: bool = True,
                 img_size: int = 800):
        """
        Args:
            root: Directory containing images.
            ann_file: Path to COCO JSON annotation file.
            class_names: List of class names (in order of IDs).
            augment: Whether to apply multi-scale and flip augmentation.
        """
        super().__init__()
        self.root = root
        self.augment = augment
        self.class_names = class_names
        self.img_size = img_size
        
        # Load COCO JSON
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
            
        # Create mapping from category ID to index
        self.cat_id_to_idx = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
        
        # Build image information lookup
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Build image to annotations mapping
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.img_ids = sorted(list(self.images.keys()))
        
        print(f"[COCODETRDataset] Loaded {len(self.img_ids)} images and {len(coco_data['annotations'])} annotations.")

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.root, file_name)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Load annotations
        anns = self.img_to_anns.get(img_id, [])
        labels = []
        boxes = []
        
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in self.cat_id_to_idx:
                continue
            
            # Label index
            labels.append(self.cat_id_to_idx[cat_id])
            
            # COCO box: [x, y, w, h] (top-left)
            # DETR box: [cx, cy, w, h] (normalized)
            x, y, w, h = ann['bbox']
            cx = x + w / 2
            cy = y + h / 2
            
            # Normalize to [0, 1] based on original size
            boxes.append([cx / orig_w, cy / orig_h, w / orig_w, h / orig_h])
            
        labels = torch.tensor(labels, dtype=torch.long)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # ---- Data Augmentation: Random Resize & Flip ----
        # Ref: Section 4. Shortest side in [480, 800], longest <= 1333
        if self.augment:
            # Random Horizontal Flip
            if torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
                if boxes.shape[0] > 0:
                    boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip cx

            # Random Resize
            shortest_side = torch.randint(int(self.img_size * 0.6), self.img_size + 1, (1,)).item()
            scale = shortest_side / min(orig_w, orig_h)
            
            # Ensure longest side proportional to img_size
            max_longest = int(1333 * (self.img_size / 800))
            if max(orig_w, orig_h) * scale > max_longest:
                scale = max_longest / max(orig_w, orig_h)
                
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = TF.resize(img, [new_h, new_w])
        else:
            # Fixed Resize for validation
            shortest_side = self.img_size
            scale = shortest_side / min(orig_w, orig_h)
            max_longest = int(1333 * (self.img_size / 800))
            if max(orig_w, orig_h) * scale > max_longest:
                scale = max_longest / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = TF.resize(img, [new_h, new_w])

        # ---- Normalize and convert to tensor ----
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)

        target = {
            'labels': labels,
            'boxes': boxes,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([img.shape[1], img.shape[2]])
        }

        return img, target


def coco_detr_collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Collate function to handle variable-size images.
    Pads to max size and generates boolean masks (True = pad).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    B = len(images)
    padded_images = torch.zeros(B, 3, max_h, max_w)
    masks = torch.ones(B, max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded_images[i, :, :h, :w] = img
        masks[i, :h, :w] = False

    return padded_images, masks, targets
