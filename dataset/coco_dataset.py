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
import random
import configs.train_config as cfg
from dataset.augmentations import mixup_detection, cutmix_detection, fmix_detection, copy_paste_minority


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

        # Index minority classes for balanced augmentation sampling
        self.minority_indices = []
        if self.augment:
            for i, img_id in enumerate(self.img_ids):
                anns = self.img_to_anns.get(img_id, [])
                for ann in anns:
                    cat_id = ann['category_id']
                    if cat_id in self.cat_id_to_idx:
                        if self.cat_id_to_idx[cat_id] in [4, 5]: # Rare classes
                            self.minority_indices.append(i)
                            break
            print(f"  --> Minority class images indexed: {len(self.minority_indices)}")

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

        target = {
            'labels': labels,
            'boxes': boxes,
        }

        # ---- Data Augmentation: Advanced (MixUp, CutMix, FMix) ----
        if self.augment:
            r = random.random()
            if r < cfg.MIXUP_PROB:
                other_img, other_target = self._get_second_sample(idx)
                img, target = mixup_detection(img, target, other_img, other_target)
            elif r < cfg.MIXUP_PROB + cfg.CUTMIX_PROB:
                other_img, other_target = self._get_second_sample(idx)
                img, target = cutmix_detection(img, target, other_img, other_target)
            elif r < cfg.MIXUP_PROB + cfg.CUTMIX_PROB + cfg.FMIX_PROB:
                other_img, other_target = self._get_second_sample(idx)
                img, target = fmix_detection(img, target, other_img, other_target)
            elif r < cfg.MIXUP_PROB + cfg.CUTMIX_PROB + cfg.FMIX_PROB + cfg.COPY_PASTE_PROB:
                # Always sample from minority pool for Copy-Paste
                other_img, other_target = self._get_second_sample(idx, force_minority=True)
                img, target = copy_paste_minority(img, target, other_img, other_target)

        # Re-extract labels/boxes after mixing
        labels = target['labels']
        boxes = target['boxes']

        # Update orig_w, orig_h if img was resized during mixing
        orig_w, orig_h = img.size

        # ---- Data Augmentation: Basic (Random Resize & Flip) ----
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

    def _get_second_sample(self, current_idx: int, force_minority: bool = False) -> Tuple[Image.Image, dict]:
        """Load a random second sample. Performs balanced sampling if needed."""
        # Weighted sampling: 50% chance to pick from minority pool if it exists
        if (force_minority or random.random() < 0.5) and len(self.minority_indices) > 0:
            idx = random.choice(self.minority_indices)
        else:
            idx = random.randint(0, len(self) - 1)
            
        if idx == current_idx and len(self) > 1:
            idx = (idx + 1) % len(self)
        
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img = Image.open(os.path.join(self.root, img_info['file_name'])).convert('RGB')
        orig_w, orig_h = img.size
        
        anns = self.img_to_anns.get(img_id, [])
        labels, boxes = [], []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self.cat_id_to_idx:
                labels.append(self.cat_id_to_idx[cat_id])
                x, y, w, h = ann['bbox']
                boxes.append([(x + w / 2) / orig_w, (y + h / 2) / orig_h, w / orig_w, h / orig_h])
        
        return img, {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32)
        }


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
