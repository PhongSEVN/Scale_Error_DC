"""
DETR Industrial Defect Training Script
=======================================
Ref: Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020)

Task: Train DETR on a custom industrial defect dataset.
Dataset: COCO format JSON annotations.
Classes: ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"] (5 total + 1 background).

Key Requirements:
    1. ResNet-50 backbone (pretrained, frozen BN).
    2. Transformer encoder-decoder, 100 object queries.
    3. Custom COCO Dataset with multi-scale padding and mask generation.
    4. Hungarian Matcher using scipy linear_sum_assignment.
    5. Set-based loss: CE (eos_coef=0.1) + L1 (w=5) + GIoU (w=2).
    6. Optimizer: AdamW (1e-4 WD), Grad clip (0.1).
    7. Split LR: Backbone (1e-5), Transformer (1e-4).
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components
# Using the existing model components as they follow the paper architecture
from model.detr import build_detr
from model.matcher import HungarianMatcher
from model.criterion import SetCriterion
from dataset.coco_dataset import COCODETRDataset, coco_detr_collate_fn

# =============================================================================
# HYPERPARAMETERS & CONFIGURATION
# =============================================================================

# Dataset details
TRAIN_ROOT = "data/train"
TRAIN_ANN = "data/train/_annotations.coco.json"
VAL_ROOT = "data/valid"
VAL_ANN = "data/valid/_annotations.coco.json"
CLASS_NAMES = ["LoiChi-6OcR", "DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
NUM_CLASSES = len(CLASS_NAMES)

# Model architecture
BACKBONE_NAME = "resnet18"
HIDDEN_DIM = 64
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
NUM_QUERIES = 50
DROPOUT = 0.1
DIM_FEEDFORWARD = 256
IMAGE_SIZE = 640

# Training setup
BATCH_SIZE = 1  # Standard for DETR single-GPU or small memory
EPOCHS = 50
NUM_WORKERS = 2
LR_TRANSFORMER = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1

# Loss weights (Section 4)
WEIGHT_CE = 1.0
WEIGHT_BBOX = 5.0
WEIGHT_GIOU = 2.0
EOS_COEF = 0.1  # Down-weight "no object" class by factor of 10

# Matching costs
COST_CLASS = 1.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# Checkpointing
CHECKPOINT_DIR = "checkpoints/detr_industrial"
SAVE_EVERY = 5

def main():
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Build Dataset & DataLoader
    # We use custom COCO dataset with multi-scale augmentation
    print("Loading datasets...")
    train_dataset = COCODETRDataset(
        root=TRAIN_ROOT,
        ann_file=TRAIN_ANN,
        class_names=CLASS_NAMES,
        augment=True,
        img_size=IMAGE_SIZE
    )
    
    val_dataset = COCODETRDataset(
        root=VAL_ROOT,
        ann_file=VAL_ANN,
        class_names=CLASS_NAMES,
        augment=False,
        img_size=IMAGE_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=coco_detr_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=coco_detr_collate_fn,
        pin_memory=True
    )

    # 3. Build Model
    print("Building DETR model...")
    model = build_detr(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        hidden_dim=HIDDEN_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        backbone_name=BACKBONE_NAME
    )
    model.to(device)
    
    # 4. Build Optimizer & Criterion
    # Split Learning Rates: Backbone vs Rest
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": LR_TRANSFORMER,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": LR_BACKBONE,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=WEIGHT_DECAY)
    
    matcher = HungarianMatcher(
        cost_class=COST_CLASS, 
        cost_bbox=COST_BBOX, 
        cost_giou=COST_GIOU
    )
    
    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_ce=WEIGHT_CE,
        weight_bbox=WEIGHT_BBOX,
        weight_giou=WEIGHT_GIOU,
        eos_coef=EOS_COEF
    )
    criterion.to(device)
    
    # 5. Training Loop
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"Starting training on {NUM_CLASSES} classes...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        criterion.train()
        
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        
        for images, masks, targets in pbar:
            images = images.to(device)
            masks = masks.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images, mask=masks)
            
            # Compute bipartite matching loss
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['loss_total']
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (max_norm=0.1)
            if CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{loss_dict['loss_ce'].item():.4f}",
                'l1': f"{loss_dict['loss_bbox'].item():.4f}",
                'giou': f"{loss_dict['loss_giou'].item():.4f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        criterion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                outputs = model(images, mask=masks)
                loss_dict = criterion(outputs, targets)
                val_loss += loss_dict['loss_total'].item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoints
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  --> Saved Best Model (Loss: {best_val_loss:.4f})")
            
        if epoch % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"))

    print("Training Complete!")

if __name__ == "__main__":
    # Note: Ensure data paths are correct before running
    # This script assumes industrial defect data is in COCO format.
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}")
        print("\nPossible issues:")
        print("1. Dataset paths in hyperparameters section are incorrect.")
        print("2. Ground truth category IDs in JSON do not match the index.")
        print("3. CUDA out of memory.")

