"""
DETR Training Script
======================
Ref: Section 4 of "End-to-End Object Detection with Transformers"

Training setup:
    - AdamW optimizer with separate learning rates:
      * Transformer + prediction heads: lr = 10^-4
      * CNN backbone: lr = 10^-5
    - Weight decay: 10^-4
    - LR drop by factor 0.1 at epoch 200
    - Gradient clipping with max norm 0.1
    - Set-based loss: CE + L1 + GIoU via Hungarian matching

Usage:
    python train.py
"""

import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.train_config import *
from model.detr import build_detr
from model.matcher import HungarianMatcher
from model.criterion import SetCriterion
from dataset.dataset import DETRDataset, detr_collate_fn


def train():
    """
    Main training function for DETR.

    Implements the full training pipeline:
        1. Build model, criterion, optimizer
        2. Load dataset with custom collation
        3. Train for specified epochs with loss logging
        4. Save checkpoints periodically
    """
    # ==================== DEVICE ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"DETR Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
    print(f"Num queries: {NUM_QUERIES}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"LR (Transformer): {LR_TRANSFORMER}")
    print(f"LR (Backbone): {LR_BACKBONE}")
    print(f"{'='*60}")

    # ==================== MODEL ====================
    print("\n[1/4] Building DETR model...")
    model = build_detr(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        hidden_dim=HIDDEN_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pretrained_backbone=PRETRAINED_BACKBONE,
    )
    model = model.to(device)

    # Print model parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {n_params:,}")

    # ==================== LOSS ====================
    print("[2/4] Building loss function...")
    matcher = HungarianMatcher(
        cost_class=COST_CLASS,
        cost_bbox=COST_BBOX,
        cost_giou=COST_GIOU,
    )

    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_ce=WEIGHT_CE,
        weight_bbox=WEIGHT_BBOX,
        weight_giou=WEIGHT_GIOU,
        eos_coef=EOS_COEF,
    )
    criterion = criterion.to(device)

    # ==================== OPTIMIZER ====================
    # Ref: "We use AdamW with lr=10^-4 for transformer and FFNs, and
    #        lr=10^-5 for the backbone." (Section 4)
    print("[3/4] Setting up optimizer...")

    # Separate backbone and non-backbone parameters
    param_dicts = [
        {
            # Transformer + prediction heads (higher LR)
            "params": [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad],
            "lr": LR_TRANSFORMER,
        },
        {
            # Backbone (lower LR to preserve pretrained features)
            "params": [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": LR_BACKBONE,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=LR_TRANSFORMER,
        weight_decay=WEIGHT_DECAY
    )

    # LR scheduler: drop by factor at specified epoch
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_DROP,
        gamma=LR_DROP_FACTOR
    )

    # ==================== DATASET ====================
    print("[4/4] Loading dataset...")

    train_dataset = DETRDataset(
        root=TRAIN_DIR,
        img_size=IMAGE_SIZE,
        class_names=CLASS_NAMES,
        augment=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detr_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Optionally load validation set
    val_loader = None
    if os.path.exists(VAL_DIR) and os.path.exists(os.path.join(VAL_DIR, 'images')):
        val_dataset = DETRDataset(
            root=VAL_DIR,
            img_size=IMAGE_SIZE,
            class_names=CLASS_NAMES,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=detr_collate_fn,
            pin_memory=True,
        )

    # ==================== CHECKPOINT DIRECTORY ====================
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ==================== TRAINING LOOP ====================
    print(f"\n{'='*60}")
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"{'='*60}\n")

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        criterion.train()

        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_bbox = 0.0
        epoch_loss_giou = 0.0
        num_batches = 0

        epoch_start = time.time()

        # Training progress bar with tqdm
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="batch",
            ncols=120,
            bar_format='{l_bar}{bar:20}{r_bar}'
        )

        for batch_idx, (images, masks, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(device)          # (B, 3, H, W)
            masks = masks.to(device)            # (B, H, W)
            targets = [{
                'labels': t['labels'].to(device),
                'boxes': t['boxes'].to(device),
            } for t in targets]

            # Forward pass
            outputs = model(images, mask=masks)

            # Compute set-based loss
            losses = criterion(outputs, targets)
            loss = losses['loss_total']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            # Ref: "We clip gradients with a max norm of 0.1." (Section 4)
            if CLIP_MAX_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)

            optimizer.step()

            # Accumulate losses
            epoch_loss += loss.item()
            epoch_loss_ce += losses['loss_ce'].item()
            epoch_loss_bbox += losses['loss_bbox'].item()
            epoch_loss_giou += losses['loss_giou'].item()
            num_batches += 1

            # Update tqdm progress bar with current loss values
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{losses["loss_ce"].item():.4f}',
                'L1': f'{losses["loss_bbox"].item():.4f}',
                'GIoU': f'{losses["loss_giou"].item():.4f}',
            })

        # Step learning rate scheduler
        lr_scheduler.step()

        # Compute epoch averages
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_ce = epoch_loss_ce / max(num_batches, 1)
        avg_bbox = epoch_loss_bbox / max(num_batches, 1)
        avg_giou = epoch_loss_giou / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        # ---- Epoch Summary ----
        current_lr_t = optimizer.param_groups[0]['lr']
        current_lr_b = optimizer.param_groups[1]['lr']

        print(
            f"\nEpoch [{epoch}/{EPOCHS}] completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} "
            f"(CE: {avg_ce:.4f}, L1: {avg_bbox:.4f}, GIoU: {avg_giou:.4f}) | "
            f"LR: T={current_lr_t:.2e}, B={current_lr_b:.2e}"
        )

        # ==================== VALIDATION ====================
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader,
                    desc=f"  Val {epoch}/{EPOCHS}",
                    unit="batch",
                    ncols=120,
                    bar_format='{l_bar}{bar:20}{r_bar}'
                )
                for images, masks, targets in val_pbar:
                    images = images.to(device)
                    masks = masks.to(device)
                    targets = [{
                        'labels': t['labels'].to(device),
                        'boxes': t['boxes'].to(device),
                    } for t in targets]

                    outputs = model(images, mask=masks)
                    losses = criterion(outputs, targets)
                    val_loss += losses['loss_total'].item()
                    val_batches += 1

                    val_pbar.set_postfix({
                        'Val Loss': f'{losses["loss_total"].item():.4f}'
                    })

            avg_val_loss = val_loss / max(val_batches, 1)
            print(f"  Val Loss: {avg_val_loss:.4f}")

            # Track best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
                print(f"  ★ New best model saved (val_loss: {best_loss:.4f})")
        else:
            # Without validation, track training loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
                print(f"  ★ New best model saved (train_loss: {best_loss:.4f})")

        # ==================== PERIODIC CHECKPOINT ====================
        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f'detr_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        print()  # Blank line between epochs

    # ==================== TRAINING COMPLETE ====================
    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"{'='*60}")

    # Save final model
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': avg_loss,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'num_queries': NUM_QUERIES,
        'hidden_dim': HIDDEN_DIM,
    }, os.path.join(CHECKPOINT_DIR, 'detr_final.pth'))
    print(f"Final model saved: {os.path.join(CHECKPOINT_DIR, 'detr_final.pth')}")


if __name__ == '__main__':
    train()