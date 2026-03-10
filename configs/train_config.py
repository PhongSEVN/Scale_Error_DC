"""
DETR Training Configuration
==============================
Ref: Section 4 of "End-to-End Object Detection with Transformers"

Hyperparameters based on the DETR paper:
    - AdamW optimizer with initial lr=10^-4 (Transformer) and 10^-5 (backbone)
    - Weight decay: 10^-4
    - lr drop at epoch 200 (for full 300 epoch training)
    - Batch size: the paper uses 64 across 16 GPUs; we default to a smaller
      value suitable for single-GPU training
"""

import os

# =============================================================================
# Data Configuration
# =============================================================================

# Path to the dataset directory (contains images/ and labels/ subdirectories)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'valid')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# Class names from data.yaml (5 classes)
CLASS_NAMES = ['DiVat', 'DiVatLoiLom', 'LoiChi', 'LoiNhua', 'LoiTray']
NUM_CLASSES = len(CLASS_NAMES)  # 5 (no-object class added internally by DETR)

# =============================================================================
# Model Configuration
# =============================================================================

# Transformer architecture
HIDDEN_DIM = 256       # Transformer hidden dimension (d_model)
NHEAD = 8              # Number of attention heads
NUM_ENCODER_LAYERS = 6 # Number of Transformer encoder layers
NUM_DECODER_LAYERS = 6 # Number of Transformer decoder layers
DIM_FEEDFORWARD = 2048 # FFN intermediate dimension
DROPOUT = 0.1          # Dropout rate

# Object queries
# Ref: "We use N = 100 detection slots (object queries)." (Section 4)
NUM_QUERIES = 100

# =============================================================================
# Training Configuration
# =============================================================================

# Image size: shortest side resized to this value
# Ref: DETR uses multi-scale with shortest side in [480, 800]
IMAGE_SIZE = 480

# Batch size (reduced from paper's 64 for single-GPU training)
BATCH_SIZE = 1

# Number of data loading workers
NUM_WORKERS = 2

# Training epochs
EPOCHS = 30

# Learning rate schedule
# Ref: "We use AdamW with lr=10^-4 for the transformer and 10^-5 for
#       the backbone." (Section 4)
LR_TRANSFORMER = 1e-4  # Learning rate for Transformer + prediction heads
LR_BACKBONE = 1e-5     # Learning rate for the CNN backbone (lower to not
                        # destroy pretrained features)
WEIGHT_DECAY = 1e-4     # AdamW weight decay

# LR scheduling
# Ref: "We drop the learning rate by a factor of 10 after 200 epochs."
LR_DROP = 35            # Epoch at which to drop LR (70% of total epochs)
LR_DROP_FACTOR = 0.1    # Factor by which to multiply LR at lr_drop

# Gradient clipping
# Ref: "We clip gradients with a max norm of 0.1." (Section 4)
CLIP_MAX_NORM = 0.1

# =============================================================================
# Loss Configuration
# =============================================================================

# Loss weights
# Ref: "λ_L1 = 5, λ_iou = 2" (Section 4)
WEIGHT_CE = 1.0         # Cross-Entropy classification loss weight
WEIGHT_BBOX = 5.0       # L1 bounding box loss weight
WEIGHT_GIOU = 2.0       # GIoU loss weight
EOS_COEF = 0.1          # No-object class weight in CE loss
                        # Ref: "Down-weighted by a factor of 10"

# Matching cost weights (for Hungarian Matcher)
COST_CLASS = 1.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# =============================================================================
# Backbone Configuration
# =============================================================================

PRETRAINED_BACKBONE = True  # Use ImageNet-pretrained ResNet-50

# =============================================================================
# Checkpointing
# =============================================================================

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
SAVE_EVERY = 10          # Save checkpoint every N epochs
LOG_INTERVAL = 10        # Print loss every N batches
