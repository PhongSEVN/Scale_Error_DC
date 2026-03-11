"""
DETR Training Configuration - Ultra-Lite (Optimized for Laptop GPU)
===================================================================
"""

import os

# =============================================================================
# Data Configuration
# =============================================================================
DATA_ROOT = "data"
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TRAIN_ANN = os.path.join(TRAIN_ROOT, "_annotations.coco.json")
VAL_ROOT = os.path.join(DATA_ROOT, "valid")
VAL_ANN = os.path.join(VAL_ROOT, "_annotations.coco.json")

# 6 classes based on dataset analysis
CLASS_NAMES = ["LoiChi-6OcR", "DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# Model Configuration - Ultra-Lite Params
# =============================================================================
BACKBONE_NAME = "resnet18"
HIDDEN_DIM = 64
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_QUERIES = 50
DROPOUT = 0.1

# =============================================================================
# Training Configuration
# =============================================================================
IMAGE_SIZE = 640
BATCH_SIZE = 1
NUM_WORKERS = 2
EPOCHS = 50

# Learning Rates
LR_TRANSFORMER = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4

# LR Scheduling
LR_DROP = 35
LR_DROP_FACTOR = 0.1

# Gradient Clipping
CLIP_MAX_NORM = 0.1

# =============================================================================
# Loss & Matching Configuration
# =============================================================================
WEIGHT_CE = 1.0
WEIGHT_BBOX = 5.0
WEIGHT_GIOU = 2.0
EOS_COEF = 0.1

COST_CLASS = 1.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# =============================================================================
# Paths & Intervals
# =============================================================================
CHECKPOINT_DIR = "checkpoints/detr_industrial"
SAVE_EVERY = 5
PRETRAINED_BACKBONE = True
