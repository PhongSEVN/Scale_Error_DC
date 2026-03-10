"""
DETR: End-to-End Object Detection with Transformers
====================================================
Reproduction of the DETR architecture from:
    "End-to-End Object Detection with Transformers"
    by Nicolas Carion et al. (2020)
    https://arxiv.org/abs/2005.12872

This package contains:
    - backbone: ResNet50 feature extractor
    - position_encoding: Fixed sine/cosine spatial positional encodings
    - transformer: Standard Transformer Encoder-Decoder
    - detr: Main DETR model combining all components
    - matcher: Hungarian bipartite matching
    - criterion: Set-based loss function (CE + L1 + GIoU)
"""

from model.detr import DETR, build_detr
from model.matcher import HungarianMatcher
from model.criterion import SetCriterion

__all__ = ['DETR', 'build_detr', 'HungarianMatcher', 'SetCriterion']
