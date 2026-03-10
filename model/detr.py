"""
DETR: DEtection TRansformer — Main Model
==========================================
Ref: "End-to-End Object Detection with Transformers" (Carion et al., 2020)

The DETR model chains:
    1. CNN Backbone (ResNet-50) → feature map (B, d, H/32, W/32)
    2. Positional Encoding → fixed sine/cosine 2D encoding
    3. Transformer Encoder-Decoder → N output embeddings
    4. Prediction FFNs → class logits + bounding box coordinates

The inference code below mirrors the "less than 50 lines" concept from the paper,
expanded for full training support with masks and variable-size batching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import BackboneResNet50
from model.position_encoding import PositionalEncodingSine
from model.transformer import DETRTransformer


class MLP(nn.Module):
    """
    Simple multi-layer perceptron (used for bounding box prediction).

    Ref: "The box prediction head is a 3-layer FFN with ReLU activation
          and hidden dimension d, predicting the normalized box center
          coordinates and its height and width." (Section 3.1)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """
    DETR: End-to-End Object Detection with Transformers.

    Architecture overview:
        Image → Backbone(ResNet50) → 1x1 Conv → Flatten + PosEnc →
        Transformer Encoder-Decoder(object queries) →
        FFN heads (class + bbox)

    The model predicts a fixed-size set of N predictions (default N=100),
    where each prediction is a (class, bounding_box) pair. Unmatched
    predictions are assigned the special "no-object" (∅) class.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256,
                 nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, num_queries: int = 100,
                 pretrained_backbone: bool = True):
        """
        Args:
            num_classes: Number of object classes (NOT including the no-object class).
                         The no-object class is added internally as class index `num_classes`.
            hidden_dim: Transformer hidden dimension d_model (default=256).
            nhead: Number of attention heads (default=8).
            num_encoder_layers: Number of Transformer encoder layers (default=6).
            num_decoder_layers: Number of Transformer decoder layers (default=6).
            dim_feedforward: FFN intermediate dimension (default=2048).
            dropout: Dropout rate (default=0.1).
            num_queries: Number of object queries N (default=100).
                         Ref: "We use N = 100 object queries." (Section 4)
            pretrained_backbone: Whether to use ImageNet-pretrained ResNet-50.
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # ==================== BACKBONE ====================
        self.backbone = BackboneResNet50(
            hidden_dim=hidden_dim,
            pretrained=pretrained_backbone
        )

        # ==================== POSITIONAL ENCODING ====================
        # Fixed sine/cosine 2D positional encoding
        self.pos_encoder = PositionalEncodingSine(hidden_dim=hidden_dim)

        # ==================== TRANSFORMER ====================
        self.transformer = DETRTransformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # ==================== PREDICTION HEADS ====================
        # Classification head: predicts (num_classes + 1) scores per query
        # The +1 is for the "no-object" (∅) class
        # Ref: "The class prediction FFN predicts the class label using a
        #        softmax function." (Section 3.1)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        # Bounding box regression head: 3-layer MLP predicting (cx, cy, w, h)
        # Output is normalized to [0, 1] via sigmoid
        # Ref: "Box coordinates are predicted by a 3-layer FFN with ReLU."
        self.bbox_embed = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3
        )

        # ==================== OBJECT QUERIES ====================
        # Learnable positional embeddings for the decoder
        # Ref: "N learnable positional embeddings, referred to as object queries"
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, samples: torch.Tensor,
                mask: torch.Tensor = None) -> dict:
        """
        Forward pass of DETR.

        Args:
            samples: Batch of images, shape (B, 3, H, W).
                     Images should be normalized and potentially padded.
            mask: Binary mask, shape (B, H, W). True = padded pixel.
                  If None, assumes no padding (all pixels valid).

        Returns:
            dict with:
                'pred_logits': Class logits, shape (B, N, num_classes + 1)
                'pred_boxes':  Predicted boxes (cx, cy, w, h) normalized to [0,1],
                               shape (B, N, 4)
        """
        B, C, H, W = samples.shape

        # ---- Step 1: Backbone feature extraction ----
        features = self.backbone(samples)  # (B, hidden_dim, H', W')
        _, _, fH, fW = features.shape

        # ---- Step 2: Generate mask for the feature map ----
        if mask is None:
            # No padding: all positions are valid
            feat_mask = torch.zeros(
                (B, fH, fW), dtype=torch.bool, device=features.device
            )
        else:
            # Downsample the image-level mask to match feature map resolution
            # Using nearest interpolation to preserve binary nature
            feat_mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=(fH, fW),
                mode='nearest'
            ).squeeze(1).bool()  # (B, fH, fW)

        # ---- Step 3: Positional encoding ----
        pos = self.pos_encoder(feat_mask)  # (B, hidden_dim, fH, fW)

        # ---- Step 4: Flatten spatial dimensions for Transformer ----
        # (B, d, fH, fW) -> (B, d, fH*fW) -> (fH*fW, B, d)
        src = features.flatten(2).permute(2, 0, 1)       # (HW, B, d)
        pos_flat = pos.flatten(2).permute(2, 0, 1)        # (HW, B, d)
        mask_flat = feat_mask.flatten(1)                   # (B, HW)

        # ---- Step 5: Transformer encoder-decoder ----
        # Object queries: (N, d)
        hs = self.transformer(
            src=src,
            pos=pos_flat,
            query_embed=self.query_embed.weight,
            mask=mask_flat
        )  # (N, B, d)

        # ---- Step 6: Prediction heads ----
        # Transpose: (N, B, d) -> (B, N, d)
        hs = hs.transpose(0, 1)  # (B, N, d)

        # Class predictions: (B, N, num_classes + 1)
        outputs_class = self.class_embed(hs)

        # Bounding box predictions: (B, N, 4) — sigmoid to [0, 1]
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
        }


def build_detr(num_classes: int, num_queries: int = 100,
               hidden_dim: int = 256, pretrained_backbone: bool = True,
               **kwargs) -> DETR:
    """
    Factory function to build a DETR model.

    Args:
        num_classes: Number of object classes (excluding no-object).
        num_queries: Number of detection slots (default=100).
        hidden_dim: Transformer hidden dimension (default=256).
        pretrained_backbone: Use ImageNet-pretrained ResNet-50 (default=True).
        **kwargs: Additional arguments passed to DETR constructor.

    Returns:
        DETR model instance.
    """
    model = DETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        pretrained_backbone=pretrained_backbone,
        **kwargs
    )
    return model
