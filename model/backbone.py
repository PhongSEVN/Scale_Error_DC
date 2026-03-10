"""
DETR Backbone: ResNet50 Feature Extractor
==========================================
Ref: Section 3.1 of "End-to-End Object Detection with Transformers"

The backbone CNN generates a lower-resolution activation map from the input image.
We use a ResNet-50 pretrained on ImageNet, removing the final classification (fc)
layer and the average pooling layer. The output feature map has stride 32, meaning
for an input image of shape (3, H, W), the output is (2048, H/32, W/32).

A 1x1 convolution then projects the 2048-channel feature map down to a smaller
dimension `d` (the hidden_dim of the Transformer, default=256).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BackboneResNet50(nn.Module):
    """
    ResNet-50 backbone for DETR.

    Extracts features from the last convolutional layer (layer4) of ResNet-50,
    then projects them to the Transformer hidden dimension using a 1x1 conv.

    Ref: "We use a ResNet-50 backbone, and the output of the last layer is
          a feature map of C=2048 channels." (Section 3.1)
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        """
        Args:
            hidden_dim: Projection dimension for the Transformer (d_model).
                        DETR uses 256 by default.
            pretrained: Whether to load ImageNet-pretrained weights.
        """
        super().__init__()

        # Load pretrained ResNet-50; remove avgpool and fc layers
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Extract all layers except avgpool and fc
        # This gives us: conv1, bn1, relu, maxpool, layer1-4
        self.backbone = nn.Sequential(
            resnet.conv1,    # (3, H, W) -> (64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (64, H/2, W/2) -> (64, H/4, W/4)
            resnet.layer1,   # (64, H/4, W/4) -> (256, H/4, W/4)
            resnet.layer2,   # (256, H/4, W/4) -> (512, H/8, W/8)
            resnet.layer3,   # (512, H/8, W/8) -> (1024, H/16, W/16)
            resnet.layer4,   # (1024, H/16, W/16) -> (2048, H/32, W/32)
        )

        # 1x1 convolution to reduce channel dimension from 2048 -> hidden_dim
        # Ref: "We reduce the channel dimension of the feature map from C to
        #        a smaller dimension d using a 1x1 convolution." (Section 3.1)
        self.conv_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        self.hidden_dim = hidden_dim
        self.backbone_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input images, shape (B, 3, H, W)

        Returns:
            features: Projected feature map, shape (B, hidden_dim, H/32, W/32)
        """
        # Extract features from ResNet-50
        features = self.backbone(x)  # (B, 2048, H/32, W/32)

        # Project to Transformer dimension
        features = self.conv_proj(features)  # (B, hidden_dim, H/32, W/32)

        return features
