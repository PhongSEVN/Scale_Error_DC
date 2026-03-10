"""
DETR Positional Encodings
==========================
Ref: Section 3.1 and Supplementary Material of
     "End-to-End Object Detection with Transformers"

DETR uses fixed sine/cosine positional encodings (similar to "Attention Is All
You Need") to supplement the Transformer input with spatial position information.

The positional encoding is 2D: for each spatial position (x, y) in the feature map,
we compute d/2 sine and d/2 cosine values for the x-coordinate, and similarly for
the y-coordinate, giving a total of d dimensions.

The encoding uses the formula:
    PE(pos, 2i)     = sin(pos / 10000^(2i/d))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i/d))

Additionally, the mask is used to indicate padded positions (True = padded).
"""

import math
import torch
import torch.nn as nn


class PositionalEncodingSine(nn.Module):
    """
    Fixed 2D sine/cosine positional encoding for spatial feature maps.

    Ref: "We supplement it with fixed positional encodings that are added to
          the input of each attention layer." (Section 3.1)

    For a feature map of size (H, W), generates positional encodings of
    shape (1, hidden_dim, H, W) that encode the (x, y) position of each cell.
    """

    def __init__(self, hidden_dim: int = 256, temperature: float = 10000.0,
                 normalize: bool = True, scale: float = 2 * math.pi):
        """
        Args:
            hidden_dim: Dimensionality of the positional encoding (must match
                        Transformer d_model).
            temperature: Temperature for the sine/cosine frequency scaling.
            normalize: Whether to normalize positions to [0, 2*pi] range.
            scale: Scale factor for normalization (default: 2*pi).
        """
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even for sine/cosine encoding"

        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D positional encodings from a binary mask.

        Args:
            mask: Binary mask of shape (B, H, W), where True indicates
                  padded (invalid) positions.

        Returns:
            pos: Positional encoding of shape (B, hidden_dim, H, W).
        """
        # ~mask: True for valid positions, False for padded
        not_mask = ~mask  # (B, H, W)

        # Cumulative sum along y-axis (height) and x-axis (width)
        # This gives each valid position a unique (y, x) coordinate
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  # (B, H, W)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  # (B, H, W)

        if self.normalize:
            eps = 1e-6
            # Normalize to [0, scale] range
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Create frequency dimension indices
        # dim_t: [0, 1, 2, ..., hidden_dim/2 - 1]
        half_dim = self.hidden_dim // 2
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=mask.device)
        # temperature^(2i / d): geometric progression of frequencies
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_dim)

        # Compute positional encodings
        # Shape: (B, H, W) -> (B, H, W, half_dim)
        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, half_dim)
        pos_y = y_embed[:, :, :, None] / dim_t  # (B, H, W, half_dim)

        # Interleave sin and cos: sin for even indices, cos for odd indices
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)  # (B, H, W, half_dim)

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)  # (B, H, W, half_dim)

        # Concatenate x and y encodings along the channel dimension
        pos = torch.cat((pos_y, pos_x), dim=3)  # (B, H, W, hidden_dim)

        # Permute to (B, hidden_dim, H, W) to match feature map layout
        pos = pos.permute(0, 3, 1, 2)  # (B, hidden_dim, H, W)

        return pos
