import math
import torch
import torch.nn as nn


class PositionalEncodingSine(nn.Module):

    def __init__(self, hidden_dim: int = 256, temperature: float = 10000.0,
                 normalize: bool = True, scale: float = 2 * math.pi):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        half_dim = self.hidden_dim // 2
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
