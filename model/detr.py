import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import BackboneResNet
from model.position_encoding import PositionalEncodingSine
from model.transformer import DETRTransformer


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):

    def __init__(self, num_classes: int, hidden_dim: int = 256,
                 nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, num_queries: int = 100,
                 pretrained_backbone: bool = True, backbone_name: str = "resnet50"):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.backbone = BackboneResNet(
            name=backbone_name, hidden_dim=hidden_dim, pretrained=pretrained_backbone
        )
        self.pos_encoder = PositionalEncodingSine(hidden_dim=hidden_dim)
        self.transformer = DETRTransformer(
            d_model=hidden_dim, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, samples: torch.Tensor, mask: torch.Tensor = None) -> dict:
        B, C, H, W = samples.shape

        features = self.backbone(samples)
        _, _, fH, fW = features.shape

        if mask is None:
            feat_mask = torch.zeros((B, fH, fW), dtype=torch.bool, device=features.device)
        else:
            feat_mask = F.interpolate(
                mask.unsqueeze(1).float(), size=(fH, fW), mode='nearest'
            ).squeeze(1).bool()

        pos = self.pos_encoder(feat_mask)

        src = features.flatten(2).permute(2, 0, 1)
        pos_flat = pos.flatten(2).permute(2, 0, 1)
        mask_flat = feat_mask.flatten(1)

        hs = self.transformer(
            src=src, pos=pos_flat,
            query_embed=self.query_embed.weight,
            mask=mask_flat
        )

        hs = hs.transpose(0, 1)
        return {
            'pred_logits': self.class_embed(hs),
            'pred_boxes': self.bbox_embed(hs).sigmoid(),
        }


def build_detr(num_classes: int, num_queries: int = 100,
               hidden_dim: int = 256, pretrained_backbone: bool = True,
               backbone_name: str = "resnet50", **kwargs) -> DETR:
    return DETR(
        num_classes=num_classes, hidden_dim=hidden_dim,
        num_queries=num_queries, pretrained_backbone=pretrained_backbone,
        backbone_name=backbone_name, **kwargs
    )
