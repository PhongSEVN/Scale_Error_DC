import torch
import torch.nn as nn
import torchvision.models as models


class BackboneResNet(nn.Module):

    def __init__(self, name: str = "resnet50", hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()

        if name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.num_channels = 2048
        elif name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.num_channels = 512
        elif name == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            self.num_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.conv_proj = nn.Conv2d(self.num_channels, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_proj(self.backbone(x))
