
import torch
import torch.nn as nn
from torchvision.models import resnet18


class RGBBackbone(nn.Module):
    """
    ResNet18 backbone for extracting spatial features
    from RGB video frames.

    Input:
        (B, 3, 224, 224)

    Output:
        (B, 512)
    """

    def __init__(self, pretrained=False):
        super().__init__()

        # Create ResNet18
        self.network = resnet18(weights=None)

        # Optionally load ImageNet pretrained weights
        if pretrained:
            from torchvision.models import ResNet18_Weights

            self.network = resnet18(
                weights=ResNet18_Weights.DEFAULT
            )

        # Remove the original 1000-class classifier
        self.network.fc = nn.Identity()

        self.feature_dim = 512

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, 3, H, W), got {tuple(x.shape)}"
            )

        if x.shape[1] != 3:
            raise ValueError(
                f"RGBBackbone expects 3 channels, got {x.shape[1]}"
            )

        features = self.network(x)

        return features
