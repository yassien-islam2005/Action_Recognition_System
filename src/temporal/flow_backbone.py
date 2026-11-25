
import torch
import torch.nn as nn
from torchvision.models import resnet18


class FlowBackbone(nn.Module):
    """
    ResNet18 backbone for extracting motion features
    from optical flow (Flow X and Flow Y).

    Input:
        (B, 2, 224, 224)

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

            weights = ResNet18_Weights.DEFAULT
            self.network = resnet18(weights=weights)

        # Replace the original RGB convolution (3 channels)
        # with a new convolution that accepts 2 flow channels.
        original_conv = self.network.conv1

        self.network.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # If pretrained weights are enabled, initialize the
        # 2-channel convolution using the RGB convolution weights.
        if pretrained:
            with torch.no_grad():
                rgb_weights = original_conv.weight
                mean_weights = rgb_weights.mean(dim=1, keepdim=True)

                self.network.conv1.weight.copy_(
                    mean_weights.repeat(1, 2, 1, 1)
                )

        # Remove the original 1000-class classifier
        self.network.fc = nn.Identity()

        self.feature_dim = 512

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, 2, H, W), got {tuple(x.shape)}"
            )

        if x.shape[1] != 2:
            raise ValueError(
                f"FlowBackbone expects 2 channels, got {x.shape[1]}"
            )

        features = self.network(x)

        return features
