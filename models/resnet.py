import torch.nn as nn
import torchvision.models as models


def get_model(num_classes: int = 10) -> nn.Module:
    """
    Returns a ResNet-18 model adapted for CIFAR-10.
    """
    model = models.resnet18(weights=None)

    # CIFAR-10 uses 32x32 images → adjust first layers
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # Final classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
