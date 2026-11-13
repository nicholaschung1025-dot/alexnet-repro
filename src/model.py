"""AlexNet, implemented from scratch.

The paper's two-GPU channel split is emulated with grouped convolutions where
valid: Conv2, Conv4 and Conv5 use groups=2. Conv1 uses groups=1 because its
input has only 3 channels; the paper split Conv1 by output filters, not by
input channels. Conv3 uses groups=1 so the two streams exchange information.
"""

from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}. Use 'relu' or 'tanh'.")


class AlexNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        activation: str = "relu",
        use_lrn: bool = True,
        init: str = "paper",
        paper_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = 224

        # Feature extractor: 5 conv layers, 3 pooling layers, optional LRN.
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.act1 = _activation(activation)
        self.lrn1 = (
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)
            if use_lrn
            else nn.Identity()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.act2 = _activation(activation)
        self.lrn2 = (
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)
            if use_lrn
            else nn.Identity()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.act3 = _activation(activation)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.act4 = _activation(activation)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.act5 = _activation(activation)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Classifier: FC1 -> ReLU -> dropout -> FC2 -> ReLU -> dropout -> FC3.
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.act6 = _activation(activation)
        self.fc2 = nn.Linear(4096, 4096)
        self.act7 = _activation(activation)
        self.fc3 = nn.Linear(4096, num_classes)

        self._initialize_weights(init, paper_bias)

    def _initialize_weights(self, init: str, paper_bias: float = 1.0) -> None:
        if init == "paper":
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            # The paper sets biases to 1 in Conv2/4/5 and the hidden FC layers,
            # giving ReLUs positive inputs early in training.
            for layer in (self.conv2, self.conv4, self.conv5, self.fc1, self.fc2):
                nn.init.constant_(layer.bias, paper_bias)
        elif init == "kaiming":
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        else:
            raise ValueError(f"Unsupported init: {init!r}. Use 'paper' or 'kaiming'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D input (N, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x.shape[1]}")
        if x.shape[2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size} input, got "
                f"{tuple(x.shape[2:])}. The FC layers assume 6x6 feature maps."
            )

        x = self.pool1(self.lrn1(self.act1(self.conv1(x))))
        x = self.pool2(self.lrn2(self.act2(self.conv2(x))))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool3(self.act5(self.conv5(x)))

        x = torch.flatten(x, 1)
        x = self.dropout(self.act6(self.fc1(x)))
        x = self.dropout(self.act7(self.fc2(x)))
        return self.fc3(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
