"""
Simple 3D U-Net for volumetric brain tumor segmentation.

Architecture overview:
  Encoder (downsampling) -> Bottleneck -> Decoder (upsampling with skip connections)

Input:  (batch, 4, D, H, W)   — 4 MRI modalities
Output: (batch, 1, D, H, W)   — tumor probability map
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two 3D convolutions with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net with 3 encoder levels and 3 decoder levels.

    This is a lightweight version suitable for intermediate-level projects
    and GPUs with limited memory.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_features: int = 16):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(base_features * 4, base_features * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_features * 2, base_features)

        # Final 1x1x1 convolution -> single output channel
        self.out_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path (save skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder path (concatenate skip connections)
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
