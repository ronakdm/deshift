import torch
import torch.nn as nn

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if act: layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBlock(channels, channels, 3),
            ConvBlock(channels, channels, 3, act=False)
        )
        self.act = nn.ReLU(inplace=True)
        self.γ = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        out = x + self.γ * self.residual(x)
        out = self.act(out)
        return out
    
class DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBlock(in_channels, out_channels, 3),
            nn.MaxPool2d(2)
        )

class ResidualLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            DownBlock(in_channels, out_channels),
            ResidualBlock(out_channels)
        )

class TemperatureScaler(nn.Module):
    def __init__(self, scaling_factor=0.1):
        super().__init__()
        self.scaler = nn.Parameter(torch.tensor(scaling_factor))

    def forward(self, x):
        return x * self.scaler
    
class Head(nn.Sequential):
    def __init__(self, in_channels, classes):
        super().__init__(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, classes),
            TemperatureScaler()
        )

class Net(nn.Sequential):
    def __init__(self, classes, hidden_channels, in_channels=3):
        channels = [hidden_channels * 2**num for num in range(4)]
        super().__init__(
            ConvBlock(in_channels, hidden_channels, 3),
            ResidualLayer(channels[0], channels[1]),
            DownBlock(channels[1], channels[2]),
            ResidualLayer(channels[2], channels[3]),
            Head(channels[3], classes)
        )