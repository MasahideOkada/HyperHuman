# Structure Guided Refiner

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class SGREncoderConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channnels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channnels, kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out

class SGREmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 4,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()
        self.embed = nn.Sequential(
            SGREncoderConv(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            *[
                SGREncoderConv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding
                ) for _ in range(num_layers - 2)
            ],
            SGREncoderConv(out_channels, out_channels, 3, stride=1, padding=1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.embed(x)
        return out

class SGREncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        num_conditions: int = 3,
        num_layers: int = 4,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()
        self.num_conditions = num_conditions
        self.embeds = nn.ModuleList(
            [
                SGREmbedding(
                    in_channels,
                    out_channels,
                    num_layers=num_layers,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ) for _ in range(num_conditions)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x (`torch.Tensor`)
                A tensor with the following shape `(num_branches, batch, channel, height, width)`
        """
        if x.shape[0] != self.num_conditions:
            raise ValueError(f"number of input conditions must be {self.num_conditions}, but got {x.shape[0]}")
        encoded = torch.stack([embed(x) for x, embed in zip(x, self.embeds)])
        return encoded.sum(dim=0)
