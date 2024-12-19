import torch
import torch.nn as nn

from typing import Any, List, Optional, Sequence, Tuple, Union

from ..blocks.base import BaseBlock
from ..blocks.cnn import get_cnn_block



class Encoder(BaseBlock):
    def __init__(
        self,
        in_channels,
        kernel_sizes,
        features,
        strides,
        maxpools,
        dropouts,
        norm_name="batch",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        block: nn.Module = nn.Identity,
        spatial_dims=2,
        **kwargs,
    ) -> Any:
        super().__init__()

        # >>> checking
        assert isinstance(kernel_sizes, list), "kernel_sizes must be a list"
        assert isinstance(features, list), "features must be a list"
        assert isinstance(strides, list), "strides must be a list"
        assert (
            len(kernel_sizes) == len(strides) == len(features)
        ), "blocks, kernel_sizes, features, and strides must have the same length"
        if not isinstance(dropouts, list):
            dropouts = [dropouts for _ in features]
        in_out_channles = [in_channels] + features
        in_out_channles = [
            (i, o) for i, o in zip(in_out_channles[:-1], in_out_channles[1:])
        ]

        self.downs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for (ich, och), ks, st, mp, do in zip(
            in_out_channles, kernel_sizes, strides, maxpools, dropouts
        ):
            self.blocks.append(block(ich))
            down = get_cnn_block(code="r")(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                kernel_size=ks,
                stride=1 if mp else st,
                norm_name=norm_name,
                act_name=act_name,
                dropout=do,
            )
            self.downs.append(down if not mp else nn.Sequential(down, nn.MaxPool2d(kernel_size=st, stride=st)))

        self.apply(self._init_weights)

    def forward(self, x):
        layer_features = []
        for block, down in zip(self.blocks, self.downs):
            x = block(x)
            layer_features.append(x.clone())
            x = down(x)
        return x, layer_features

