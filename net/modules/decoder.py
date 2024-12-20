import torch
import torch.nn as nn

from typing import Any, List, Optional, Sequence, Tuple, Union

from ..blocks.base import BaseBlock, get_conv_layer
from ..blocks.cnn import get_cnn_block



class Decoder(BaseBlock):
    def __init__(
        self,
        in_channels,
        features,
        up_strides,
        block: nn.Module = nn.Identity,
        spatial_dims=2,
        up_transpose=True,
        skip_mode="cat",
        **kwargs,
    ) -> Any:
        super().__init__()
        assert isinstance(features, list), "features must be a list"
        assert (
            len(up_strides) == len(features)
        ), "features, and up_strides must have the same length"
        up_strides = [s if isinstance(s, list) else spatial_dims*[s] for s in up_strides]

        conv_block = nn.Conv2d if spatial_dims == 2 else nn.Conv3d

        in_out_channles = [in_channels] + features
        in_out_channles = [
            (i, o) for i, o in zip(in_out_channles[:-1], in_out_channles[1:])
        ]
        self.skip_mode = skip_mode
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for (ich, och), upst in zip(
            in_out_channles, up_strides
        ):
            self.ups.append(get_conv_layer(spatial_dims=spatial_dims, is_transposed=True,
                    in_channels=ich, out_channels=och, stride=upst, dropout=0.0, kernel_size=5, 
                    conv_only=True, bias=True, # since there is no batchnorm
                ) if up_transpose else nn.Sequential(
                    conv_block(ich, och, kernel_size=1, stride=1, bias=True),
                    nn.Upsample(scale_factor=tuple(upst), mode="trilinear" if spatial_dims==3 else "bilinear", align_corners=True),
                )
            )

            if skip_mode == "cat":
                self.blocks.append(nn.Sequential(
                    conv_block(och*2, och, kernel_size=1, stride=1, bias=True),
                    block(och)
                ))
            else:
                self.blocks.append(block(och))

        self.apply(self._init_weights)

    def forward(self, x, skips: list, return_outs=False):
        outs = []
        for up, block in zip(self.ups, self.blocks):
            x_up = up(x)
            skip = skips.pop()
            x = torch.cat([x_up, skip], dim=1) if self.skip_mode == "cat" else x_up+skip
            x = block(x)
            if return_outs:
                outs.append(x.clone())
        return (x, outs) if return_outs else x
