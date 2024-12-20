import torch
import torch.nn as nn

from ..blocks.cnn import UnetResBlock, UnetOutBlock
from ..modules.encoder import Encoder
from ..modules.decoder import Decoder
from ..modules.moga import MogaBlock


class Net(nn.Module):
    def __init__(self,
        in_channels=1, 
        out_channels=9, 
        features=[32, 64, 128], 
        kernel_size=[3, 3, 3], 
        stride=[2, 2, 2],
        maxpools = [True, False, False],
        dropouts = [0.05, 0.05, 0.05],
        skip_mode="cat",
        norm_name="batch",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        block = MogaBlock, #nn.Identity,
        up_transpose=False,
        spatial_dims=2,
    ):

        super(Net, self).__init__()

        head_ch = features[0]//2

        self.init = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=head_ch,
            kernel_size=5,
            stride=1,
            dropout=0.0,
            norm_name=norm_name,
        )

        self.encoder = Encoder(
            in_channels=head_ch,
            features=features,
            kernel_sizes=kernel_size,
            strides=stride,
            maxpools=maxpools,
            dropouts=dropouts,
            norm_name=norm_name,
            act_name=act_name,
            block=block,
            spatial_dims=spatial_dims,
        )
        self.decoder = Decoder(
            in_channels=features[-1],
            features=features[:-1][::-1]+[head_ch],
            up_strides=stride[::-1],
            block=block,
            up_transpose=up_transpose,
            spatial_dims=spatial_dims,
            skip_mode=skip_mode,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=2*head_ch,
            out_channels=out_channels,
            dropout=0,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, x):
        x_head = self.init(x)
        x, skips = self.encoder(x_head)
        x = self.decoder(x, skips)
        x = self.out(torch.cat((x, x_head), dim=1))
        return x