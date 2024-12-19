import torch
import torch.nn as nn

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
        dropouts = [0.1, 0.1, 0.1],
        norm_name="batch",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        block = MogaBlock, #nn.Identity,
        up_transpose=False,
    ):

        super(Net, self).__init__()

        head_ch = features[0]//2

        self.init = nn.Sequential(
            nn.Conv2d(in_channels, head_ch, 3, 1, 1),
            nn.BatchNorm2d(head_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
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
            spatial_dims=2,
        )
        self.decoder = Decoder(
            in_channels=features[-1],
            features=features[:-1][::-1]+[head_ch],
            up_strides=stride[::-1],
            block=block,
            up_transpose=up_transpose,
            spatial_dims=2,
        )

        self.out = nn.Sequential(
            nn.Conv2d(head_ch, out_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_head = self.init(x)
        x, skips = self.encoder(x_head)
        x = self.decoder(x, skips)
        x = self.out(x+x_head)
        return x