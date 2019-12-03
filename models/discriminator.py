import torch.nn as nn
from .helper import SNConvWithActivation, get_pad

class InpaintSADisciminator(nn.Module):
    def __init__(self):
        super(InpaintSADisciminator, self).__init__()
        imgsize = 128
        cnum = 64
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(4, cnum, 5, 2, padding=get_pad(imgsize, 5, 2)),
            SNConvWithActivation(cnum, 2*cnum, 5, 2, padding=get_pad(imgsize/2, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 5, 2, padding=get_pad(imgsize/4, 5, 2)),
            SNConvWithActivation(4*cnum, 4*cnum, 5, 2, padding=get_pad(imgsize/8, 5, 2)),
            SNConvWithActivation(4*cnum, 4*cnum, 5, 2, padding=get_pad(imgsize/16, 5, 2)),
            SNConvWithActivation(4*cnum, 4*cnum, 5, 2, padding=get_pad(imgsize/32, 5, 2)),
        )
    def forward(self, input):
        x = self.discriminator_net(input)
        #print(x.shape)
        x = x.view((x.size(0),-1))
        return x