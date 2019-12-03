import torch
import torch.nn as nn
from .helper import GatedConv2dWithActivation, GatedDeConv2dWithActivation, get_pad

class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self):
        super(InpaintSANet, self).__init__()
        imgsize = 128
        cnum = 48
        self.coarse_net = nn.Sequential(
            GatedConv2dWithActivation(5, cnum, 5, 1, padding=get_pad(imgsize, 5, 1)),
            # downsample 
            GatedConv2dWithActivation(cnum//2, 2*cnum, 3, 2, padding=get_pad(imgsize, 3, 2)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(imgsize/2, 3, 1)),
            #downsample 
            GatedConv2dWithActivation(cnum, 4*cnum, 3, 2, padding=get_pad(imgsize/2, 3, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(imgsize/4, 3, 1, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(imgsize/4, 3, 1, 4)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(imgsize/4, 3, 1, 8)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(imgsize/4, 3, 1, 16)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 2*cnum, 2*cnum, 3, 1, padding=get_pad(imgsize/2, 3, 1)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(imgsize/2, 3, 1)),
            GatedDeConv2dWithActivation(2, cnum, cnum, 3, 1, padding=get_pad(imgsize, 3, 1)),
            GatedConv2dWithActivation(cnum//2, cnum//2, 3, 1, padding=get_pad(imgsize, 3, 1)),
            GatedConv2dWithActivation(cnum//4, 3, 3, 1, padding=get_pad(imgsize, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            GatedConv2dWithActivation(3, cnum, 5, 1, padding=get_pad(imgsize, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum//2, cnum, 3, 2, padding=get_pad(imgsize, 3, 2)),
            GatedConv2dWithActivation(cnum//2, 2*cnum, 3, 1, padding=get_pad(imgsize/2, 3, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 2, padding=get_pad(imgsize/2, 3, 2)),
            GatedConv2dWithActivation(cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(imgsize/4, 3, 1, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(imgsize/4, 3, 1, 4)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(imgsize/4, 3, 1, 8)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(imgsize/4, 3, 1, 16))
        )
        self.refine_attention_net_1 = nn.Sequential(
            GatedConv2dWithActivation(3, cnum, 5, 1, padding=get_pad(imgsize, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum//2, cnum, 3, 2, padding=get_pad(imgsize, 3, 2)),
            GatedConv2dWithActivation(cnum//2, 2*cnum, 3, 1, padding=get_pad(imgsize/2, 3, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 2, padding=get_pad(imgsize/2, 3, 2)),
            GatedConv2dWithActivation(cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1), activation=torch.nn.ReLU()),
        )
        self.refine_attention_net_2 = nn.Sequential(
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1))
        )
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(imgsize/4, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, 2*cnum, 3, 1, padding=get_pad(imgsize/2, 3, 1)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(imgsize/2, 3, 1)),
            GatedDeConv2dWithActivation(2, cnum, cnum, 3, 1, padding=get_pad(imgsize, 3, 1)),

            GatedConv2dWithActivation(cnum//2, cnum//2, 3, 1, padding=get_pad(imgsize, 3, 1)),
            GatedConv2dWithActivation(cnum//4, 3, 3, 1, padding=get_pad(imgsize, 3, 1), activation=None),
        )


    def forward(self, img, mask):
        
        #Coarse
        #print(img.shape)
        masked_img =  img * (1 - mask) + mask
        input_img = torch.cat([masked_img, mask, torch.full_like(mask, 1.)], dim=1)
        #print(input_img.shape)
        x = self.coarse_net(input_img)
        x = torch.tanh(x) 
        coarse_x = x
        
        # Refine
        masked_img = img * (1 - mask) + coarse_x * mask
        input_img = masked_img
        #input_imgs = torch.cat([masked_img, mask, torch.full_like(mask, 1.)], dim=1)
        x_conv = self.refine_conv_net(input_img)
        x_a_1 = self.refine_attention_net_1(input_img)
        x_att = self.refine_attention_net_2(x_a_1)
        x = torch.cat([x_conv, x_att], dim = 1)
        x = self.refine_upsample_net(x)
        x = torch.tanh(x)
        
        return coarse_x, x

