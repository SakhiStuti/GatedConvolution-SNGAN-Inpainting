from argparse import ArgumentParser
import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from models.generator import InpaintSANet
from inpaint_random_mask_dataset import Places2_rmask
from util.config import Config

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/train_1.yml',
                    help="training configuration")
parser.add_argument('--image_root', type=str, default = "D:/STUTI/Research/Inpainting/DATA/Validation/val_256/" )
parser.add_argument('--output_root', type=str, default='checkpoints/train_1/test/')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/train_1/model/latest_ckpt.pth.tar')


def main():
    args = parser.parse_args()
    config = Config(args.config)

    device = torch.device('cuda')
    
    #Load Model
    netG = InpaintSANet()
    nets = torch.load(args.checkpoint_path)
    netG_state_dict = nets['netG_state_dict']
    netG.load_state_dict(netG_state_dict)
    netG.to(device)
    netG.eval()


    dataset = Places2_rmask(config.VAL_IMG_ROOT)
    loader = DataLoader(dataset, batch_size=1, 
                               shuffle=False, 
                               num_workers=0,
                               pin_memory=True)
    for i, (img, mask) in enumerate(loader):
            
            img, mask = img.to(device), mask.to(device)
            print(img.shape, mask.shape)
            coarse_img, recon_img = netG(img, mask)
            complete_img = recon_img * mask + img * (1 - mask)
            viz_images = torch.stack([img * (1 - mask), coarse_img, recon_img, complete_img, img], dim=1)
            viz_images = viz_images.view(-1, *list(coarse_img.size())[1:])
            vutils.save_image(viz_images,
                                  os.path.join(args.output_root, '{}.png'.format(i)),
                                  nrow=5,
                                  normalize=True)
            
            print(i)
            if i > 100:
                break

if __name__ == '__main__':
    main()
