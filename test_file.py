from argparse import ArgumentParser
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models.generator import InpaintSANet
from PIL import Image

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/train_1.yml',
                    help="training configuration")
parser.add_argument('--image_path', type=str, default = "D:/STUTI/Research/Inpainting/DATA/Validation/val_256/Places365_val_00000216.jpg" )
parser.add_argument('--mask_path', type=str, default= "D:/STUTI/Research/Inpainting/DATA/Mask/mask_nvidia/train_mask/00028.png")
parser.add_argument('--img_size', default=(256, 256))
parser.add_argument('--output_path', type=str, default='checkpoints/train_1/test/output.jpg')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/train_1/model/latest_ckpt.pth.tar')
args = parser.parse_args()
transform = transforms.ToTensor()

def load_img(path):
    img = Image.open(path).convert('RGB')
    img = img.resize(args.img_size)
    img = transform(img)
    img = 2*img - 1
    img = img.unsqueeze_(0)
    return img

def load_mask(path):
    mask = Image.open(path).convert('RGB')
    mask = mask.resize(args.img_size)
    mask = mask.point(lambda p: p > 127 and 255)  
    mask = transform(mask)
    mask = mask[0:1,:,:].unsqueeze_(0)
    return mask

def main():
    
    #device
    device = torch.device('cuda')
    #Load Model
    netG = InpaintSANet()
    nets = torch.load(args.checkpoint_path)
    netG_state_dict = nets['netG_state_dict']
    netG.load_state_dict(netG_state_dict)
    netG.to(device)
    netG.eval()
    
    img = load_img(args.image_path).to(device)
    mask = load_mask(args.mask_path).to(device)
    print(img.shape, mask.shape)
    coarse_img, recon_img = netG(img, mask)
    complete_img = recon_img * mask + img * (1 - mask)
    viz_images = torch.stack([img * (1 - mask), coarse_img, recon_img, complete_img, img], dim=1)
    viz_images = viz_images.view(-1, *list(coarse_img.size())[1:])
    vutils.save_image(viz_images,
                      args.output_path,
                      nrow=5,
                      normalize=True)

if __name__ == '__main__':
    main()

