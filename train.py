import os
from argparse import ArgumentParser

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from util.config import Config
from util.logger import get_logger
from inpaint_random_mask_dataset import Places2_rmask
from loss import SNDisLoss, SNGenLoss, ReconLoss, accuracy
from models.generator import InpaintSANet
from models.discriminator import InpaintSADisciminator

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
#Config File Path
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/train_1.yml', help="training configuration")
args = parser.parse_args()

#get all arguments
config = Config(args.config)

#checkpoint paths and directories
checkpoint_path = os.path.join('./checkpoints', config.LOG_DIR)
check_dir(checkpoint_path)
tensorboard_dir = os.path.join(checkpoint_path, 'tensorboard')
check_dir(tensorboard_dir)
model_dir = os.path.join(checkpoint_path, 'model')
check_dir(model_dir)
img_dir = os.path.join(checkpoint_path, 'images')
check_dir(img_dir)
val_img_dir = os.path.join(checkpoint_path, 'val_images')
check_dir(val_img_dir)

#logger and tensorboard initializations
logger = get_logger(os.path.join(checkpoint_path))
writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'tensorboard'))

#Device
device = torch.device('cuda:{}'.format(config.GPU_ID))
    
def main():
    
    #logger.info("Arguments: {}".format(config))
    
    #Load dataset
    logger.info("Initialize the dataset...")
    train_dataset = Places2_rmask(config.IMG_ROOT,
                                  resize_shape=tuple(config.IMG_SHAPES),
                                  random_ff_setting=config.RANDOM_FF_SETTING,
                                  transforms_oprs=['resize', 'to_tensor'])
    train_loader = DataLoader(train_dataset, 
                               batch_size=config.BATCH_SIZE, 
                               shuffle=True, 
                               num_workers=0,
                               pin_memory=True)
    
    val_dataset = Places2_rmask(config.VAL_IMG_ROOT)
    val_loader = DataLoader(val_dataset, 
                               batch_size=4, 
                               shuffle=False, 
                               num_workers=0,
                               pin_memory=True)
    
    val_img, val_mask = iter(val_loader).next()
    val_img, val_mask = val_img.to(device), val_mask.to(device)
    
    logger.info("Finish the dataset initialization.")

   
    # Define the Network Structure
    logger.info("Define the Network Structure and Losses")
    netG = InpaintSANet()
    netD = InpaintSADisciminator()
    start_iter = 0

    #Resume Network
    if config.MODEL_RESTORE != '':
        whole_model_path = os.path.join(model_dir,config.MODEL_RESTORE)
        nets = torch.load(whole_model_path)
        netG_state_dict, netD_state_dict, start_iter = nets['netG_state_dict'], nets['netD_state_dict'], nets['start_iter']
        netG.load_state_dict(netG_state_dict)
        netD.load_state_dict(netD_state_dict)
        logger.info("Loading pretrained models from {} ...".format(config.MODEL_RESTORE))
        
    #Transfer to device
    netG.to(device)
    netD.to(device)

    # Define loss
    recon_loss = ReconLoss(*(config.L1_LOSS_ALPHA))
    SN_gen_loss = SNGenLoss(config.GAN_LOSS_ALPHA)
    SN_dis_loss = SNDisLoss()
    lr, decay = config.LEARNING_RATE, config.WEIGHT_DECAY
    beta1, beta2 = 0.5, 0.99
    optG = torch.optim.Adam(netG.parameters(), lr=lr, weight_decay=decay, betas=(beta1, beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, weight_decay=decay, betas=(beta1, beta2))
    logger.info("Finish Define the Network Structure and Losses")

    
    
    # Start Training
    logger.info("Start Training...")
    epoch = config.EPOCH
    N = len(train_loader)
    for e in range(epoch):
        for i, (img, mask) in enumerate(train_loader):
            
            img, mask = img.to(device), mask.to(device)
            # mask is 1 on masked region
            #img is -1 to 1
            
            if config.PRETRAIN:
                # OPTIMIZE GENERATOR
                optG.zero_grad(), netG.zero_grad()
                coarse_img, recon_img = netG(img, mask)
                complete_img = recon_img * mask + img * (1 - mask)
                r_g_loss_course, r_g_loss_fine = recon_loss(img, coarse_img, recon_img, mask)
                r_g_loss  = r_g_loss_course + r_g_loss_fine
                g_loss = r_g_loss 
                # Backprop for g
                g_loss.backward()
                optG.step()   
                
            else:    
                # OPTIMIZE DISCRIMINATOR
                optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()
                #forward on generator
                coarse_img, recon_img = netG(img, mask)
                #Reconstructed image
                complete_img = recon_img * mask + img * (1 - mask)
                #Prepare input to discriminator
                pos_img = torch.cat([img, mask], dim=1)
                neg_img = torch.cat([complete_img, mask], dim=1)
                pos_neg_img = torch.cat([pos_img, neg_img], dim=0)
                #forward on discriminator
                pred_pos_neg = netD(pos_neg_img)
                pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
                acc_pos, acc_neg = accuracy(pred_pos, pred_neg)
                acc = torch.div((acc_pos + acc_neg),2.0)
                sn_d_loss_pos, sn_d_loss_neg = SN_dis_loss(pred_pos, pred_neg)
                sn_d_loss = sn_d_loss_pos + sn_d_loss_neg
                #Backprop on discriminator
                sn_d_loss.backward(retain_graph=True)
                optD.step()
            
            
                # OPTIMIZE GENERATOR
                optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
                pred_neg = netD(neg_img)
                #gan loss
                sn_g_loss = SN_gen_loss(pred_neg)
                r_g_loss_course, r_g_loss_fine = recon_loss(img, coarse_img, recon_img, mask)
                r_g_loss  = r_g_loss_course + r_g_loss_fine
                g_loss = sn_g_loss + r_g_loss
                # Backprop for g
                g_loss.backward()
                optG.step()
                
            #SAVE the training progress
            if (i) % config.SUMMARY_FREQ == 0:
                # Tensorboard logger for scaler and images
                if config.PRETRAIN:
                    info_terms = {'recon_course':r_g_loss_course.item(), 'recon_fine':r_g_loss_fine.item(), 'G':g_loss.item()}
                    for tag, value in info_terms.items():
                        writer.add_scalar(tag, value,  (e*N + i + start_iter))
                else:
                    info_terms = {'G':g_loss.item(),  'Recon_G':r_g_loss.item(), 
                                  "GAN_G":sn_g_loss.item(), "GAN_D":sn_d_loss.item(), "acc_D":acc.item()}
                    for tag, value in info_terms.items():
                        writer.add_scalar(tag, value,  (e*N + i + start_iter))
                    writer.add_scalar(f'GAN_d/GAN_d_pos',    sn_d_loss_neg.item(),    (e*N + i + start_iter))
                    writer.add_scalar(f'GAN_d/GAN_d_neg',    sn_d_loss_pos.item(),    (e*N + i + start_iter))
                    writer.add_scalar(f'accuracy/acc_pos',    acc_pos.item(),    (e*N + i + start_iter))
                    writer.add_scalar(f'accuracy/acc_neg',   acc_neg.item(),    (e*N + i + start_iter))
                    writer.add_scalar(f'recon/coarse',    r_g_loss_course.item(),    (e*N + i + start_iter))
                    writer.add_scalar(f'recon/fine',   r_g_loss_fine.item(),    (e*N + i + start_iter))
                    
            if (i) % config.IMG_FREQ == 0:
                viz_images = torch.stack([img * (1 - mask), coarse_img, recon_img, complete_img, img], dim=1)
                viz_images = viz_images.view(-1, *list(coarse_img.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (img_dir, e*N + i + start_iter),
                                  nrow=5,
                                  normalize=True)
            if (i) % config.VAL_IMG_FREQ == 0:
                netG.eval()
                val_coarse_img, val_recon_img = netG(val_img, val_mask)
                val_complete_img = val_recon_img * val_mask + val_img * (1 - val_mask)
                val_images = torch.stack([val_img * (1 - val_mask), val_coarse_img, val_recon_img, val_complete_img, val_img], dim=1)
                val_images = val_images.view(-1, *list(val_coarse_img.size())[1:])
                vutils.save_image(val_images,
                                  '%s/niter_%03d.png' % (val_img_dir, e*N + i + start_iter),
                                  nrow=5,
                                  normalize=True)
                netG.train()
            if (i) % config.SAVE_MODEL_FREQ == 0:
                saved_model = {
                    'start_iter': e*N+i+start_iter+1,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                }
                modelname = 'iter_{}.pth.tar'.format(e*N+i+start_iter)
                torch.save(saved_model, os.path.join(model_dir, modelname))
                torch.save(saved_model, os.path.join(model_dir, 'latest_ckpt.pth.tar'))
                
            
            
            
            
               
        
if __name__ == '__main__':
    main()
