import torch
import torch.nn.functional as F

def accuracy(pred_pos, pred_neg):
    batch_size = pred_pos.shape[0]
    acc_pos = torch.div(torch.sum(torch.mean(pred_pos, dim = 1) > 0).type(torch.DoubleTensor),batch_size)
    acc_neg = torch.div(torch.sum(torch.mean(pred_neg, dim = 1) < 0).type(torch.DoubleTensor),batch_size)
    return acc_pos, acc_neg

class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=0.5):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return (self.weight * (torch.mean(F.relu(1.-pos))), self.weight * (torch.mean(F.relu(1.+neg))))


class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)
    

class ReconLoss(torch.nn.Module):
    """
    Reconstruction loss contain l1 loss, may contain perceptual loss

    """
    def __init__(self, c_alpha, r_alpha):
        super(ReconLoss, self).__init__()
        self.c_alpha = c_alpha
        self.r_alpha = r_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        course_recon_loss = self.c_alpha*torch.mean(torch.abs(imgs - coarse_imgs))
        fine_recon_loss = self.r_alpha*torch.mean(torch.abs(imgs - recon_imgs)) 
        return (course_recon_loss, fine_recon_loss)
