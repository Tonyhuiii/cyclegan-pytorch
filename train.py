#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
import os
import time
from torchvision.utils import save_image

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from models import VGGLoss
from models import MultiscaleDiscriminator
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--display', type=int, default=5, help='display frequency')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--generator_A2B', type=str, default='checkpoint/905fresh/netG_A2B_50.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='checkpoint/905fresh/netG_B2A_50.pth', help='B2A generator checkpoint file')
parser.add_argument('--netD_A', type=str, default='checkpoint/905fresh/netD_A_50.pth', help='netD_A checkpoint file')
parser.add_argument('--netD_B', type=str, default='checkpoint/905fresh/netD_B_50.pth', help='netD_B checkpoint file')

opt = parser.parse_args()
print(opt)
print('start from {} epoch'.format(opt.epoch))

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
    
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
# netD_A = Discriminator(opt.input_nc)
# netD_B = Discriminator(opt.output_nc)
netD_A = MultiscaleDiscriminator(opt.input_nc, opt.ndf, opt.n_layers_D, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, num_D=1, getIntermFeat=False)   
netD_B = MultiscaleDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_D, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, num_D=1, getIntermFeat=False) 


#print(netG_A2B)
if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)


# Load state dicts

# netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
# netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
# netD_A.load_state_dict(torch.load(opt.netD_A))
# netD_B.load_state_dict(torch.load(opt.netD_B))
# print("load {} epoch model".format(opt.generator_A2B.split('/')[-1].split('.')[0].split('_')[-1]))


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
# criterion_VGG= VGGLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
# input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
# input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# input_C = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [
                transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.epoch, opt.n_epochs, len(dataloader),opt.display)
###################################


# Create checkpoint dirs if they don't exist
checkpoint_path='checkpoint/'+opt.dataroot.split('/')[-1]
# print(checkpoint_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Create medium dirs if they don't exist
medium_path='medium/'+opt.dataroot.split('/')[-1]
# print(medium_path)
if not os.path.exists(medium_path):
    os.makedirs(medium_path)

###
def GANloss(predict, target_is_real):
    loss=0
    # print(len(predict[0]))
    for pred in predict[0]:
        if target_is_real:
            target = Variable(Tensor(pred.size()).fill_(1.0), requires_grad=False)
        else:
            target = Variable(Tensor(pred.size()).fill_(0.0), requires_grad=False)
        loss += criterion_GAN(pred, target) 
    return loss


###
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

 
    
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    start=time.time()
    for i, batch in enumerate(dataloader):
        # Set model input
        A = batch['A'].cuda()

        real_B = batch['B'].cuda()
        real_A = A
        # real_A = -A

    
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0


        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        # loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        loss_GAN_A2B = GANloss(pred_fake, True)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        # loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        loss_GAN_B2A = GANloss(pred_fake, True)
        
        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        
      

        # ssimloss
        ssim_module = SSIM(data_range=1, size_average=True, channel=3)
        X1 = (real_A + 1)*0.5   # [-1, 1] => [0, 1]
        Y1 = (recovered_A + 1)*0.5  

        X2 = (real_B + 1)*0.5   # [-1, 1] => [0, 1]
        Y2 = (recovered_B + 1)*0.5

        ssim_A = 1 - ssim_module(X1, Y1)
        ssim_B = 1 - ssim_module(X2, Y2)

        #ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
        #ms_ssim_loss = 1 - ms_ssim_module(X, Y)
        
        # Total loss
        # loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB 
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + ssim_A + ssim_B
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = GANloss(pred_real, True)

        # Fake loss
        fake_A1=fake_A
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = GANloss(pred_fake, False)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = GANloss(pred_real, True)
        
        # Fake loss
        fake_B1=fake_B
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = GANloss(pred_fake, False)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        #############################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_cycle': (loss_cycle_ABA), 
                    'loss_identity': (loss_identity_A + loss_identity_B), 
                    'ssim_A': ssim_A,
                    'ssim_B': ssim_B,
                    # 'loss_VGG' :loss_VGG,
                    # 'loss_TV' :loss_TV,
                    'loss_D': (loss_D_A + loss_D_B)},
                    images={'A': A,
                    # 'blue_A': blue_A,  
                            'real_A': real_A, 'fake_B': fake_B1, 'rec_A':recovered_A,
                            'real_B': real_B,'fake_A': fake_A1, 'rec_B':recovered_B})
    last=time.time()
    print("time per epoch: {}s".format(last-start))
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    

    # Save models checkpoints
    if ((epoch+1)%5 ==0):
        torch.save(netG_A2B.state_dict(), checkpoint_path+'/netG_A2B_{}.pth'.format(epoch+1))
        torch.save(netG_B2A.state_dict(), checkpoint_path+'/netG_B2A_{}.pth'.format(epoch+1))
        torch.save(netD_A.state_dict(), checkpoint_path+'/netD_A_{}.pth'.format(epoch+1))
        torch.save(netD_B.state_dict(), checkpoint_path+'/netD_B_{}.pth'.format(epoch+1))

    # Save intermedium output
    if ((epoch+1)%1==0):
        A=0.5*(A+1.0)
        real_A=0.5*(real_A+1.0)
        fake_B=0.5*(fake_B1+1.0)
        # rec_A=0.5*(recovered_A+1.0)
        save_image(A[0], medium_path+'/{}_A.jpg'.format(epoch+1),padding=0)    
        save_image(real_A[0], medium_path+'/{}_real_A.jpg'.format(epoch+1),padding=0)    
        save_image(fake_B[0], medium_path+'/{}_fake_B.jpg'.format(epoch+1),padding=0)    
        # save_image(rec_A[0], medium_path+'/{}_rec_A.jpg'.format(epoch+1),padding=0)    
    
###################################
