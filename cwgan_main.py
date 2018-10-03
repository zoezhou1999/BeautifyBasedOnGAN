from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
import models as models
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from faces_dataset import FacesDataset

beauty_rates_number = 60

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
#os.system('mkdir {0}'.format(opt.experiment))
#TODO: maybe create folder
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              #transforms.Pad((20,0)),
                              transforms.Resize(opt.imageSize),
                              transforms.CenterCrop(opt.imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])
                              
#data_dir = 'beauty_dataset_labeled'
data_dir = 'celebA_aligned'
dataset = FacesDataset(data_dir, transform)
print("num elements in folder is: "+str(len(dataset)))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = models.CGAN_G(opt.imageSize, nz, nc, beauty_rates_number, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = models.CGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)

netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_beauty_rates_d = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
input_beauty_rates_g = torch.FloatTensor(opt.batchSize, 1, beauty_rates_number)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    input_beauty_rates_d = input_beauty_rates_d.cuda()
    input_beauty_rates_g = input_beauty_rates_g.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

mse_loss = nn.MSELoss()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

errD_real_list = []
errD_fake_list = []
errD_list = []
errG_list = []

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    
    # sum D and G errors per Epoch
    errD_real_batch = 0.0
    errD_fake_batch = 0.0
    errD_batch = 0.0
    errG_batch = 0.0
    
    i = 0
    batch_counter = 0
    while i < len(dataloader):
        
        batch_counter += 1
        
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        
        # sum D errors per Diters
        errD_real_diter = 0.0
        errD_fake_diter = 0.0
        errD_diter = 0.0

        j = 0
        while j < Diters and i < len(dataloader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, beauty_rates, _ = data

            padder = (2, 2, 31, 32)
            beauty_rates_d = F.pad(beauty_rates, padder, 'constant', 0)
            beauty_rates_d = torch.unsqueeze(beauty_rates_d, 1)
            
            netD.zero_grad()
            batch_size = real_cpu.size(0)
            
            if opt.cuda:
                beauty_rates_d = beauty_rates_d.cuda()
            input_beauty_rates_d.resize_as_(beauty_rates_d).copy_(beauty_rates_d)
            input_beauty_rates_dv = Variable(input_beauty_rates_d)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv,input_beauty_rates_dv)
            errD_real.backward(one)

            # train with fake
            beauty_rates = torch.unsqueeze(beauty_rates, 1).transpose(1,3)
            if opt.cuda:
                beauty_rates = beauty_rates.cuda()
            input_beauty_rates_g.resize_as_(beauty_rates).copy_(beauty_rates)
            input_beauty_rates_gv = Variable(input_beauty_rates_g)
            
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                
            noisev = Variable(noise, requires_grad=True) # totally freeze netG
            """
            print("input_beauty_rates_gv:")
            print(input_beauty_rates_gv.size())
            print("noisev:")
            print(noisev.size())
            """
            fake_out = netG(noisev, input_beauty_rates_gv)
            
            fake = Variable(fake_out.data)
            
            inputv = fake
            errD_fake = netD(inputv, input_beauty_rates_dv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()
            
            errD_real_diter += errD_real.data[0]
            errD_fake_diter += errD_fake.data[0]
            errD_diter += errD.data[0]
            
        errD_real_batch += (errD_real_diter/float(j))
        errD_fake_batch += (errD_fake_diter/float(j))
        errD_batch += (errD_diter/float(j))

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        input_beauty_rates_gv = Variable(input_beauty_rates_g)
        input_beauty_rates_dv = Variable(input_beauty_rates_d)

        #print("beauty rates:")
        #print(input_beauty_rates_g[0])

        fake = netG(noisev, input_beauty_rates_gv)
        errG = netD(fake, input_beauty_rates_dv)
        errG.backward(one)
        
        optimizerG.step()
        gen_iterations += 1
        
        errG_batch += errG.data[0]
        
        iterations_to_save_images = 500
        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(dataloader), gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        if gen_iterations % iterations_to_save_images == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            fake = netG(Variable(fixed_noise, requires_grad=True), Variable(input_beauty_rates_g, requires_grad=True))
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
            torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
            
    errD_real_list.append(errD_real_batch/float(batch_counter))
    errD_fake_list.append(errD_fake_batch/float(batch_counter))
    errD_list.append(errD_batch/float(batch_counter))
    errG_list.append(errG_batch/float(batch_counter))

    if epoch % 100 == 0:
        plt.clf()
        plt.title('D and G errors over iterations')
        plt.xlabel('Batch')
        plt.plot(range(len(errD_real_list)), errD_real_list, label='error D(x) (real)')
        plt.plot(range(len(errD_fake_list)), errD_fake_list, label='error D(G(z)) (fake)')
        plt.plot(range(len(errD_list)), errD_list, label='error D real - fake')
        plt.plot(range(len(errG_list)), errG_list, label='error G(z)')
        plt.legend(['error D(x) (real)', 'error D(G(z)) (fake)', 'error D real - fake', 'error G(z)'], loc='upper right')
        plt.savefig('{0}/gan_distribution.png'.format(opt.experiment))

