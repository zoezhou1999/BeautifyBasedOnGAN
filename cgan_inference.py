from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import csv
import models.dcgan_ours as dcgan_ours
import copy

from PIL import Image
from faces_dataset import FacesDataset
from torch.autograd import Variable
from numpy import genfromtxt
from torchvision import transforms, models

cudnn.benchmark = True

def get_beauty_level_from_index(mid,index):
    if index == mid:
        return 0
    else:
        #print("adding "+ str((index-mid)*0.2))
        return (index-mid)*0.1

parser = argparse.ArgumentParser()
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# create dataset and dataloader to give beauty rates vector
# transform=transforms.Compose([
#                               transforms.RandomHorizontalFlip(),
#                               transforms.Resize(opt.imageSize),
#                               transforms.CenterCrop(opt.imageSize),
#                               transforms.ToTensor(),
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                               ])
#
# data_dir = 'beauty_dataset_labeled'
# dataset = FacesDataset(data_dir, transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=2)
# data_iter = iter(dataloader)
# _, beauty_rates,_ = data_iter.next()


transform=transforms.Compose([
                              transforms.Resize(opt.imageSize),
                              transforms.CenterCrop(opt.imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])


device = torch.device("cuda:0")
nz = 100
ngf = 64
ndf = 64
nc = 3
raters_number = 60
ngpu = 2
n_extra_layers = 0


ngpu = 1
#netG = Generator(ngpu).to(device)
netG = dcgan_ours.CGAN_G(opt.imageSize, nz, nc, raters_number, ngf, ngpu, n_extra_layers).to(device)
#netG.apply(weights_init)
#folder = 'samples_cwgan_400faces_v2/'
folder = 'samples_cwgan_celeba_v1/'
#file = 'netG_epoch_3762.pth'
file = 'netG_epoch_160.pth'

netG.load_state_dict(torch.load(folder + file))
netG.eval()
#print(netG)

#noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

### make different beauty levels from noise made out of a real picture
pic = 'other_faces/bar.jpg'
img = Image.open(pic)
img = transform(img)
img = torch.from_numpy(np.asarray(img))
noise = img

if opt.cuda:
    noise = noise.cuda()

#print("noise:")
#print(noise)

noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
noisev = Variable(noise, requires_grad=True) # totally freeze netG


beauty_rates_folder = "beauty_rates/"
beauty_rates_file = "beauty_rates_np.csv"

#features_list = []
# face_level = 4
# for i in range(0,raters_number):
#     features_list.append((np.asarray(face_level, dtype=np.float32) / 5.0))



beauty_levels = 20
mid = beauty_levels / 2
features_list_array = {}

# my_mean = 1
# my_std = 0
#
# random_beauty_rates = np.random.normal(loc = my_mean, scale = my_std, size = raters_number)
# random_beauty_rates.astype(np.float32)
# for i in range(0,raters_number):
#     random_beauty_rates[i] = round(random_beauty_rates[i],1)
# print(random_beauty_rates)
# print("len random_beauty_rates: ")

#
for i in range(0,beauty_levels):
    features_list_array[i] = genfromtxt(beauty_rates_folder + beauty_rates_file)
    #features_list_array[i] = copy.deepcopy(random_beauty_rates)

    print("features_list_array[i]")
    print(features_list_array[i])



    print("read elements from csv:")
    for element in np.nditer(features_list_array[i]):
        element = round(element,1)
        #print(element)

    features_list_array[i].astype(np.float32)


    print("enhancing:")
    for j in range(0,len(features_list_array[i])):
        # print("before:")
        # print(features_list_array[i][j])
        features_list_array[i][j]+=get_beauty_level_from_index(mid,i)
        # print("after:")
        # print(features_list_array[i][j])

    features_list_array[i].astype(np.float32)




    beauty_rates = torch.from_numpy(np.asarray(features_list_array[i]).reshape([1,raters_number]))
    beauty_rates = beauty_rates.type(torch.FloatTensor)


    hist =torch.histc(beauty_rates, bins = 5 , min = 0.2, max = 1)
    print("beauty ranking histogram:")
    print(hist)


    ### save beauty rates to csv
    print("beauty_rates_np:")
    beauty_rates_np = np.array(beauty_rates)
    beauty_rates_np_one_dim = []
    for element in np.nditer(beauty_rates_np):
        beauty_rates_np_one_dim.append(element)

    beauty_rates_np_one_dim = np.array(beauty_rates_np_one_dim)
    np.savetxt('{0}/beauty_rates_np.csv'.format(opt.experiment), beauty_rates_np_one_dim, delimiter=",")

    ###

    input_beauty_rates_g = torch.FloatTensor(opt.batchSize, 1, raters_number)



    if opt.cuda:
        input_beauty_rates_g = input_beauty_rates_g.cuda()
        beauty_rates = beauty_rates.cuda()


    #print("beauty_rates:")
    #print(beauty_rates.size())

    #print("input_beauty_rates_g:")
    #print(input_beauty_rates_g.size())

    input_beauty_rates_g.resize_as_(input_beauty_rates_g).copy_(beauty_rates)
    input_beauty_rates_g = torch.unsqueeze(input_beauty_rates_g, 1).transpose(1,3)
    input_beauty_rates_gv = Variable(input_beauty_rates_g)



    print("dimentions of noisv: ")
    print(noisev.size())

    print("dimentions of input_beauty_rates_gv: ")
    print(input_beauty_rates_gv.size())

    print('mean: {0}'.format(input_beauty_rates_gv.mean()))
    print('median: {0}'.format(input_beauty_rates_gv.median()))
    print('std: {0}'.format(input_beauty_rates_gv.std()))

    mean_2d = round(input_beauty_rates_gv.mean(),1)

    output = netG(noisev, input_beauty_rates_gv)
    output.data = output.data.mul(0.5).add(0.5)
    vutils.save_image(output, '{0}/GAN_output_{1}_mean_{2}.png'.format(opt.experiment,i,mean_2d))


