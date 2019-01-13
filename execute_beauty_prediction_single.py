from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from torch.autograd import Variable
import os
import numpy as np
from PIL import Image
import csv

beauty_rates_number = 60

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--parallel', dest='parallel', action='store_true')
opt = parser.parse_args()
print(opt)

# define cuda as device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# VGG-16 Takes 224x224 images as input
transform=transforms.Compose([
                              transforms.Resize(opt.imageSize),
                              transforms.CenterCrop(opt.imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)
#print(vgg16.classifier[6].out_features) # 1000

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 60)]) # Add our layer with 5 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

# check if several GPUs exist and move model to gpu if available
if opt.parallel and torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs.")
    vgg16 = nn.DataParallel(vgg16)
else:
    print("Running on single GPU.")
vgg16.to(device)

# upload pretrained weights from beauty labeled dataset
folder = 'experiments/train_beauty_vgg_bestsofar/'
file = 'VGG16_beauty_rates.pt'
vgg16.load_state_dict(torch.load(folder + file))
vgg16.eval()

# open image, transform and upload to gpu
img = Image.open("sample1.png")
img = transform(img)
img = torch.from_numpy(np.asarray(img))
if torch.cuda.is_available():
    with torch.no_grad():
        img = Variable(img.cuda())
else:
    with torch.no_grad():
        img = Variable(img)
img = torch.unsqueeze(img,0)

# infer image to receive beauty rates
output = vgg16(img)

print("beauty_rates:")
print(output)
print("mean:")
print(output.mean())



