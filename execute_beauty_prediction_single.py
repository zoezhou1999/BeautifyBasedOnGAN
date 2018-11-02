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

# upload pretrained weights from beauty labeled dataset
folder = 'experiments/train_beauty_classifier_03/'
file = 'VGG16_beauty_rates.pt'
vgg16.load_state_dict(torch.load(folder + file))
vgg16.eval()

# move model to gpu if available
vgg16.to(device)

# open image, transform and upload to gpu
img = Image.open("1.jpg")
img = transform(img)
img = torch.from_numpy(np.asarray(img))
if torch.cuda.is_available():
    with torch.no_grad():
        img = Variable(img.cuda())
else:
    with torch.no_grad():
        img = Variable(img)
img = torch.unsqueeze(img,0)

print(img.size())

# infer image to receive beauty rates
output = vgg16(img)

print("grade:")
print(output.mean())



