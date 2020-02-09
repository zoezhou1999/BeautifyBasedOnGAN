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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='experiments/train_beauty_vgg/VGG16_beauty_rates-new.pt', help='path to the trained VGG16 model')
parser.add_argument('--dataset', type=str, default='../datasets/CelebA-HQ', help='path to the dataset we want to label')
parser.add_argument('--beauty_rates', type=int, default=60, help='number of beauty rates/output neurons for the last layer')
parser.add_argument('--pad_x', type=int, default=0, help='pixels to pad the given images from left and right')
parser.add_argument('--pad_y', type=int, default=0, help='pixels to pad the given images from up and down')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# VGG-16 Takes 224x224 images as input
transform=transforms.Compose([
                              transforms.Pad((opt.pad_x,opt.pad_y)),
                              transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, opt.beauty_rates)]) # Add our layer with opt.beauty_rates outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

# move model to gpu
if torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs.")
    vgg16 = nn.DataParallel(vgg16)
else:
    print("Running on CPU.")
vgg16.to(device)

#For CPU
# torch.device('cpu')
vgg16.load_state_dict(torch.load(opt.model))

#upload pretrained weights from beauty labeled dataset
# vgg16.load_state_dict(torch.load(opt.model))
vgg16.eval()

# create beauty rates lists for each image in dataset
files = []
beauty_rates = []
images_dir = "{0}/img".format(opt.dataset)
number_of_images = len(os.listdir(images_dir))

for i, file in enumerate(sorted(os.listdir(images_dir))):

    # open image, transform and upload to gpu
    img = Image.open(os.path.join(images_dir,file))
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

    # convert output tensor into list with rounded values
    output_list = (output.data.cpu().numpy().tolist())[0]
    output_list = [round(x,4) for x in output_list]

    # add file and beauty rates to lists
    files.append(file)
    beauty_rates.append(output_list)

    if (i % 100 == 0):
        print('{0}/{1} images done'.format(i,number_of_images))

# convert lists to csv lines
csv_lines = []
for i in range(0,opt.beauty_rates):
    for j in range(0,number_of_images):
        csv_lines.append('{0},{1},{2},'.format(str(i+1),files[j],str(beauty_rates[j][i]*5.0)))

# write csv lines to file
csv_path = "{0}/All_Ratings.csv".format(opt.dataset)
with open(csv_path, "w") as csv_file:
    for line in csv_lines:
        csv_file.write(line)
        csv_file.write('\n')
