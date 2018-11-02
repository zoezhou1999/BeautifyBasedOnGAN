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
import h5py
import csv

beauty_rates_number = 60

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
opt = parser.parse_args()
print(opt)

cudnn.benchmark = True

# define cuda as device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset from h5 file
dataset_folder = './datasets/CelebA-HQ'
h5_file = 'celebahq_256.h5'
dataset = h5py.File(os.path.join(dataset_folder, h5_file), 'r')

# VGG-16 Takes 224x224 images as input
transform = transforms.Compose([
                              # transforms.Pad((50,0)),
                              # transforms.CenterCrop(178),
                              transforms.Resize(opt.imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])


# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)
# print(vgg16.classifier[6].out_features) # 1000

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 60)]) # Add our layer with 5 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

# upload pretrained weights from beauty labeled dataset
folder = 'experiments/train_beauty_classifier_02/'
file = 'VGG16_beauty_rates.pt'
vgg16.load_state_dict(torch.load(folder + file))
vgg16.eval()

# move model to gpu if available
vgg16.to(device)

# create beauty rates lists for each image in dataset
beauty_rates = []
number_of_images = len(dataset['data256x256'])


for i, img in enumerate(dataset['data256x256']):

    # TODO: check if img does contains a PIL image and not np array
    # img = np.array(img, dtype=np.float32)
    # img = Image.fromarray(img)
    img = transform(img)

    img = torch.from_numpy(np.asarray(img))
    if torch.cuda.is_available():
        with torch.no_grad():
            img = Variable(img.cuda())
    else:
        with torch.no_grad():
            img = Variable(img)
    img = torch.unsqueeze(img, 0)

    # infer image to receive beauty rates
    output = vgg16(img)

    # convert output tensor into list with rounded values
    output_list = (output.data.cpu().numpy().tolist())[0]
    output_list = [round(x, 4) for x in output_list]

    # add beauty rates to lists
    beauty_rates.append(output_list)

    if i % 100 == 0:
        print('{0}/{1} images done'.format(i, number_of_images))

# convert lists to csv lines
csv_lines = []
for i in range(0, beauty_rates_number):
    for j in range(0, number_of_images):
        csv_lines.append('{0},{1},'.format(str(i+1), str(beauty_rates[j][i]*5.0)))

# write csv lines to file
csv_path = "{0}/All_Ratings.csv".format(dataset_folder)
with open(csv_path, "wb") as csv_file:
    for line in csv_lines:
        csv_file.write(line)
        csv_file.write('\n')
