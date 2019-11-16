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

class BeautyRater:
    def __init__(self,model_path,beauty_rates=60,pad_x=0,pad_y=0):
        self.model_path=model_path
        self.beauty_rates=beauty_rates
        # define cuda as device if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True

        # VGG-16 Takes 224x224 images as input
        self.transform=transforms.Compose([
                                    transforms.Pad((pad_x,pad_y)),
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

        # Load the pretrained model from pytorch
        self.vgg16 = models.vgg16_bn(pretrained=True)
        #print(vgg16.classifier[6].out_features) # 1000

        # Freeze training for all layers
        for param in self.vgg16.features.parameters():
            param.require_grad = False
        # Newly created modules have require_grad=True by default
        num_features = self.vgg16.classifier[6].in_features
        features = list(self.vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, beauty_rates)]) # Add our layer with opt.beauty_rates outputs
        self.vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

        # check if several GPUs exist and move model to gpu if available
        if torch.cuda.device_count() > 1:
            print("Running on", torch.cuda.device_count(), "GPUs.")
            self.vgg16 = nn.DataParallel(self.vgg16)
        else:
            print("Running on single GPU.")
        self.vgg16.to(device)

        # upload pretrained weights from beauty labeled dataset
        self.vgg16.load_state_dict(torch.load(self.model_path))
        self.vgg16.eval()

    def predict(self, img_path):

        # open image, transform and upload to gpu
        img = Image.open(img_path)
        img = self.transform(img)
        img = torch.from_numpy(np.asarray(img))
        if torch.cuda.is_available():
            with torch.no_grad():
                img = Variable(img.cuda())
        else:
            with torch.no_grad():
                img = Variable(img)
        img = torch.unsqueeze(img,0)

        # infer image to receive beauty rates
        output=self.vgg16(img)
        # convert output tensor into list with rounded values
        output_list = (output.data.cpu().numpy().tolist())[0]
        output_list = [round(x,4) for x in output_list]
        ratings=np.array(output_list, dtype=np.float32)
        ratings=ratings.reshape((1,-1))
        print(ratings)
        print(ratings.shape)
        return ratings.astype(np.float32)
