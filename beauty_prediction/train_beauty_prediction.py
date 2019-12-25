from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from torch.autograd import Variable
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from faces_dataset import FacesDataset
from torch.utils.data.sampler import SubsetRandomSampler
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, help='Where to load the dataset from')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--beauty_rates', type=int, default=60, help='number of beauty rates/output neurons for the last layer')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=1e-4')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
opt = parser.parse_args()
print(opt)

# create experiments directory
if not os.path.exists(opt.experiment):
    os.makedirs(opt.experiment)

# use cuda if available, cpu if not
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# VGG-16 Takes 224x224 images as input
transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.RandomRotation(10),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])

# load labeled beauty rates dataset
dataset = FacesDataset(opt.dataset, transform)

# split dataset to 80% train, 20% validation
train_split = .8
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split_val = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices = indices[split_val:]
val_indices = indices[:split_val]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)

# create data loaders for train and validation sets
train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            sampler=train_sampler, num_workers=int(opt.workers))
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            sampler=validation_sampler, num_workers=int(opt.workers))

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(pretrained=True)

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, opt.beauty_rates)]) # Add our layer with opt.beauty_rates outputs and activation on it
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

# check if several GPUs exist and move model to gpu if available
if torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs.")
    vgg16 = nn.DataParallel(vgg16)
else:
    print("Running on single GPU.")
vgg16.to(device)

# define loss and optimization
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=opt.lr)

# function to train the model
def train_model(vgg, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    min_loss = 9999.0
    
    # save losses to plot graph
    avg_loss = 0
    avg_loss_val = 0
    avg_loss_list = []
    avg_loss_val_list = []
    
    train_batches = len(train_loader)
    val_batches = len(validation_loader)
    
    for epoch in range(opt.niter):
        print("Epoch {}/{}".format(epoch, opt.niter))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        
        # change model to training mode
        vgg.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches))

            # get images and labels from data loader
            images, beauty_rates, beauty_class = data
            
            # move to gpu if available
            if torch.cuda.is_available():
                images, beauty_rates = Variable(images.to(device)), Variable(beauty_rates.to(device))
            else:
                images, beauty_rates = Variable(images), Variable(beauty_rates)
            
            optimizer.zero_grad()

            # infer images and compute loss
            outputs = vgg(images)
            beauty_rates = torch.squeeze(beauty_rates, 1)
            loss = criterion(outputs, beauty_rates)
            
            loss.backward()
            optimizer.step()

            # sum batches losses
            #loss_train += loss.data[0]
            loss_train += loss.data

            # free memory
            del images, beauty_class, beauty_rates, outputs
            torch.cuda.empty_cache()
    
        print()
        
        avg_loss = loss_train / (len(dataset)*train_split)
        avg_loss_list.append(avg_loss)
        
        vgg.train(False)
        vgg.eval()
        
        for i, data in enumerate(validation_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches))
            
            # get images and labels from data loader
            images, beauty_rates, beauty_class = data
            
            # move to gpu if available
            if torch.cuda.is_available():
                with torch.no_grad():
                    images, beauty_rates = Variable(images.to(device)), Variable(beauty_rates.to(device))
            else:
                with torch.no_grad():
                    images, beauty_rates = Variable(images), Variable(beauty_rates)
            
            optimizer.zero_grad()

            # out of memory runtime error
            with torch.no_grad():
                # infer images and compute loss
                outputs = vgg(images)
                beauty_rates = torch.squeeze(beauty_rates, 1)
                loss = criterion(outputs, beauty_rates)

                # sum batches losses
                loss_val += loss.data

                # free memory
                del images, beauty_rates, beauty_class, outputs
                torch.cuda.empty_cache()
            
        avg_loss_val = loss_val / (len(dataset)*validation_split)
        avg_loss_val_list.append(avg_loss_val)
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print('-' * 10)
        print()

        # save model state if validation loss improved
        if avg_loss_val < min_loss:
            min_loss = avg_loss_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Minimal loss: {:.4f}".format(min_loss))
    
    # plot graph
    plt.clf()
    plt.title('Beauty rates loss')
    plt.xlabel('Epoch')
    plt.plot(range(opt.niter), avg_loss_list, label='Train loss')
    plt.plot(range(opt.niter), avg_loss_val_list, label='Validation loss')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.savefig('{0}/beauty_rates_loss.png'.format(opt.experiment))

    vgg.load_state_dict(best_model_wts)
    return vgg

# train model and save final model
vgg16 = train_model(vgg16, criterion, optimizer, opt.niter)
torch.save(vgg16.state_dict(), '{0}/VGG16_beauty_rates-new.pt'.format(opt.experiment))
