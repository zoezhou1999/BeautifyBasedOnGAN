import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import matplotlib.pyplot as plt

#####   Dataset for Face images with beauty rates   #####
# Each entry will contain:                              #
# Face image                                            #
# List of 60 beauty grades in the range of [1,5]        #

raters_number = 60

class FacesDataset(Dataset):
    
    # lists to store dataset:
    images = [] # each var is a string of image name
    beauty_rates = [] # each var is numpy in size of 60 with floats in range of [0,1]
    
    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        
        # Dictionary to load dataset
        # key: image name
        # value: list of 60 beauty rates from raters
        dataset_dict = {}
        
        # read raters csv file
        with open(folder_dataset + '/All_Ratings.csv', 'r') as csvfile:

            raw_dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(raw_dataset):
                row = ','.join(row)
                row = row.split(',')
                
                # create list of rates for each image
                if row[1] in dataset_dict:
                    dataset_dict[row[1]][0].append(float(row[2]))
                else:
                    dataset_dict[row[1]] = [[float(row[2])]]

        # move dict to lists, convert beauty rates to numpy ranged in [0,1]
        for key, value in dataset_dict.items():
            self.images.append(folder_dataset + '/img/' + key)
            self.beauty_rates.append((np.asarray(value, dtype=np.float32) / 5.0))

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        
        img = Image.open(self.images[index])
        #img = img.convert('RGB') #TODO: check if necessary
        
        # perform transform only on the image (!!)
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and beauty rates to torch tensors
        img = torch.from_numpy(np.asarray(img))
        features = torch.from_numpy(np.asarray(self.beauty_rates[index]).reshape([1,raters_number]))
        
        # compute class for beauty rates in [1,10]
        features_class = (torch.mean(features)* 10.0).int()
        
        #return img, features, Is_Beauty
        return img, features, features_class
    
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    
    train_dataset = FacesDataset('../datasets/beauty_dataset')

    # sample one image and beauty rates to test correlation
    image, features, features_class = train_dataset.__getitem__(5)
    
    print("beauty rates: "+ str(features))
    print("beauty rate mean: "+ str(features.mean()))
    print("beauty rate class: "+ str(features_class))
