# -*- coding: utf-8 -*-
import os, scipy.misc
from glob import glob
import numpy as np
import h5py
import csv



def get_img(img_path, is_crop=True, crop_h=256, resize_h=64, normalize=False):
    img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
    resize_w = resize_h
    if is_crop:
        crop_w = crop_h
        h, w = img.shape[:2]
        j = int(round((h - crop_h)/2.))
        i = int(round((w - crop_w)/2.))
        cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
    else:
        cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])
    if normalize:
        cropped_image = cropped_image/127.5 - 1.0
    return np.transpose(cropped_image, [2, 0, 1])

class CelebAHQ():
    
    beauty_rates = [] # each var is numpy in size of 60 with floats in range of [0,1]
    
    def __init__(self):
        
        datafolder = './datasets/CelebA-HQ'

        # import CelebA-HQ images
        datafile = 'celebahq_256.h5'
        resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64', \
                      'data128x128', 'data256x256']
        self._base_key = 'data'
        self.dataset = h5py.File(os.path.join(datafolder, datafile), 'r')
        self._len = {k:len(self.dataset[k]) for k in resolution}
        assert all([resol in self.dataset.keys() for resol in resolution])

        # import predicted beauty rates
        
        # Dictionary to load dataset
        # key: image index
        # value: list of 60 beauty rates from raters
        dataset_dict = {}
        
        with open('./' + datafolder + '/All_Ratings.csv', 'r') as csvfile:
            
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
            self.beauty_rates.append((np.asarray(value, dtype=np.float32) / 5.0))
    

    def __call__(self, batch_size, size, level=None):
        
        # sample [batch_size] images randomly
        key = self._base_key + '{}x{}'.format(size, size)
        idx = np.random.randint(self._len[key], size=batch_size)
        batch_x = np.array([self.dataset[key][i]/127.5-1.0 for i in idx], dtype=np.float32)
        
        # sample [batch_size] beauty rated according to the given images
        batch_ranking = []
        
        for i in idx:
            batch_ranking.append(self.beauty_rates[i-1])
        batch_ranking = np.array(batch_ranking)
        
        # TODO: check if needed
        if level is not None:
            if level != int(level):
                min_lw, max_lw = int(level+1)-level, level-int(level)
                lr_key = self._base_key + '{}x{}'.format(size//2, size//2)
                low_resol_batch_x = np.array([self.dataset[lr_key][i]/127.5-1.0 for i in idx], dtype=np.float32).repeat(2, axis=2).repeat(2, axis=3)
            # batch_x = batch_x * max_lw + low_resol_batch_x * min_lw
        
        # print("image index: {}".format(idx[0]))
        # print("beauty rates: {}".format(list(batch_ranking[0])))

        return batch_x, batch_ranking
    
    def save_imgs(self, samples, file_name):
        N_samples, channel, height, width = samples.shape
        N_row = N_col = int(np.ceil(N_samples**0.5))
        combined_imgs = np.ones((channel, N_row*height, N_col*width))
        for i in range(N_row):
            for j in range(N_col):
                if i*N_col+j < samples.shape[0]:
                    combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
        combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
        scipy.misc.imsave(file_name+'.png', combined_imgs)

class CelebA():
    def __init__(self):
        datapath = os.path.join('./datasets', 'CelebA/img_celeba')
        self.channel = 3
        self.data = glob(os.path.join(datapath, '*.jpg'))
    
    def __call__(self, batch_size, size):
        batch_number = len(self.data)/batch_size
        path_list = [self.data[i] for i in np.random.randint(0, len(self.data), size=batch_size)]
        file_list = [p.split('/')[-1] for p in path_list]
        batch = [get_img(img_path, True, 178, size, True) for img_path in path_list]
        batch_imgs = np.array(batch).astype(np.float32)
        batch_ranking = np.array([]).astype(np.float32) # CelebA is untagged with beuty rates
        return batch_imgs, batch_ranking
    
    def save_imgs(self, samples, file_name):
        N_samples, channel, height, width = samples.shape
        N_row = N_col = int(np.ceil(N_samples**0.5))
        combined_imgs = np.ones((channel, N_row*height, N_col*width))
        for i in range(N_row):
            for j in range(N_col):
                if i*N_col+j < samples.shape[0]:
                    combined_imgs[:,i*height:(i+1)*height, j*width:(j+1)*width] = samples[i*N_col+j]
        combined_imgs = np.transpose(combined_imgs, [1, 2, 0])
        scipy.misc.imsave(file_name+'.png', combined_imgs)


class RandomNoiseGenerator():
    def __init__(self, size, noise_type='gaussian'):
        self.size = size
        self.noise_type = noise_type.lower()
        assert self.noise_type in ['gaussian', 'uniform']
        self.generator_map = {'gaussian': np.random.randn, 'uniform': np.random.uniform}
        if self.noise_type == 'gaussian':
            self.generator = lambda s: np.random.randn(*s)
        elif self.noise_type == 'uniform':
            self.generator = lambda s: np.random.uniform(-1, 1, size=s)

    def __call__(self, batch_size):
        return self.generator([batch_size, self.size]).astype(np.float32)
