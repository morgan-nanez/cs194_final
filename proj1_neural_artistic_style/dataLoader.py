# imports
import numpy as np

import os 
import torch
import pandas as pd
from skimage import io, transform, color
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

import torch.nn as nn
import torch.nn.functional as F
import cv2

from PIL import Image

import functools
import math
import random

import skimage

import tensorflow as tf

#adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class PhotoDataset(Dataset):
    """PHOTO dataset."""

    def __init__(self, root_dir, image_names, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_name_frame = image_names
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_name_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_name_frame[idx])

       # grey_im = io.imread(img_name)
        grey_im = Image.open(img_name)
        
        #convert images to grey scale
        #grey_im = color.rgb2gray(image)
        

        sample = {'image':  grey_im}

        if self.transform:
            for t in self.transform:
                sample = t(sample)


        return sample
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return {'image': image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        s = image.shape
        image = image.reshape((s[2], s[0], s[1]))
       # image = np.transpose(image, (2, 0, 1))

        return {'image': torch.from_numpy(image)}
    
class ToGreyNormalize(object):
    def __call__(self, sample):
        image = sample['image']
        
        
        # convert the image into grayscale
        grey_im = color.rgb2gray(image)
        
        #normalized float values in range -0.5 to 0.5 
        grey_im = grey_im - 0.5
        
        return {'image': grey_im}
    
#abstarcted from https://www.programcreek.com/python/?CodeExample=color+jitter
class ColorJitter(object):
    
    def __init__(self, values):
        self.brightness = values[0]
        self.contrast = values[1]
        self.saturation = values[2]
        self.hue = values[3]
            
    def __call__(self,sample):
        image = sample['image']
        
        tforms = []
        if self.brightness > 0:
            tforms.append(functools.partial(tf.image.random_brightness, max_delta=self.brightness))

        if self.contrast > 0:
            tforms.append(functools.partial(tf.image.random_contrast, lower=max(0, 1 - self.contrast), upper=1 + self.contrast))

        if self.saturation > 0:
            tforms.append(functools.partial(tf.image.random_saturation, lower=max(0, 1 - self.saturation), upper=1 + self.saturation))

        if self.hue > 0:
            tforms.append(functools.partial(tf.image.random_hue, max_delta=self.hue))

        random.shuffle(tforms)

        for tform in tforms:
            image = tform(image)

        return {'image': image, 'landmarks': landmarks} 
    
class Rotate(object):
               
    def __call__(self,sample):
        
        angle = random.uniform(-10, 10)
        
        image, landmarks = sample['image'], sample['landmarks']
        
        rot_im = skimage.transform.rotate(image, angle)
        
        n, m =  image.shape[0] / 2 - 0.5, image.shape[1] / 2 - 0.5
        
        angle = np.radians(angle)
        cosr = np.cos(angle)
        sinr = np.sin(angle)

        new_landmarks = landmarks.copy()
        new_landmarks[:,1] = cosr*(landmarks[:,1]-m) + (landmarks[:,0]-n)*sinr + m
        new_landmarks[:,0] = -sinr*(landmarks[:,1]-m) + cosr*(landmarks[:,0]-n) + n
        
        return {'image': rot_im, 'landmarks': new_landmarks} 