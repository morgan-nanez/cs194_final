#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

#adapted from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class NoseNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        
        #3-4 convolution layers
        #requirements:
            # 12-32 channels each
            # kernel size 7x7, 5x5 or 3x3
            
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 12, kernel_size = 7)
        self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 16, kernel_size = 5)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)


        #Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        # 2 Fully Connected layers
        self.fc1 = nn.Linear(in_features = 576, out_features = 200)  # from torch.Size([32, 3, 6])
        self.fc2 = nn.Linear(in_features = 200, out_features = 2)    # 120 = 60 * 2 (num features)
        


    def forward(self, x):

        #Each convolutional layer will be followed by a ReLU followed by a maxpool.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
       # print(x.shape)
        
        
        #have to flatten x
        
        x = x.view(-1)
        #x = x.flatten()
        #print(x.shape)
        
        #2 fully connected layers
        x = self.fc1(x)
        
        #Apply ReLU after the first fully connected layer
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
#adapted from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class FaceNet(nn.Module):

    def __init__(self):
        super().__init__()

        
        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 7)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 7)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5)
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 56, kernel_size = 3)
        self.conv6 = nn.Conv2d(in_channels = 56, out_channels = 112, kernel_size = 2)
        
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 336, out_features = 200) # from size of x flatten
        self.fc2 = nn.Linear(in_features = 200,    out_features = 112) # 56*2

        
        
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv6(x)
        x = F.relu(x)


        x = x.view(-1)

        #2 fully connected layers
        x = self.fc1(x)
        
        x = F.relu(x)

        x = self.fc2(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

