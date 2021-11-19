#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

Authors:
Abrar Amin   z5018626
Demiao Chen  z5289988
group id     g025514

submssion accuracy: 92.5%

Date: Nov.19 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

    See hw2.pdf
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    """
    # Data Augmentation
    if mode == 'train':
        return transforms.Compose(
            [   
                transforms.RandomResizedCrop(size=80, scale=(0.55, 1.0), ratio=(0.75, 1.3)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(p=0.2),
                transforms.RandomAffine(degrees=(-15, 15), translate=(0.0, 0.5)),
                transforms.RandomRotation((-10,10)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1, hue=0.02),
                transforms.RandomPosterize(bits=3, p=0.3),
                transforms.RandomEqualize(p=0.1),
                transforms.RandomGrayscale(p=0.01),
                transforms.RandomPerspective(distortion_scale=0.05, p=0.1, fill=0),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.ToTensor()
            ]
        )
    # Keep the testing data original to ensure accuracy
    elif mode == 'test':
        return transforms.Compose(
            [   
                transforms.ToTensor()
            ]
        )

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ######### block 1 #########
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            
            
            ######### block 2 #########
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            

            ######### block 3 #########   
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
        
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            
            
            ######### block 4 #########
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ELU(inplace=True),
            
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )
        
        # shrink final conv layer width to 3
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # flatten from conv layers

            nn.Dropout(p=0.3),
            nn.Linear(512*3*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Dropout(p=0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        
            nn.Dropout(p=0.3),
            nn.Linear(512, 8)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.fc_layers(x)
        return x

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
learning_rate = 0.0005
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################
def weights_init(m):
    return

scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1  # 1 for submssion, 0.8 for testing to select the best network
batch_size = 256 
epochs = 1500  # training and validation accuracy converge around 400 epochs
               # later epochs only slightly improved the testing accuracy
