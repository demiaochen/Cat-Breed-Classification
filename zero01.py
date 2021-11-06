#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

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
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """

    # channel size = 3

    if mode == 'train':
        return transforms.Compose(
            [   
                # ref: https://d2l.ai/chapter_computer-vision/kaggle-dog.html

                # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
                # the original area and height-to-width ratio between 3/4 and 4/3. Then,
                # scale the image to create a new 224 x 224 image
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                        ratio=(3.0 / 4.0, 4.0 / 3.0)),
                transforms.RandomHorizontalFlip(),
                # Randomly change the brightness, contrast, and saturation
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                saturation=0.4),
                # Add random noise
                transforms.ToTensor(),
                # Standardize each channel of the image
                transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
            ]
        )
    elif mode == 'test':
        return transforms.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(512, 800, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(800, 400, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(400, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(200, 150, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(150*13*13, 5200),
            nn.ReLU(),

            nn.Linear(5200 , 4800),
            nn.ReLU(),

            nn.Linear(4800 , 2400),
            nn.ReLU(),

            nn.Linear(2400 , 300),
            nn.ReLU(),

            nn.Linear(300 , 8),
        )
        
    def forward(self, input):
        #print(input.shape)
        x = self.cnn_layers(input)
        #print(x.shape)
        x = self.fc_layers(input)
        return x

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr = 0.05)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8  # 0.8 for testing, 1.0 for final submitting
batch_size = 32
epochs = 100
