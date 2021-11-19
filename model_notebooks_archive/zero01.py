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
import torchvision.transforms as T

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
        return T.Compose(
            [   
                T.RandomHorizontalFlip(),
                T.RandomRotation((-10,10)),
                T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2),
                T.RandomPosterize(bits=3, p=0.4),
                T.RandomEqualize(p=0.1),
                T.RandomGrayscale(p=0.1),
                T.RandomPerspective(distortion_scale=0.05, p=0.1, fill=0),
                # T.RandomErasing(),
                # T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                # T.RandomInvert(p=0.05),
                T.ToTensor(),
                # Standardize each channel of the image
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif mode == 'test':
        return T.Compose(
            [   
                T.ToTensor(),
                # Standardize each channel of the image
                # transforms.Normalize([0.485, 0.456, 0.406],
                #                                 [0.229, 0.224, 0.225]),
            ]
        )


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 60, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(60, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.45), # reduce overfitting
            nn.Linear(32*19*19, 960),
            nn.ReLU(),

            nn.Dropout(p=0.45), # reduce overfitting
            nn.Linear(960 , 800),
            nn.ReLU(),

            nn.Linear(800 , 8),
        )
        
    def forward(self, x):
        # print(x.shape)
        x = self.cnn_layers(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr=0.0008)

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
batch_size = 64
epochs = 100
