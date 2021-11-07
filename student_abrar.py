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
    if mode == 'train':
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(60),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4817, 0.4347, 0.3928], std=[0.2483, 0.2402, 0.2335])
            ]
        )
    elif mode == 'test':
        return transforms.Compose(
            [   
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(60),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4817, 0.4347, 0.3928], std=[0.2483, 0.2402, 0.2335])
            ]
        )

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(30, 120, kernel_size=5)
        self.conv3 = nn.Conv2d(120, 360, kernel_size=3)
        #self.conv4 = nn.Conv2d(360, 540, kernel_size=3)
        self.fc1 = nn.Linear(360*5*5, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 8) # output layer
        # Dropout
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        ###############
        # CONV Layers #
        ###############
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv3(out)), (2, 2))
        #out = F.max_pool2d(F.relu(self.conv4(out)), (2, 2))
        #print(out.shape)
        out = out.view(out.size(0), -1)
        
        ##########################
        # Fully Connected Layers #
        ##########################
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        
        ################
        # Output Layer #
        ################
        predicted_output = F.log_softmax(self.fc3(out), dim=1)
        return predicted_output

net = Network()
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
learning_rate = 0.001
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# loss_func = F.nll_loss
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
train_val_split = 0.8
batch_size = 200
epochs = 10
