## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # convolutional layer (sees 224x224x5  image tensor)
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) #224/2 = 112
        # convolutional layer (sees 224x224x5  image tensor)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)#112/2 = 56
        # convolutional layer (sees 224x224x3  image tensor)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)#56/2=28
         # convolutional layer (sees 224x224x3  image tensor)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)#28/2 = 14
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)#14/2 = 7
        
        ##_______##^##_____##
        ##_______##^##_____##
        ##_______##^##_____##
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2) 
        # linear layer (512 * 7 *7 -> 600)
        self.fc1 = nn.Linear( 25088, 600)
        # linear layer (600 -> 136)
        self.fc2 = nn.Linear(600, 136)
        # dropout layer (p=0.36)
        self.dropout = nn.Dropout(0.2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        #x = self.pool(F.relu(self.conv5(x)))
        # flatten image input
        x = x.view(-1, 25088)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
