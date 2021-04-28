import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class LeNet5(nn.Module):
    def __init__(self):                 # function that defines the layers and its parameters in the neural network
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, bias = False)  #first convolutional layer that takes 1 input and returns 16 outputs. consists of a 5x5 kernel with no biasing
        self.conv2 = nn.Conv2d(16, 32, 5, bias = False) #second convolutional layer that takes 16 input and returns 32 outputs. consists of a 5x5 kernel with no biasing
        self.fc1 = nn.Linear(512, 256, bias = False)    #first fully connected layer that takes 512 inputs and returns 256 outputs with no biasing
        self.fc2 = nn.Linear(256, 64, bias = False)     #second fully connected layer that takes 256 inputs and returns 64 outputs with no biasing
        self.fc3 = nn.Linear(64, 10, bias = False)      #final fully connected layer that takes 64 inputs and returns 10 outputs matching the 10 output classes with no biasing
        self.relu = nn.ReLU()               #Relu activation function
        self.dropout = nn.Dropout(0.25)     #dropout layer that drops out 25% of the neurons
        self.avepooling = nn.AvgPool2d(2)   #2D averaging pooling layer
        self.softmax = nn.LogSoftmax(-1)    #Logsoftmax activation function that returns the probabilty distribution of the outcome

    def forward(self, input):           #function that displays the architecture of the model
        out = self.conv1(input)
        out = self.relu(out)
        out = self.avepooling(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.avepooling(out)

        out = out.view(out.size(0), -1)     #flattening the data so it can be passed through the fully connected linear layers.

        out = self.fc1(out)
        out = self.relu(out)


        out  = self.fc2(out)
        out = self.relu(out)


        out = self.fc3(out)
        out = self.softmax(out)

        return out


