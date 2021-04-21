import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, bias = False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias = False)
        self.fc1 = nn.Linear(512, 256, bias = False)
        self.fc2 = nn.Linear(256, 64, bias = False)
        self.fc3 = nn.Linear(64, 10, bias = False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.avepooling = nn.AvgPool2d(2)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.avepooling(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.avepooling(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)

        #out = self.dropout(out)

        out  = self.fc2(out)
        out = self.relu(out)


        out = self.fc3(out)
        out = self.softmax(out)

        return out


