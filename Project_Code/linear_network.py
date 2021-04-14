import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520, bias=False)
        self.l2 = nn.Linear(520, 320, bias=False)
        self.l3 = nn.Linear(320, 240, bias=False)
        self.l4 = nn.Linear(240, 120, bias=False)
        self.l5 = nn.Linear(120, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)