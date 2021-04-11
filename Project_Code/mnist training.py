from __future__ import print_function
import torch
import torchvision
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

#training settings
batch_size = 32


#Mnist dataset
trainset = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 


testset = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    

#Data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=True)

from neural_network import LeNet5

#initialisations
model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))

#testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} '
          f'({100. * correct / len(testloader.dataset):.0f}%)')


#running the training
test()
for epoch in range (1, 5):
    train(epoch)
    test()