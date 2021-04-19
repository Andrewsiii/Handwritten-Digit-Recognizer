from __future__ import print_function
import torch
import torchvision
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io; io.use_plugin('matplotlib')
from skimage import data, io, filters
from PIL import Image, ImageOps
import os
#training settings
batch_size = 64 


#Mnist dataset
trainset = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 


testset = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    

#Data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=True)


from neural_network import LeNet5
#from linear_network import Net

#initialisations
model = LeNet5()
#model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00035, momentum=0.5)

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

working_dir = os.getcwd()
data_dir = (working_dir + '\\testdata')
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])

def open_image(directory):
    #directory = 'C://Users//jhpau//testdata/5/5'
    image = Image.open(directory)

    image = ImageOps.grayscale(image)
    image = np.array(image)
    image = resize((image), (28,28), anti_aliasing= True)

    # plt.imshow(image)
    # plt.show()

    image = test_transforms(image).float()
    image = image.unsqueeze(1)

    return image


model.load_state_dict(torch.load('model_weights.pth'))

#running the training
def training_and_testing():
    model.eval()
    test()
    for epoch in range (1, 16):
        train(epoch)
        test()

# torch.save(model.state_dict(), 'model_weights.pth')

def prediction(image):
    model.eval()
    output = model(image)

    index = output.data.cpu().numpy().argmax()
    #print(index)



    probabilities = torch.exp(output)
    #print(probabilities)

    
    #output = output.detach().numpy()
    #print(probabilities)
    probabilities = probabilities.detach().numpy()
    x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # x = np.arange(len(x_label))
    # y = list(probabilities)
    
    #print(probabilities)
    
    y = np.append(probabilities[0,[0]], probabilities[0,[1]])
    y = np.append(y, probabilities[0,[2]])
    y = np.append(y, probabilities[0,[3]])
    y = np.append(y, probabilities[0,[4]])
    y = np.append(y, probabilities[0,[5]])
    y = np.append(y, probabilities[0,[6]])
    y = np.append(y, probabilities[0,[7]])
    y = np.append(y, probabilities[0,[8]])
    y = np.append(y, probabilities[0,[9]])
    
    plt.bar(x,y)
    plt.xlabel('Digits')
    plt.ylabel('probability')
    plt.title('Classified Digit:' + str(index))
    #plt.show()
    return x, y, index

def viewTrainDataset():
    y, _ = trainset[2]
    to_pil = transforms.ToPILImage()
    image = to_pil(y)
    image.show()

def viewTestDataset():
    x, _ = testset[0]
    to_pil = transforms.ToPILImage()
    im = to_pil(x)
    im.show()

# model.eval()
# test()
# training_and_testing()
# torch.save(model.state_dict(), 'model_weights.pth')
# image = open_image('C://Users//jhpau//testdata/9/9.png')
# prediction(image)