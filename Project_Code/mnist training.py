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

#training settings
batch_size = 32


#Mnist dataset
trainset = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 


testset = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    

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


data_dir = 'C://Users//jhpau//testdata'
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])

def predict_image(image):
    input = test_transforms(image).float()

    input = input.unsqueeze(1)


    output = model(input)

    index = output.data.cpu().numpy().argmax()

    return index

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

#running the training
#test()
#for epoch in range (1, 2):
    #train(epoch)
    #test()

to_pil = transforms.ToPILImage()
images, labels = get_random_images(5)
fig=plt.figure(figsize=(10,10))
data = datasets.ImageFolder(data_dir, transform=test_transforms)
classes = data.classes
for ii in range(len(images)):
    image = to_pil(images[ii])
    #image = torchvision.transforms.functional.to_grayscale(image)
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()