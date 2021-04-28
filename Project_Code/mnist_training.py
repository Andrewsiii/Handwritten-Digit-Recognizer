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
import Project_Gui
from neural_network import LeNet5


#training settings
batch_size = 64 


#Data handling
#Downloading, transforming into tensor and loading the MNSIT dataset shuffled into the loader
def Initial():
    trainset = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 


    testset = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=True)


#model initialisations
model = LeNet5()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.5)



#training function
#Batches of the MNIST data that is loaded into the trainset is passed through the model and the weights are adjusted using the back propagation function.
#loss is also calculated and displayed together with the progress of the training in a print function
def train(epoch):
    trainset = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 


    testset = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=True)
    model.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:    #display progress of training in each epoch
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))   

#testing function
#passes the testset MNIST data through the model while model is in Evaluation mode so weights are not adjusted
#Loss is calculated together with whether the model predicted correct
def test():
    trainset = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 


    testset = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=True)
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



    test_loss /= len(testloader.dataset)    # Loss and number of correct predictions are displayed together with model accuracy for that test in print function
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} '
          f'({100. * correct / len(testloader.dataset):.0f}%)')


working_dir = os.getcwd()   #defining the working directory
data_dir = (working_dir + '\\testdata')
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))]) #transform to tensor function.


#function that opens and processes an image so that it can pass through the model for recognition
def open_image(directory):

    image = Image.open(directory)

    image = ImageOps.grayscale(image) #transform to grayscale image
    image = np.array(image)             #transform to numpy array
    image = resize((image), (28,28), anti_aliasing= True)   #process the image to a 28x28 image with anti aliasing 



    image = test_transforms(image).float()
    image = image.unsqueeze(1)              #unsqueeze image dim 1

    return image


#loading the pretrained model
def Loading():
    model.load_state_dict(torch.load('model_weights.pth'))


#running the training for 10 epochs and evaluating after each epoch to display model accuracy during training.
def training_and_testing():
    model.eval()
    for epoch in range (1, 11):
        train(epoch)
        test()


#function that takes a processed image as an input and passes it through the model
#returns the classes on the x-axis, probabilites of prediction on y-axis and the most dominant class as the 'index'
def prediction(image):
    model.eval()
    output = model(image)

    index = output.data.cpu().numpy().argmax()
   
    probabilities = torch.exp(output)
 
    probabilities = probabilities.detach().numpy()
    x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    
    #transforming the tensor of probabilties from the softmax output to a numpy array so it can be represented on a matplotlib graph
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

#function that cycles through and saves the MNIST training dataset as an individual pillow image.
def viewTrainDataset(i):
    trainset = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    
    y, _ = trainset[i]
    to_pil = transforms.ToPILImage() #function that transform tensor to pillow image
    image = to_pil(y)
    image.save('training_image.png')


#function that cycles through and saves the MNIST testing dataset as an individual pillow image.
def viewTestDataset(j):
    testset = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    
    x, _ = testset[j]
    to_pil = transforms.ToPILImage() #function that transform tensor to pillow image
    im = to_pil(x)
    im.save('testing_image.png')

#function that connects to train button in GUI which trains the model and saves the trained model.
def TrainingButton():
    model.eval()
    test()
    training_and_testing()
    torch.save(model.state_dict(), 'model_weights.pth')


#function used to calculate and display the precision, recall and F1 score for the trained model.
def analysis():
    testset = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
    testloader = torch.utils.data.DataLoader(testset, batch_size= 1, shuffle=True)
    model.eval()

    count0 = 0  #True positive + False positive count
    count1 = 0  #true positive count
    total = 0   #True positive + False negative count
    for data, target in testloader:
        output = model(data)


        pred = output.data.max(1, keepdim=True)[1]

        predict = output.data.cpu().numpy().argmax()
        
        #loop that extracts the true positive, false positive and false negative values for a single class
        if predict == 9:
            count0 = count0 + 1
            if pred.eq(target.data.view_as(pred)).cpu().sum():
                count1 = count1 + 1
        if target.data.numpy() == [9]:
            total = total + 1

    #Calculation of the statistical analysis values.
    precision = (count1/count0)
    recall = count1/total
    f1 = (2*precision*recall)/(precision+recall)

    print(precision)
    print(recall)
    print(f1)

