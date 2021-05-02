### Handwritten Digit Recognizer By Team 42
The goal of the project was to design a deep neural network model and create a programme that allows a user to input a handwritten digit (0 to 9) onto a drawing canvas where the programme would then take the input and classify it into one of 10 classes (0 to 9) and show the probability of the prediction. The DNN model was to be trained and validated with the MNIST dataset which needed to be imported using TorchVision.

<img width="576" alt="302 system diagram" src="https://user-images.githubusercontent.com/68927814/116801612-91b91200-ab5f-11eb-9fb5-1bdc338bfcd0.png">

## Model Versions
The first version of the model was just the generic Lenet-5 architecture but later on batch normalisation layers and dropout layers were added.
The parameters such as learning rate, batch size and number of epochs were also varied while trying to discover the most optimum model.

## Installation
The installation requirements are written under requirements.txt.Please download these libraries before running this project.

Note:
Please ensure the current working directory is the same as where the project files are. The MNIST dataset will be downloaded to the current working directory

## Usage

First, run the Project_GUI.py file  

When the code is run, the user will see a Main Window as show below.

![image](https://user-images.githubusercontent.com/68887738/116235621-33cca900-a7b2-11eb-99a4-7826961d27e7.png)

The User will first go into the main menu bar under file -> train model. A window should appear like the one below.

![image](https://user-images.githubusercontent.com/68887738/116236247-f0266f00-a7b2-11eb-9c61-75d91c98ef0c.png)

This is the train Model window. It is here in which the user will be able to download the MNIST dataset and also train the model for the GUI to use to recognize the digits written in the canvas.The MNIST download button will only work if MNIST does not exist in the current working directory. The train model will create a file called model_weights.pth when it is done. This will appear in the current working directory. Once both have been completed, the user may open up the Canvas window found under the the file menubar in the main window under drawing canvas.

A canvas window should appear like the one below.

![image](https://user-images.githubusercontent.com/68887738/116237155-f9640b80-a7b3-11eb-8e9d-4fb2d5b2bfe9.png)

The user will be able to draw any digit between 0-9 on the canvas on the left of the window. Once done, the user can press recognize and the program will recognize the digit. If done correctly there should be no error messages. However, if the model is not trained or the MNIST dataset is not downloaded the error messages will appear. Please refer to the instructions for the train model window. 

The image below shows result if done correctly.

![image](https://user-images.githubusercontent.com/68887738/116257240-fc68f700-a7c7-11eb-9d42-b32002ea77f2.png)

This probability chart will show the probability of each digit and also show what digit it has recognized from the user canvas

## Image viewer
Another feature that our GUI includes is that it can show the images within the MNIST dataset. This feature can be found in the main window under view menubar. The user will have the option of choosing viewing either the testing images or the training images. A image viewer window will appear and show the digit and the user will be able to browse using the show next button. The image viewer is shown below.

![image](https://user-images.githubusercontent.com/68887738/116258358-fd4e5880-a7c8-11eb-90a7-6d8efa1e5efe.png)

