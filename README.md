### Handwritten Digit Recognizer By Team 42
This Handwritten Digit Recognizer is a GUI which will recognize numbers 0-9 when handwritten in the GUI using the Lenet-5 Model which has been trained by using the MNIST dataset which has been imported.
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


## DNN MODEL

The DNN model can be found under the project Code

