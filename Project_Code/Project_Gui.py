import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QGridLayout, QPushButton, QLineEdit, QTextEdit, QLabel,QVBoxLayout,QWidget,QHBoxLayout, QMenu, QTabWidget, QMdiArea, QSizePolicy, QProgressBar, QTextBrowser
from PyQt5.QtCore import QCoreApplication, Qt, QProcess
from PyQt5.QtGui import QIcon, QPen, QPixmap, QPainter, QImage, QPainterPath, QFont
import mnist_training
import neural_network
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os 
import torch
from os import path



class ViewTrainingData(QLabel):  #Image viewer for the training data
    
    def __init__(self):
        super().__init__()
        self.title = 'Training Images'

        self.initUI()
    
    def initUI(self):       # initialisations
        self.setWindowTitle(self.title)
        self.setGeometry(800,300,400,300)
        self.label = QLabel(self)
        self.label1 = QLabel(self)
        self.data = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
        self.digit = torch.utils.data.DataLoader(self.data, shuffle=False)
        self.k = 0

        self.button1 = QPushButton("show next", self)
        self.button1.setGeometry(300,235,100,30)
        self.button1.clicked.connect(self.display_next)

        self.display_next()  

    
    def display_next(self):    #This function will display the next image in the training dataset
        y, _ = self.data[self.k]
        self.string = str(self.digit.dataset.targets[self.k])
        self.string1 = self.string.replace('tensor', '')
        to_pil = transforms.ToPILImage()
        image = to_pil(y)
        image.save('training_image.png')
        root_dir = os.getcwd()
        grid = QGridLayout()
        self.label.setLayout(grid)
        pixmap = QPixmap(root_dir + '\\training_image.png')
        pixmap = pixmap.scaledToWidth(300)
        self.label.setPixmap(pixmap)
        self.label.setGeometry(0,0,300,300)
        self.label1.setText(self.string1)
        self.label1.setFont(QFont('Times font',20, italic = False))
        self.label1.setGeometry(300,0,50,50)
        self.k = self.k + 1


        
class ViewTestingData(QLabel):     # this class is the image viewer of the testing dataset
    
    def __init__(self):
        super().__init__()
        self.title = 'Testing Images'

        self.initUI()
    
    def initUI(self):        #initialisations of the image viewer
        self.setWindowTitle(self.title)
        self.setGeometry(800,300,400,300)
        self.label = QLabel(self)
        self.label1 = QLabel(self)
        self.data = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
        self.digit = torch.utils.data.DataLoader(self.data, shuffle=False)
        self.k = 0

        self.button1 = QPushButton("show next", self)
        self.button1.setGeometry(300,235,100,30)
        self.button1.clicked.connect(self.display_next)
        self.display_next()  

    
    def display_next(self):    # this function will display the next image in the testing dataset
        y, _ = self.data[self.k]
        self.string = str(self.digit.dataset.targets[self.k])
        self.string1 = self.string.replace('tensor', '')
        to_pil = transforms.ToPILImage()
        image = to_pil(y)
        image.save('testing_image.png')
        root_dir = os.getcwd()
        grid = QGridLayout()
        self.label.setLayout(grid)
        pixmap = QPixmap(root_dir + '\\testing_image.png')
        pixmap = pixmap.scaledToWidth(300)
        self.label.setPixmap(pixmap)
        self.label.setGeometry(0,0,300,300)
        self.label1.setText(self.string1)
        self.label1.setFont(QFont('Times font',20, italic = False))
        self.label1.setGeometry(300,0,50,50)
        self.k = self.k + 1 

class Plot(FigureCanvas):     
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(5,4), dpi=200)
        super().__init__(fig)
        self.setParent(parent)

        

    def graph(self):   # this will display the graph
        
        root_dir = os.getcwd()
        image = mnist_training.open_image(root_dir + '\\data.png')
        mnist_training.Initial()
        mnist_training.Loading()
        x, y, index = mnist_training.prediction(image)
        
        plt.bar(x,y)
        plt.xlabel('Digits')
        plt.ylabel('probability')
        plt.title('Classified Digit: ' + str(index))

class Canvas(QLabel):    #This class is for the canvas widget that the user draws

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_StaticContents)
        h = 375
        w = 375
        self.myPenWidth = 60
        self.myPenColor = Qt.white
        self.image = QImage(w, h, QImage.Format_RGB32)
        self.path = QPainterPath()
        self.clearImage()

    def setPenColor(self, newColor):    #Change the colour of the pen
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):   #Change the width of the pen
        self.myPenWidth = newWidth

    def clearImage(self):   #Clears the image on the canvas by filling it with a black background
        self.path = QPainterPath()
        self.image.fill(Qt.black)
        self.update()



    def paintEvent(self, event):    #triggers when an event happens
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mousePressEvent(self, event):   #this function will trigger when the mouse is pressed
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):  #this function will trigger when the mouse moves and draws a line
        self.path.lineTo(event.pos())
        p = QPainter(self.image)
        p.setPen(QPen(self.myPenColor,
                      self.myPenWidth, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        p.drawPath(self.path)
        p.end()
        self.update()

    def saveImage(self):    # This function is to save the image on the canvas
        root_dir = cwd = os.getcwd()
        self.image.save('data.png')
        
    
class CanvasWindow(QMainWindow):   #This class is for the canvas window which contains buttons and canvas widget

    def __init__(self,parent = None):
        super(CanvasWindow,self).__init__(parent,Qt.Window)
        self.canvas = Canvas()
        canvas = QWidget()
        grid =QGridLayout()
        canvas.setLayout(grid)
        grid.addWidget(self.canvas,0,1)
        Recognize = QPushButton('&Recognize', self)
        Recognize.clicked.connect(self.GraphShow)

        Clear = QPushButton('&Clear', self)
        Clear.clicked.connect(self.canvas.clearImage)

        

        Model = QPushButton('&Model', self)
        menu = QMenu(self)
        menu.addAction('Lenet Model')
        Model.setMenu(menu)
         
        #Make 1 Widget for all the buttons
        
        ButtonWidget = QWidget()
        vbox = QVBoxLayout()
        vbox.addWidget(Recognize)
        vbox.addWidget(Model)
        vbox.addWidget(Clear)
     
        ButtonWidget.setLayout(vbox)
        vbox.addStretch(1)
        grid.addWidget(ButtonWidget,0,2)
        self.setCentralWidget(canvas)    
        self.setWindowTitle('Canvas')
        self.setGeometry(300, 200, 800, 400)
    
    def GraphShow(self):   #This function is for the graph to show, only when MNIST dataset has been downloaded and the model have been trained
        if(path.exists("model_weights.pth") == True) and (path.exists("MNIST") == True):
            self.canvas.saveImage()
            self.show_plot = Plot(self)
            self.show_plot.graph()
            self.newwindow = Graph(self)
            self.newwindow.show()
        elif (path.exists("model_weights.pth") == False) and (path.exists("MNIST") == False):
            ErrorBox = QMessageBox()
            ErrorBox.setIcon(QMessageBox.Information)
            ErrorBox.setText("Error: Please Download MNIST and then train the model")
            ErrorBox.setWindowTitle("Error")
            ErrorBox.setStandardButtons(QMessageBox.Ok)
            ErrorBox.buttonClicked.connect(ErrorBox.close)
            ErrorBox.exec()  
        elif(path.exists("model_weights.pth") == True) and (path.exists("MNIST") == False):
            ErrorBox = QMessageBox()
            ErrorBox.setIcon(QMessageBox.Information)
            ErrorBox.setText("Error: Please Download MNIST first")
            ErrorBox.setWindowTitle("Error")
            ErrorBox.setStandardButtons(QMessageBox.Ok)
            ErrorBox.buttonClicked.connect(ErrorBox.close)
            ErrorBox.exec()
        else:
            ErrorBox = QMessageBox()
            ErrorBox.setIcon(QMessageBox.Information)
            ErrorBox.setText("Error: Please train the model first")
            ErrorBox.setWindowTitle("Error")
            ErrorBox.setStandardButtons(QMessageBox.Ok)
            ErrorBox.buttonClicked.connect(ErrorBox.close)
            ErrorBox.exec() 
        

class Graph(QMainWindow):     # This class is for the window of the graph

    def __init__(self,parent=None):
        super(Graph,self).__init__(parent,Qt.Window)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Probability chart')
        self.setGeometry(500, 50, 900, 900)
        

        chart = Plot(self)
        chart.graph()
        

        
class MyApp(QMainWindow):   # This is the class for the main window which is initialised

    def __init__(self,parent=None):
        super(MyApp,self).__init__(parent)
        self.initUI()

    def initUI(self):   #Anything under the GUI/first window is here 
        exitAction = QAction('Quit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        self.lbl = QLabel('    Handwritten Digit\nRecognition Programme', self)    #The label on the main window
        self.lbl.setGeometry(200, 150,600,300)
        self.lbl.setFont(QFont('Times font',20, italic = False))

        self.lbl2 = QLabel('by Paul Kim & Andrew Sio\nCompsys 302 2021', self)
        self.lbl2.setGeometry(580, 390,600,300)
        self.lbl2.setFont(QFont('Times font',8, italic = True))

        TrainModel = QAction ('Train Model', self)
        TrainModel.setShortcut('Ctrl+T')
        TrainModel.setStatusTip('Train the model')
        TrainModel.triggered.connect(self.OpenWindow)
        

        viewTrainingImages = QAction ('View Training Images', self)
        viewTrainingImages.setStatusTip('View the training images')  #to test use triggered function like drawing canvas
        viewTrainingImages.triggered.connect(self.view_training)

        viewTestingImages = QAction ('View Testing Images', self)
        viewTestingImages.setStatusTip('View the testing images')
        viewTestingImages.triggered.connect(self.view_testing)

        DrawingCanvas = QAction('Drawing Canvas',self)
        DrawingCanvas.setStatusTip('View the testing images')
        self.newwindow = CanvasWindow(self)
        DrawingCanvas.triggered.connect(self.CanvasClk)


        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(TrainModel)
        filemenu.addAction(DrawingCanvas)
        filemenu.addAction(exitAction)
        viewmenu = menubar.addMenu('&View')
        viewmenu.addAction(viewTrainingImages)
        viewmenu.addAction(viewTestingImages)
        self.setWindowTitle('Handwritten Digit Recognizer')
        self.setGeometry(300, 200, 800, 600)
        self.show()


    def OpenWindow(self):   #This function is for the dialog box when Train model is pressed. This creates 3 buttons.
        dialog = QDialog(self)
        dialog.setWindowTitle('Train Model')
        dialog.setGeometry(300, 300, 900, 500)
        grid = QGridLayout()
        dialog.setLayout(grid)
        btn1 = QPushButton('&Download MNIST', self)      #This creates the 3 buttons that download MNIST, Train and cancel button.
        btn1.clicked.connect(MyApp.MNIST_Download)
        btn2 = QPushButton(self)
        btn2.setText('Train')
        btn2.clicked.connect(mnist_training.TrainingButton)
        btn3 = QPushButton('cancel', self)
        btn3.clicked.connect(dialog.close)
        
        self.tb = QTextBrowser()
        self.tb.setAcceptRichText(True)
        self.tb.setOpenExternalLinks(True)
        grid.addWidget(self.tb,2,2)
        self.tb.append('click to start download or training')

        grid.addWidget(btn1,4,1)
        grid.addWidget(btn2,4,2)
        grid.addWidget(btn3,4,3)
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 240, 500, 25)
        self.pbar.setFormat('0')
        grid.addWidget(self.pbar,3,2)
        dialog.show()
    def CanvasClk(self):     #this function is to open the canvas window
        self.newwindow.show()
    def MNIST_Download(self):  #This function will download the MNIST dataset if the dataset has not been downloaded
        if(path.exists("MNIST") == True):
            ErrorBox = QMessageBox()
            ErrorBox.setIcon(QMessageBox.Information)
            ErrorBox.setText("Error: MNIST already downloaded")
            ErrorBox.setWindowTitle("Error")
            ErrorBox.setStandardButtons(QMessageBox.Ok)
            ErrorBox.buttonClicked.connect(ErrorBox.close)
            ErrorBox.exec()
        else:
            dataset = datasets.MNIST(root='', download=True)

    def view_training(self):   #function to view the training data
        self.train = ViewTrainingData()
        self.train.show()
    def view_testing(self):  #function to view the testing data
        self.test = ViewTestingData()
        self.test.show()
    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
