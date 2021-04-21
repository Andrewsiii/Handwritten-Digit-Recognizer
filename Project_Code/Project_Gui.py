import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QGridLayout, QPushButton, QLineEdit, QTextEdit, QLabel,QVBoxLayout,QWidget,QHBoxLayout, QMenu
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPen, QPixmap, QPainter, QImage, QPainterPath
import mnist_training
import neural_network
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os 
from os import path


class ViewTrainingData(QLabel):
    
    def __init__(self):
        super().__init__()
        self.title = 'Training Images'

        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(800,300,400,300)
        self.label = QLabel(self)
        self.data = datasets.MNIST(root='', train=True, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
        self.k = 0

        self.button1 = QPushButton("show next", self)
        self.button1.setGeometry(300,235,100,30)
        self.button1.clicked.connect(self.display_next)
        self.button2 = QPushButton("load next", self)
        self.button2.setGeometry(300,200,100,30)
        self.button2.clicked.connect(self.load_next)
        self.load_next()
        self.display_next()  

    
    def display_next(self):
        root_dir = os.getcwd()
        grid = QGridLayout()
        self.label.setLayout(grid)
        pixmap = QPixmap(root_dir + '\\training_image.png')
        pixmap = pixmap.scaledToWidth(300)
        self.label.setPixmap(pixmap)
        self.label.setGeometry(0,0,300,300)

    def load_next(self):
        y, _ = self.data[self.k]
        to_pil = transforms.ToPILImage()
        image = to_pil(y)
        image.save('training_image.png')
        self.k = self.k + 1

        
class ViewTestingData(QLabel):
    
    def __init__(self):
        super().__init__()
        self.title = 'Testing Images'

        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(800,300,400,300)
        self.label = QLabel(self)
        self.data = datasets.MNIST(root='', train=False, download=False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))])) 
        self.k = 0

        self.button1 = QPushButton("show next", self)
        self.button1.setGeometry(300,235,100,30)
        self.button1.clicked.connect(self.display_next)
        self.button2 = QPushButton("load next", self)
        self.button2.setGeometry(300,200,100,30)
        self.button2.clicked.connect(self.load_next)
        self.load_next()
        self.display_next()  

    
    def display_next(self):
        root_dir = os.getcwd()
        grid = QGridLayout()
        self.label.setLayout(grid)
        pixmap = QPixmap(root_dir + '\\testing_image.png')
        pixmap = pixmap.scaledToWidth(300)
        self.label.setPixmap(pixmap)
        self.label.setGeometry(0,0,300,300)

    def load_next(self):
        y, _ = self.data[self.k]
        to_pil = transforms.ToPILImage()
        image = to_pil(y)
        image.save('testing_image.png')
        self.k = self.k + 1    

class Plot(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(5,4), dpi=200)
        super().__init__(fig)
        self.setParent(parent)

        

    def graph(self):
        
        root_dir = os.getcwd()
        image = mnist_training.open_image(root_dir + '\\data.png')
        mnist_training.Initial()
        mnist_training.Loading()
        x, y, index = mnist_training.prediction(image)
        
        plt.bar(x,y)
        plt.xlabel('Digits')
        plt.ylabel('probability')
        plt.title('Classified Digit: ' + str(index))

class Canvas(QLabel):    #Canvas Widget itself

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

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(Qt.black)
        self.update()



    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        p = QPainter(self.image)
        p.setPen(QPen(self.myPenColor,
                      self.myPenWidth, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        p.drawPath(self.path)
        p.end()
        self.update()

    def saveImage(self):
        root_dir = cwd = os.getcwd()
        self.image.save('data.png')
        
    
class CanvasWindow(QMainWindow):   #The Canvas Window

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

        Random = QPushButton('&Random', self)

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
        vbox.addWidget(Random)
        ButtonWidget.setLayout(vbox)
        vbox.addStretch(1)
        grid.addWidget(ButtonWidget,0,2)
        self.setCentralWidget(canvas)    
        self.setWindowTitle('Canvas')
        self.setGeometry(300, 200, 800, 400)
    
    def GraphShow(self):
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
        

class Graph(QMainWindow):  

    def __init__(self,parent=None):
        super(Graph,self).__init__(parent,Qt.Window)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Probability chart')
        self.setGeometry(500, 50, 900, 900)
        

        chart = Plot(self)
        chart.graph()
        

        
class MyApp(QMainWindow):   # The GUI ITSELF

    def __init__(self,parent=None):
        super(MyApp,self).__init__(parent)
        self.initUI()

    def initUI(self):   #Anything under the GUI/first window is here 
        exitAction = QAction('Quit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        TrainModel = QAction ('Train Model', self)
        TrainModel.setShortcut('Ctrl+T')
        TrainModel.setStatusTip('Train the model')
        TrainModel.triggered.connect(self.OpenWindow)
        

        viewTrainingImages = QAction ('View Training Images', self)
        viewTrainingImages.setStatusTip('View the training images')  #to test use triggered function like drawing canvas
        self.view_training = ViewTrainingData()
        viewTrainingImages.triggered.connect(self.view_training.show)

        viewTestingImages = QAction ('View Testing Images', self)
        viewTestingImages.setStatusTip('View the testing images')
        self.view_testing = ViewTestingData()
        viewTestingImages.triggered.connect(self.view_testing.show)

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

    def OpenWindow(self):   #This function is for the dialog box when Train model is pressed. Need to update
        dialog = QDialog(self)
        dialog.setWindowTitle('Train Model')
        dialog.setGeometry(300, 300, 300, 200)
        grid = QGridLayout()
        dialog.setLayout(grid)
        btn1 = QPushButton('&Download MNIST', self)
        btn1.clicked.connect(MyApp.MNIST_Download)
        btn2 = QPushButton(self)
        btn2.setText('Train')
        btn2.clicked.connect(mnist_training.TrainingButton)
        btn3 = QPushButton('cancel', self)
        btn3.clicked.connect(dialog.close)
        
        grid.addWidget(btn1,3,1)
        grid.addWidget(btn2,3,2)
        grid.addWidget(btn3,3,3)
        dialog.show()
    def CanvasClk(self):
        self.newwindow.show()
    def MNIST_Download(self):
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
    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
