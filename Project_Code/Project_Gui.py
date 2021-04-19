import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QGridLayout, QPushButton, QLineEdit, QTextEdit, QLabel,QVBoxLayout,QWidget,QHBoxLayout, QMenu
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPen, QPixmap, QPainter, QImage, QPainterPath
import mnist_training
import neural_network
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os 
class Plot(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(5,4), dpi=200)
        super().__init__(fig)
        self.setParent(parent)

        

    def graph(self):
        

        root_dir = os.getcwd()
        image = mnist_training.open_image(root_dir + '\\data.png')
        x, y, index = mnist_training.prediction(image)
        
        plt.bar(x,y)
        plt.xlabel('Digits')
        plt.ylabel('probability')
        plt.title('Classified Digit: ' + str(index))

class Canvas(QLabel):    #Canvas Widget itself

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_StaticContents)
        h = 400
        w = 400
        self.myPenWidth = 45
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
        self.canvas.saveImage()
        self.show_plot = Plot(self)
        self.show_plot.graph()
        self.newwindow = Graph(self)
        self.newwindow.show()
    def RecognizeButton(self):
        Graph(self)
        




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

        viewTestingImages = QAction ('View Testing Images', self)
        viewTestingImages.setStatusTip('View the testing images')

        DrawingCanvas = QAction('Drawing Canvas',self)
        DrawingCanvas.setStatusTip('View the testing images')
        self.newwindow = CanvasWindow(self)
        DrawingCanvas.triggered.connect(self.CanvasClk)


        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(TrainModel)
        filemenu.addAction(exitAction)
        filemenu.addAction(DrawingCanvas)
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
        
        btn2 = QPushButton(self)
        btn2.setText('Train')

        btn3 = QPushButton('cancel', self)
        btn3.clicked.connect(dialog.close)
        
        grid.addWidget(btn1,3,1)
        grid.addWidget(btn2,3,2)
        grid.addWidget(btn3,3,3)
        dialog.show()
    def CanvasClk(self):
        self.newwindow.show()


    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
