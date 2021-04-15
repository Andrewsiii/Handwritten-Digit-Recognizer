import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QGridLayout, QPushButton, QLineEdit, QTextEdit, QLabel,QVBoxLayout,QWidget,QHBoxLayout, QMenu
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPen, QPixmap, QPainter

class Canvas(QLabel):    #Canvas Widget itself

    def __init__(self):
        super().__init__()
        self.canvas = QPixmap(600, 400)
        self.canvas.fill(Qt.black)
        self.setPixmap(self.canvas)
    def mouseMoveEvent(self, e):
        self.painter = QPainter(self.pixmap())
        self.painter.setPen(QPen(Qt.white, 50))  #Change Pen thickness HERE increase number for thicker pen
        self.painter.drawPoint(e.x(), e.y())  #think u do canvas.save to save the widget as image. Will test later. it works like canvas.save("example.png")
        self.canvas.save('image.png')
        self.painter.end()
        self.update()
    def Clear(self):    #Clears the canvas
        self.painter = QPainter(self.pixmap())
        self.painter.eraseRect(0,0,600,400)
        self.painter.end()
        self.canvas.fill(Qt.black)
        self.setPixmap(self.canvas)
        self.update()
    
class CanvasWindow(QMainWindow):   #The Canvas Window

    def __init__(self,parent = None):
        super(CanvasWindow,self).__init__(parent,Qt.Window)

        self.canvas = Canvas()
        canvas = QWidget()
        grid =QGridLayout()
        canvas.setLayout(grid)
        grid.addWidget(self.canvas,0,1)
        Recognize = QPushButton('&Recognize', self)
        Clear = QPushButton('&Clear', self)
        Clear.clicked.connect(self.canvas.Clear)
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
