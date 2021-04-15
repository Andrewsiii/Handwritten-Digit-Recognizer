import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QGridLayout, QPushButton, QLineEdit, QTextEdit, QLabel,QVBoxLayout
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPen, QPixmap, QPainter

class CanvasWindow(QMainWindow):
        def __init__(self,parent = None):
            super(CanvasWindow,self).__init__(parent,Qt.Window)
            self.label = QLabel()
            canvas = QPixmap(500,500)
            canvas.fill(Qt.black)
            self.label.setPixmap(canvas)

            self.setCentralWidget(self.label)
            vbox = QVBoxLayout()
            vbox.addWidget(self.label)
            self.setLayout(vbox)
            self.setWindowTitle('Canvas')
            self.setGeometry(300, 200, 800, 600)
           


        def mouseMoveEvent(self, e):
            painter = QPainter(self.label.pixmap())
            painter.setPen(QPen(Qt.white,  20))
            painter.drawPoint(e.x(), e.y())
            painter.end()
            self.update()

class MyApp(QMainWindow):

    def __init__(self,parent=None):
        super(MyApp,self).__init__(parent)
        self.initUI()

    def initUI(self):
        exitAction = QAction('Quit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        TrainModel = QAction ('Train Model', self)
        TrainModel.setShortcut('Ctrl+T')
        TrainModel.setStatusTip('Train the model')
        TrainModel.triggered.connect(self.OpenWindow)
        

        viewTrainingImages = QAction ('View Training Images', self)
        viewTrainingImages.setStatusTip('View the training images')

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

    def OpenWindow(self):
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
