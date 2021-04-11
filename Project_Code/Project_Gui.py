import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QGridLayout, QPushButton, QLineEdit, QTextEdit
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
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

        

        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(TrainModel)
        filemenu.addAction(exitAction)
        viewmenu = menubar.addMenu('&View')
        viewmenu.addAction(viewTrainingImages)
        viewmenu.addAction(viewTestingImages)
       
       

        self.setWindowTitle('Handwritten Digit Recognizer')
        self.setGeometry(300, 300, 300, 200)
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
        
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
