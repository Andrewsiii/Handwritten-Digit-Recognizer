import sys
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QGridLayout, QPushButton, QLineEdit, QTextEdit, QLabel,QVBoxLayout
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPen, QPixmap, QPainter



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

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
        self.show()


    def mouseMoveEvent(self, e):
        painter = QPainter(self.label.pixmap())
        painter.setPen(QPen(Qt.white,  20))
        painter.drawPoint(e.x(), e.y())
        painter.end()
        self.update()
        

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()