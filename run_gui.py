from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap,QIntValidator
from PyQt5.QtWidgets import QGridLayout, QMainWindow,QPushButton,QGroupBox,QWidget,QVBoxLayout,QApplication,QSlider,QLabel,QButtonGroup,QComboBox,QPushButton,QLineEdit



class MainWindow(QWidget):
    def __init__(self):
        super(Login, self).__init__()

        self.setWindowTitle("Login")
        self.y=200
        self.x=500
        self.width = 500
        self.height = 500
        self.setGeometry(self.x,self.y,self.width,self.height)
        self.login_window()

    def start_window(self):
        self.login_layout = QGridLayout()

        self.login_button = QPushButton("Login")
        self.signup_button = QPushButton("Sign Up")

        self.login_layout.addWidget(self.login_button, 2, 0)
        self.login_layout.addWidget(self.signup_button, 3, 0)

        self.signup_button.clicked.connect(self.signup_show)

        self.setLayout(self.login_layout)
        self.show()

    def signup_show(self):
        self.signupshow = SignUp()
        self.hide()
        self.signupshow.show()

    def check_signup(self):
        SignUp.check_signup()
        self.show()


class SignUp(QWidget):
    def __init__(self):
        super(SignUp, self).__init__()

        self.setWindowTitle("Sign Up")
        self.y=200
        self.x=500
        self.width = 500
        self.height = 500
        self.setGeometry(self.x,self.y,self.width,self.height)
        self.signup_window()

    def signup_window(self):

        self.signup_layout = QGridLayout()
        self.signup_button = QPushButton("Sign Up")
        self.signup_layout.addWidget(self.signup_button, 2, 0, 1, 0)
        self.signup_button.clicked.connect(self.check_signup)
        self.setLayout(self.signup_layout)
        self.show()

    def check_signup(self):
        self.login = Login()
        self.login.show()
        self.close()

def main():
    app = QApplication([])
    signup = SignUp()
    app.exec_()

if __name__ == '__main__':
    main()