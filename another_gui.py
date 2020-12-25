from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap,QIntValidator
from PyQt5.QtWidgets import QGridLayout,QScrollArea, QMainWindow,QPushButton,QGroupBox,QWidget,QVBoxLayout,QApplication,QSlider,QLabel,QButtonGroup,QComboBox,QPushButton,QLineEdit



class NetStat(QWidget):
    def __init__(self):
        super(NetStat, self).__init__()

        self.setWindowTitle("Network Statistics")
        self.y=200
        self.x=500
        self.width = 500
        self.height = 500
        self.setGeometry(self.x,self.y,self.width,self.height)
        self.NetStat_window()

    def NetStat_window(self):
        self.NetStat_layout = QVBoxLayout()

        #Create Combo box(Network Selection)
        self.label1=QLabel("Select Netwrok")
        self.chooseNetwork = QComboBox()
        self.chooseNetwork.addItem('')
        self.chooseNetwork.addItem('Alexnet')
        self.chooseNetwork.addItem('VGG')
        self.chooseNetwork.addItem('ResNet')
        self.chooseNetwork.activated[str].connect(self.changeViznetowrk)
        
        #Create Combo box(Info to display)
        self.label2=QLabel("Select Statistics")
        self.chooseInfo = QComboBox()
        self.chooseInfo.addItem('')
        self.chooseInfo.addItem('Leyer Wise Interpretable Unit Distribution')
        self.chooseInfo.addItem('Class Wise network Relevance')
        self.chooseInfo.activated[str].connect(self.changeInfo)
        self.chooseInfo.setEnabled(False)
       
        #Placeholder for images 
        self.imageCanvas_stat=QLabel() 
        self.imageCanvas_stat.setAlignment(Qt.AlignCenter)
        self.showImage('test.jpg')
       
        #Navigation Keys
        self.fig_label=QLabel("Iou")
        self.fig_label.setAlignment(Qt.AlignCenter)
        self.navigation_layout = QGridLayout()
        self.next_button=QPushButton("Next")
        self.prev_button=QPushButton("Previous")
        self.navigation_layout.addWidget(self.prev_button,1,0)
        #self.navigation_layout.addWidget(self.fig_label,1,2)
        self.navigation_layout.addWidget(self.next_button,1,3)

        self.NetStat_layout.addWidget(self.label1)
        self.NetStat_layout.addWidget(self.chooseNetwork)
        self.NetStat_layout.addWidget(self.label2)
        self.NetStat_layout.addWidget(self.chooseInfo)
        self.NetStat_layout.addWidget(self.imageCanvas_stat)
        self.NetStat_layout.addWidget(self.fig_label)
        self.NetStat_layout.addLayout(self.navigation_layout)
      
        self.setLayout(self.NetStat_layout)
        self.show()
    
    def showImage(self,path):
        '''convert it into pillow image '''
        # img = Image.fromarray(image)
        # qt_img = ImageQt.ImageQt(img)
        #self.imageCanvas_stat.setPixmap(QtGui.QPixmap.fromImage(qt_img))
        
        pixmap = QPixmap(path)
       
        pixmap = pixmap.scaled(450, 450)
        self.imageCanvas_stat.setMaximumHeight(pixmap.height())
        self.imageCanvas_stat.setMaximumWidth(pixmap.width())
        self.imageCanvas_stat.setPixmap(pixmap)
        self.imageCanvas_stat.setAlignment(Qt.AlignCenter)

      

    def signup_show(self):
        self.mainwindow = MainWindow()
        self.hide()
        self.mainwindow.show()


    def check_signup(self):
        SignUp.check_signup()
        self.show()


    def changeInfo(self,info):
        if info =='Leyer Wise Interpretable Unit Distribution':
            print("ok1")
        elif info == 'Class Wise network Relevance':
            pass


    def changeViznetowrk(self,netowrk):
        ''' Selects between the different Network'''
        if netowrk =='Alexnet':
            self.chooseInfo.setEnabled(True)    
            print("alex")
        elif netowrk=='VGG':
            self.chooseInfo.setEnabled(True)
            pass
        elif netowrk =='ResNet':
            self.chooseInfo.setEnabled(True)
            pass
        else:
            print ("Wrong Netowrk Selected")



class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Main Window")
        self.y=200
        self.x=500
        self.width = 500
        self.height = 500
        self.setGeometry(self.x,self.y,self.width,self.height)
        self.start_window()

    def start_window(self):

        self.start_layout = QGridLayout()
        self.stat_button = QPushButton("Network Statistics")
        self.attr_button = QPushButton("Network Attribution")
        self.start_layout.addWidget(self.stat_button, 2, 0, 1, 0)
        self.start_layout.addWidget(self.attr_button, 3, 0, 1, 0)
        
        self.stat_button.clicked.connect(self.check_signup)
        self.attr_button.clicked.connect(self.check_signup)

        self.setLayout(self.start_layout)
        self.show()

    def check_signup(self):
        self.NetStat = NetStat()
        self.NetStat.show()
        self.close()

def main():
    app = QApplication([])
    signup = MainWindow()
    app.exec_()

if __name__ == '__main__':
    main()