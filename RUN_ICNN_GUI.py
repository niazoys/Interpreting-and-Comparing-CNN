from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap,QIntValidator
from PyQt5.QtWidgets import QGridLayout,QFileDialog, QScrollArea, QMainWindow,QPushButton,QGroupBox,QWidget,QVBoxLayout,QApplication,QSlider,QLabel,QButtonGroup,QComboBox,QPushButton,QLineEdit
from PIL import Image, ImageQt
import os 
import torch
import numpy as np
from torchvision import models
from utility import Utility
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr import visualization as vizu
import matplotlib.pyplot as plt


class NetStat(QWidget):
    def __init__(self):
        super(NetStat, self).__init__()

        self.setWindowTitle("Network Statistics")
        self.y=50
        self.x=300
        self.width = 700
        self.height = 500
        self.setGeometry(self.x,self.y,self.width,self.height)
        self.NetStat_window()

    def NetStat_window(self):
        self.NetStat_layout = QVBoxLayout()

        #Create Combo box(Network Selection)
        self.label1=QLabel("Select Netwrok")
        self.chooseNetwork = QComboBox()
        self.chooseNetwork.addItem('')
        self.chooseNetwork.addItem('alexnet')
        self.chooseNetwork.addItem('vgg11')
        self.chooseNetwork.addItem('resnet18')
        self.chooseNetwork.activated[str].connect(self.changeViznetowrk)
        
        

        #Create Combo box(Info to display)
        self.label2=QLabel("Select Statistics")
        self.chooseInfo = QComboBox()
        self.chooseInfo.addItem('')
        self.chooseInfo.addItem('Leyer Wise Interpretable Unit Distribution')
        self.chooseInfo.addItem('Class Wise network Relevance')
        self.chooseInfo.addItem('Ablation Statistics')
        self.chooseInfo.activated[str].connect(self.changeInfo)
        self.chooseInfo.setEnabled(False)
       
        #Placeholder for images 
        self.imageCanvas_stat=QLabel() 
        self.imageCanvas_stat.setAlignment(Qt.AlignCenter)
        pixmap=QPixmap('gui_resources/black.jpg')
        self.showImage(pixmap)
        
        #Navigation Keys
        self.fig_label=QLabel("Iou")
        self.fig_label.setAlignment(Qt.AlignCenter)
        self.navigation_layout = QGridLayout()
        self.next_button=QPushButton("Next")
        self.prev_button=QPushButton("Previous")
        self.navigation_layout.addWidget(self.prev_button,1,0)
        self.navigation_layout.addWidget(self.next_button,1,3)
        self.next_button.clicked.connect(self.navigate_next)
        self.prev_button.clicked.connect(self.navigate_prev)
       
        self.back_mainWindow=QPushButton("Back to Main Menu")
        self.back_mainWindow.clicked.connect(self.showMainWindow)


        self.NetStat_layout.addWidget(self.label1)
        self.NetStat_layout.addWidget(self.chooseNetwork)
        self.NetStat_layout.addWidget(self.label2)
        self.NetStat_layout.addWidget(self.chooseInfo)
        self.NetStat_layout.addWidget(self.imageCanvas_stat)
        #self.NetStat_layout.addWidget(self.fig_label)
        self.NetStat_layout.addLayout(self.navigation_layout)
        self.NetStat_layout.addWidget(self.back_mainWindow)      
        self.setLayout(self.NetStat_layout)
        self.show()
    
    def showImage(self,pixmap): 
        pixmap = pixmap.scaled(700, 700)
        self.imageCanvas_stat.setMaximumHeight(pixmap.height())
        self.imageCanvas_stat.setMaximumWidth(pixmap.width())
        self.imageCanvas_stat.setPixmap(pixmap)
        self.imageCanvas_stat.setAlignment(Qt.AlignCenter)

    def resolveDirectory(self,prefix):
        '''convert it into pillow image '''
        # img = Image.fromarray(image)
        # qt_img = ImageQt.ImageQt(img)
        #self.imageCanvas_stat.setPixmap(QtGui.QPixmap.fromImage(qt_img))
        self.path='gui_resources/'+prefix+'/'
        
        self.filename_list=[]
        for filename in os.listdir(self.path):
            self.filename_list.append(filename)
        
        self.current_count=0
        pixmap = QPixmap(os.path.join(self.path, self.filename_list[0]))
        self.showImage(pixmap)
        
    def navigate_next(self):    
       if self.current_count < len(self.filename_list)-1:
           self.current_count+=1
           pixmap = QPixmap(os.path.join(self.path, self.filename_list[self.current_count]))
           self.showImage(pixmap)

    def navigate_prev(self):
       if self.current_count>0:
           self.current_count-=1
           pixmap = QPixmap(os.path.join(self.path, self.filename_list[self.current_count]))
           self.showImage(pixmap)
        
    def showMainWindow(self):
        self.mainwindow = MainWindow()
        self.hide()
        self.mainwindow.show()

    def changeInfo(self,info):
        if info =='Leyer Wise Interpretable Unit Distribution':
            self.prefix=self.networkName+'_iou_dist'
            self.resolveDirectory(self.prefix)
        elif info == 'Class Wise network Relevance':
            self.prefix=self.networkName+'_class_rel_map'
            self.resolveDirectory(self.prefix)
        elif info=='Ablation Statistics':
            print("heloo")
            self.prefix='output_imgs/'+self.networkName
            self.resolveDirectory(self.prefix)

    def changeViznetowrk(self,netowrk):
        ''' Selects between the different Network'''
        if netowrk =='alexnet':
            self.chooseInfo.setEnabled(True)    
            self.networkName=netowrk
        elif netowrk=='vgg11':
            self.chooseInfo.setEnabled(True)
            self.networkName=netowrk
        elif netowrk =='resnet18':
            self.chooseInfo.setEnabled(True)
            self.networkName=netowrk
        else:
            print ("Wrong Netowrk Selected")

class NetAttribute(QWidget):
    def __init__ (self):
        super(NetAttribute, self).__init__()

        self.setWindowTitle("Network Attribution (Integrated Gradient)")
        self.y=40
        self.x=300
        self.width =1100
        self.height = 790
        self.setGeometry(self.x,self.y,self.width,self.height)
        self.NetAttribute_window()
        self.util=Utility()

    def NetAttribute_window(self):
        self.Netatt_layout = QVBoxLayout()

        #Create Combo box(Network Selection)
        self.label1=QLabel("Select Netwrok")
        self.chooseNetwork = QComboBox()
        self.chooseNetwork.addItem('')
        self.chooseNetwork.addItem('Alexnet')
        self.chooseNetwork.addItem('VGG')
        self.chooseNetwork.addItem('ResNet')
        self.chooseNetwork.activated[str].connect(self.changeNetowrk)

        #image place holder &  labels
        self.images_layout = QGridLayout()
        self.imageCanvas_original=QLabel() 
        self.imageCanvas_all=QLabel() 
        self.imageCanvas_abs=QLabel()
        self.imageCanvas_overlay=QLabel()

        self.images_layout.addWidget(self.imageCanvas_original,1,0)
        l1=QLabel('Original Image')
        l1.setAlignment(Qt.AlignCenter)
        self.images_layout.addWidget(l1,2,0)   
        self.images_layout.addWidget(self.imageCanvas_all,1,1)
        l2=QLabel('IG Map with Negative & Positive Gradient')
        l2.setAlignment(Qt.AlignCenter)
        self.images_layout.addWidget(l2,2,1)
        self.images_layout.addWidget(self.imageCanvas_abs,3,0)
        l3=QLabel('IG Map with Absolute Values of Gradient')
        l3.setAlignment(Qt.AlignCenter)
        self.images_layout.addWidget(l3,4,0)
        self.images_layout.addWidget(self.imageCanvas_overlay,3,1)
        l4=QLabel('IG Map Overlay on Original Image')
        l4.setAlignment(Qt.AlignCenter)
        self.images_layout.addWidget(l4,4,1)

        #put black temoprary template on image place holder
        template= QPixmap('gui_resources/black.jpg')
        self.plotQlabelImage(template,self.imageCanvas_original)
        self.plotQlabelImage(template,self.imageCanvas_abs)
        self.plotQlabelImage(template,self.imageCanvas_all)
        self.plotQlabelImage(template,self.imageCanvas_overlay)
        
        ## Browse Image File 
        self.get_button=QPushButton("Browse Image")
        self.get_button.clicked.connect(self.getImage)
        self.get_button.setEnabled(False)

        # Steps for IG 
        self.inputSteps = QLineEdit()
        self.inputSteps.setEnabled(False)
        self.inputSteps.textChanged.connect(self.setSteps)
        
        # Generate IG
        self.ig_button=QPushButton("Show IG")
        self.ig_button.clicked.connect(self.generateIG)
        self.ig_button.setEnabled(False)

        #go back to main window
        self.back_mainWindow=QPushButton("Back to Main Menu")
        self.back_mainWindow.clicked.connect(self.showMainWindow)
        
        ## Add to the Layout 
        
        self.Netatt_layout.addWidget(self.label1)
        self.Netatt_layout.addWidget(self.chooseNetwork)
        self.Netatt_layout.addWidget(QLabel("Number of Steps of IG"))
        self.Netatt_layout.addWidget(self.inputSteps)
        self.Netatt_layout.addWidget(self.get_button)
        self.Netatt_layout.addWidget(self.ig_button)
        self.Netatt_layout.addLayout(self.images_layout)
        self.Netatt_layout.addWidget(self.back_mainWindow)
        self.setLayout(self.Netatt_layout)
        self.show()


    def generateIG(self):
        self.getAttribution()
        self.showImage()

    def getAttribution(self):

        _, preds = torch.max(self.model(self.x),dim=1)
        
        #Initiate layer IG object
        self.attributor=IntegratedGradients(self.model)

        # get the IG map
        attribution=self.attributor.attribute(self.x,target=preds,n_steps=int(self.IG_steps))
        self.attribution=np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    def showImage(self):
        modes=["absolute_value","all"]
        for i in range(len(modes)):
            fig,_ = vizu.visualize_image_attr(self.attribution,sign=modes[i],
                                show_colorbar=True, title="Integrated Gradient",use_pyplot=False)
            fig.savefig('gui_resources/temp/'+str(i)+'.jpg')

        input_img=np.transpose(self.x.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        fig1,_ = vizu.visualize_image_attr(self.attribution,input_img,sign="absolute_value", method="blended_heat_map",use_pyplot=False,
                                show_colorbar=False, title="Integrated Gradient Overlayed on Input")
        fig1.savefig('gui_resources/temp/overlay.jpg')

        self.plotQlabelImage(QPixmap('gui_resources/temp/0.jpg'),self.imageCanvas_abs)
        self.plotQlabelImage(QPixmap('gui_resources/temp/1.jpg'),self.imageCanvas_all)
        self.plotQlabelImage(QPixmap('gui_resources/temp/overlay.jpg'),self.imageCanvas_overlay)

    def getImage(self):
      self.fname,_ = QFileDialog.getOpenFileName(self, 'Open file', 
         'c:\\',"Image files (*.jpg *.gif *.png *.jpeg *.JPEG)")
      self.x=self.util.load_single_image(self.fname,False)
      
      if self.x != None:
        pixmap = QPixmap(self.fname)
        self.plotQlabelImage(pixmap,self.imageCanvas_original)
        self.ig_button.setEnabled(True)
    
    def setSteps(self,value):
        self.IG_steps=value
        print(self.IG_steps)
        self.get_button.setEnabled(True)

    def changeNetowrk(self,netowrk):
        ''' Selects between the different Network'''
        if netowrk =='Alexnet':
            self.inputSteps.setEnabled(True)    
            self.model= models.alexnet(pretrained=True)
        elif netowrk=='VGG':
            self.inputSteps.setEnabled(True)
            self.model = models.vgg11(pretrained=True)

        elif netowrk =='ResNet':
            self.inputSteps.setEnabled(True)
            self.model = models.resnet18(pretrained=True)
        else:
            print ("Wrong Netowrk Selected")
    
    def plotQlabelImage(self,pixmap,imageCanvas):
        pixmap = pixmap.scaled(400, 360)
        imageCanvas.setMaximumHeight(pixmap.height())
        imageCanvas.setMaximumWidth(pixmap.width())
        imageCanvas.setPixmap(pixmap)
        imageCanvas.setAlignment(Qt.AlignCenter)

    def showMainWindow(self):
        self.mainwindow = MainWindow()
        self.hide()
        self.mainwindow.show()

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
        
        self.stat_button.clicked.connect(self.goto_netStat)
        self.attr_button.clicked.connect(self.goto_netAttribute)

        self.setLayout(self.start_layout)
        self.show()

    def goto_netStat(self):
        self.NetStat = NetStat()
        self.NetStat.show()
        self.close()
    
    def goto_netAttribute(self):
        self.NetAtt = NetAttribute()
        self.NetAtt.show()
        self.close()


if __name__ == '__main__':
    app = QApplication([])
    signup = MainWindow()
    app.exec_()

   