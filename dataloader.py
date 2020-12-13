from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms.functional as TF

class imageLoader(Dataset):
    """Broaden Dataset Image Loader"""
    def __init__(self, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_counter = 0   #Keeps track of the data drawn from the dataset
        csv_file = os.path.join(root_dir,"index.csv") 
        root_dir = os.path.join(root_dir,"images") 
        self.fileNames = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.fileNames)
    

    def load_batch(self,batch_size):
        data=torch.zeros([batch_size, 3, 227, 227])
        for i in range (batch_size):
            data_temp=self.__getitem__(i+self.data_counter)
            data[i]=data_temp.unsqueeze_(0)
        self.data_counter +=batch_size 
        return data.half()


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.fileNames.iloc[idx, 0])
        
        image = Image.open(img_name)
        sample = TF.to_tensor(image)

        return sample



concepts = ["color","object","part","material","scene","texture"]

class conceptLoader(Dataset):
    """Broaden Dataset Image Loader"""
    
    def __init__(self, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            conceptType (string): type of the concept to load.
        """
        self.c_flag = False  # Concept Flag
        csv_file = os.path.join(root_dir,"index.csv") 
        root_dir = os.path.join(root_dir,"images") 
        self.fileNames = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.fileNames)
    

    def load_concept(self,idx,conceptType):
        c_index = str(self.fileNames.iloc[idx, concepts.index(conceptType)+6])
        if c_index=='nan':
            self.c_flag = False
            sample =0
        else:
            self.c_flag = True
            if conceptType =="scene" or conceptType =="texture":
                sample  = c_index.split(";") 
            elif conceptType =="part":
                items  = c_index.split(";") 
                sample = []
                for i in items:
                    img_name = os.path.join(self.root_dir,i)
                    image = Image.open(img_name)
                    px = np.array(image)
                    item = self.decodeClassMask(px)
                    sample.append(item)

            else:
                img_name = os.path.join(self.root_dir,c_index)
                image = Image.open(img_name)
                px = np.array(image)
                sample = self.decodeClassMask(px)
                # sample = Image.fromarray(sample)
                # sample = TF.to_tensor(sample)     

        # sample.unsqueeze_(0)
        return self.c_flag, sample

    def decodeClassMask(self,im):
        ''' Decodes pixel-level object/part class and instance data from
        the given image, previously encoded into RGB channels.'''
    # Classes are a combination of RG channels
        return (im[:,:,0] + (im[:,:,1])*256)

class classLoader(Dataset):
    """Broaden Dataset class wise Loader"""
    
    def __init__(self, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            conceptType (string): type of the concept to load.
        """
        self.data_counter = 0 #data counter
        self.c_flag = False  # Concept Flag
        self.csv_file_path = './New_Dataset'
        root_dir = os.path.join(root_dir,"images") 
        self.root_dir = root_dir
    
    def load_batch(self,idx,batch_size):
        image_data=torch.zeros([batch_size, 3, 227, 227])
        mask_data=torch.zeros([batch_size, 113, 113])
        
        for i in range (batch_size):
            # #array
            _flag,image_temp,mask_temp=self.load_class(idx,i+self.data_counter)
            image_data[i]=image_temp.unsqueeze_(0)
            mask_data[i]=mask_temp.unsqueeze_(0)
        self.data_counter +=batch_size 
        return image_data,mask_data

    def get_length(self,class_number):
        csv_file = os.path.join(self.csv_file_path,str(class_number)+'.csv')
        fileNames = pd.read_csv(csv_file)
        return len(fileNames)

    def load_class(self,idx,k):
        csv_file = os.path.join(self.csv_file_path,str(idx)+'.csv')
        self.fileNames = pd.read_csv(csv_file)
        c_index = str(self.fileNames.iloc[k, 1])
        if c_index=='nan':
            self.c_flag = False
            sample = torch.zeros([ 3, 227, 227])
            mask   = torch.zeros([ 113, 113])
        else:
            #To generate the mask
            self.c_flag = True
            img_name = os.path.join(self.root_dir,c_index)
            image = Image.open(img_name)
            px = np.array(image)
            mask = self.decodeClassMask(px)
            mask[mask!=idx]=0
            mask[mask==idx]=1
            mask = Image.fromarray(mask)
            mask = TF.to_tensor(mask)   

            #To load the Image
            c_index = str(self.fileNames.iloc[k, 0])
            img_name = os.path.join(self.root_dir,c_index)
            image = Image.open(img_name)
            px = np.array(image)
            sample = Image.fromarray(px)
            sample = TF.to_tensor(sample) 
        # sample.unsqueeze_(0)
        return self.c_flag, sample, mask

    def decodeClassMask(self,im):
        ''' Decodes pixel-level object/part class and instance data from
        the given image, previously encoded into RGB channels.'''
    # Classes are a combination of RG channels
        return (im[:,:,0] + (im[:,:,1])*256)

  
if __name__ == "__main__":

    '''  Sample Useages '''
    dataset_path='D:\\Net\\NetDissect\\dataset\\broden1_227'
    # data_obj = imageLoader(dataset_path)
    # data_obj = conceptLoader(dataset_path)
    data_obj = classLoader(dataset_path)
    # flag,sample,mask = data_obj.load_class(1,1)
    sample,mask = data_obj.load_batch(93,20)
    # print(aa.shape)
    # print(np.unique(aa.numpy()))
    # print(flag)
    print(sample.shape)
    print(mask.shape)
    