import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from skimage.transform import resize
import torchvision.datasets as datasets
from torchvision import models
import matplotlib.pyplot as plt
import time
import numpy as np
import gc

from Layer_hooker import  Hooker
from  utility import Utility
from dataloader import imageLoader,conceptLoader

class probe_model(nn.Module):
    def __init__(self,half_mode=False):
        super(probe_model,self).__init__()
        self.model = models.vgg11(pretrained=True)
        util=Utility()

        # Get the GPU if available
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
       
        # attach the model to cuda
        self.model.to(self.device)

        # run in half precision modes for less memory consumption
        if half_mode:
            self.model.half()    

        
        self.dataset_path='broden1_227'
       
        
        #create dataloader
        self.imLoader= imageLoader(self.dataset_path)
        
    def get_model(self):
        return self.model

    def probe(self,iteration,batch_size,hooker,layer,part_ln,part):
        featuremap=[]
        idx_flag=True
        
        self.idx_list=[]

        for i in range (iteration):
          
            #load data
            data_multi=self.imLoader.load_batch(batch_size).detach()
            
            # transfer the tensor to GPU       
            data_multi=data_multi.to(self.device)
            
          
            #forwards pass
            self.model(data_multi)
            #delete the data from GPU mem 
            del data_multi

            
            # get the activation 
            interm_output_list=hooker.get_interm_output_aslist()
            

            # Convert the list into tensor 
            interm_output_list=interm_output_list[0]


            #Figure out the start and end point of each part
            if idx_flag:
                self.step=int(list(interm_output_list.size())[1]/part_ln)
                bound=list(interm_output_list.size())[1]
                for n in range (0,bound+1,self.step):
                    self.idx_list.append(n)
                idx_flag=False
           
            # Keep the part of activations
            interm_output_list=interm_output_list[:, self.idx_list[part-1]: self.idx_list[part],:,:]
            
            
            # detach the featuremap tensor from GPU and convert it to numpy array 
            featuremap.append(interm_output_list.cpu().detach().numpy())
            
            # delete the featuremap from gpu 
            del interm_output_list

            # Free up the cuda cache & collect garbages 
            torch.cuda.empty_cache() 
            gc.collect()
            print("Batch Processed: "+str(i))
        
        print("Stacking The Featuremap")
        #convert the list of numpy array to numpy array
        featuremap=np.vstack(featuremap)
    
        return featuremap
        