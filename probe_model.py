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


from visualize_layers import  VisualizeLayers
from  utility import Utility
from dataloader import *
from compute_iou import compute_iou



class probe_model(nn.Module):
    def __init__(self,half_mode=False):
        super(probe_model,self).__init__()
        self.model = models.vgg16(pretrained=True)

        # Get the GPU if available
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        #self.device=torch.device("cpu")
        
        # attach the model to cuda
        self.model.to(self.device)

        # run in half precision modes for less memory consumption
        if half_mode:
            self.model.half()    

        
        # self.dataset_path='D:\\Net\\NetDissect\\dataset\\broden1_227'
        # self.dataset_path='E:\\TRDP_python\\Crack-The-CNN\\broden1_227'
        self.dataset_path='broden1_227'
        # self.dataset_path='D:\\Net\\NetDissect\\dataset\\broden1_227'
        #self.dataset_path='broden1_227'
        
        #create dataloader
        self.imLoader= imageLoader(self.dataset_path)
        
    def get_model(self):
        return self.model


    def model_eval(self,dataloader):
        total = 0
        correct = 0
        
        for x,y in dataloader:
            
            #transfer the data to GPU 
            x=x.to(self.device)
            y=y.to(self.device)
            
            out = self.model(x)
            
            max_val, preds = torch.max(out,dim=1)
            total += y.shape[0]                 
            correct += (preds == y).sum().item()
        
        accuracy = (100 * correct)/total
    
        return accuracy
    

    def probe(self,iteration,batch_size,vis,layer,part_ln,part):
        
        # Timer for execution time 
        start_time = time.time()

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
            interm_output_list=vis.get_interm_output_aslist()
            

            # Convert the list into tensor 
            interm_output_list=interm_output_list[0]


            #Figure out the start and end point of each part
            if idx_flag:
                self.step=int(list(interm_output_list.size())[1]/part_ln)
                bound=list(interm_output_list.size())[1]
                for n in range (0,bound+1,self.step):
                    self.idx_list.append(n)
                idx_flag=False

            
            interm_output_list=interm_output_list[:, self.idx_list[part-1]: self.idx_list[part],:,:]
            
            #interm_output_list=interm_output_list[:,1,:,:]
            
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
        
        #print("Process finished --- %s seconds ---" % (time.time() - start_time))

#%%
if __name__=="__main__":
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()])

    # trainset=torchvision.datasets.CIFAR100(root='./dataset',train=True,
    #                               download=True,transform=transform)

    # trainloader=torch.utils.data.CIFAR100(trainset,batch_size=100,shuffle=True)

    testset=torchvision.datasets.CIFAR100(root='./dataset',train=False,
                                 download=True,transform=transform)

    testloader=torch.utils.data.DataLoader(testset,batch_size=2,shuffle=False) 

    modella = probe_model()
    acc = modella.model_eval(testloader) 
    print(acc)  
    