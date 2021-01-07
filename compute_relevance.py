
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr import visualization as vizu
import matplotlib.pyplot as plt

from Layer_hooker import  Hooker
from  utility import Utility
from probe_model import probe_model
import numpy as np 
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision import models
from dataloader import classLoader

from PIL import Image

class ComputeRelevance():
    def __init__(self):
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model.to(self.device) 
        self.hooker=Hooker(self.model)

    def revelence_score_pipeline (self,x,mask,model,layer,attribute_to_layer_input):
        '''
        Args:
        Input  : X     : input data in batch fashion. dimension (b*c*h*w)
                Mask  : Coresponding mask for each image samples. (b*h*w)
                Model : The Model. 
                Layer : The layer for which the relevance scores need to calculated. 
        OUTPUT :
                Raw Relevance attribution for entire layer. size 4D array (batch *Units * LayerInputSize)
                Maked (with ROI annotation) Relevance attribution for entire layer. size 4D array (batch *Units * LayerInputSize)
                Average Relevance score for each Neurons. size equivalent to number of units    
        '''
    
        #Get the prediction 
        max_val, preds = torch.max(model(x),dim=1)

    
        #Initiate layer IG object
        layer_ig=LayerIntegratedGradients(model,layer)

        # get the activation for selected layer 
        attribution=layer_ig.attribute(x,target=preds,attribute_to_layer_input=attribute_to_layer_input)
        attribution_=attribution.cpu().detach().squeeze(0).numpy()
        
        #Resize the attribution 
        attribution=Utility.resize_IG_batch(attribution_)
        
        #Apply the mask on the IG map 
        masked_attribution= np.copy(attribution)
        for img in range(attribution.shape[0]):
            for unit in range(attribution.shape[1]):
                masked_attribution[img,unit,:,:]=np.multiply(masked_attribution[img,unit,:,:],mask[img,:,:])

        #calculate the relevance score per unit 
        relevance_score =np.sum(np.sum(abs(masked_attribution),axis=3),axis=2)
                
        return attribution,masked_attribution,relevance_score

    def main(self):
              
        # Get the data and annotation mask 
        dataset_path='broden1_227'
        clLoader=classLoader(dataset_path)
     
        #Get the layers 
        layer_names=self.hooker.get_saved_layer_names()
        

        class_list =[105,88,70,50,519,135,123,121]

        for idx in range(1,len(layer_names)):
            layer=self.hooker.conv_layers[layer_names[idx]]
            print(layer_names[idx])
            for selected_class in class_list:
                list_batch_relevance_score=[]
                batch_size=2
                clLoader.data_counter=0
                sample_count   = clLoader.get_length(selected_class)
                iterations     =int( np.floor(sample_count/batch_size) )
                print("total batch",iterations)

                for i in range(iterations):     
                    #load data & Mask in a batch  
                    x,mask=clLoader.load_batch(selected_class,batch_size)
                    x=x.to(self.device)

                    if idx==0:
                        #Get relevance score 
                        _,_,batch_relevance_score=self.revelence_score_pipeline(x,mask,self.model,layer,True)
                    else:
                        #Get relevance score 
                        _,_,batch_relevance_score=self.revelence_score_pipeline(x,mask,self.model,layer,True)

                    list_batch_relevance_score.append (batch_relevance_score)
                    print("Processed Class: "+str(selected_class)+ " Batch:",i)

                #stack the samples gathered 
                relevance_score=np.vstack(list_batch_relevance_score)
                #Compute 
                avg_relevance_score=np.average(relevance_score,axis=0)
                
                # save IG score    
                np.save('IG/resnet18/IG_'+str(layer_names[idx])+'_class_0'+str(selected_class)+'.npy',avg_relevance_score)


if  __name__ == "__main__":
    relevance=ComputeRelevance()
    relevance.main()