
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr import visualization as vizu
import matplotlib.pyplot as plt

from visualize_layers import  VisualizeLayers
from  utility import Utility
from probe_model import probe_model
import numpy as np 
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision import models
from dataloader import classLoader




 


def revelence_score_pipeline (x,mask,model,layer,attribute_to_layer_input):
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
    attribution=attribution.cpu().detach().squeeze(0).numpy()
    
    #Resize the attribution 
    attribution=Utility.resize_IG_batch(attribution)
    
    #Apply the mask on the IG map 
    masked_attribution= np.copy(attribution)
    for img in range(attribution.shape[0]):
        for unit in range(attribution.shape[1]):
            masked_attribution[img,unit,:,:]=np.multiply(masked_attribution[img,unit,:,:],mask[img,:,:])

    #calculate the relevance score per unit 
    relevance_score =np.sum(np.sum(abs(masked_attribution),axis=3),axis=2)
    
    # plt.imshow(mask[1,:,:])
    # plt.show()
    # plt.imshow(np.transpose(x[1,:,:,:].cpu().numpy(),(1,2,0)))
    # plt.show()
    # plt.imshow(attribution[1,50,:,:],cmap='gray')
    # plt.show()
    # plt.imshow(masked_attribution[1,50,:,:],cmap='gray')
    # plt.show()
    
    
    return attribution,masked_attribution,relevance_score



if  __name__ == "__main__":
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #device=torch.device("cpu")   
    # Get the data and annotation mask 
    dataset_path='broden1_227'
    #dataset_path='D:\\Net\\NetDissect\\dataset\\broden1_227'
    clLoader=classLoader(dataset_path)
   
    #Get the model
    '''
    Remember to switch  the full(float32) computation mode by setting the 
    2nd argument to False for non Residual networks.(e.g. ALexnet ,VGG) 
    '''
    model = models.vgg11(pretrained=True)
    model.to(device)
    #Get the layers 
    '''
    Remember to change the 2nd argument to False for non Residual networks.(e.g. ALexnet ,VGG) 
    '''
    vis=VisualizeLayers(model)
    layer_names=vis.get_saved_layer_names()
    

    # Dog=93 ,cat=105.mosque=1062,hen=830

    class_list =[123,50,191,121,135]

    for idx in range(1,len(layer_names)):
        layer=vis.conv_layers[layer_names[idx]]
        for selected_class in class_list:
            list_batch_relevance_score=[]
            batch_size=10
            clLoader.data_counter=0
            sample_count   = clLoader.get_length(selected_class)
            iterations     =int( np.floor(sample_count/batch_size) )
            print("total batch",iterations)

            for i in range(iterations):     
                #load data & Mask in a batch  
                x,mask=clLoader.load_batch(selected_class,batch_size)
                x=x.to(device)

                if idx==0:
                    #Get relevance score 
                    _,_,batch_relevance_score=revelence_score_pipeline(x,mask,model,layer,True)
                else:
                    #Get relevance score 
                    _,_,batch_relevance_score=revelence_score_pipeline(x,mask,model,layer,True)

                #visualize the masked maps 
                # _ = vizu.visualize_image_attr(np.expand_dims(masked_attribution[:,10,:,:],axis=2),sign="absolute_value",
                #                 show_colorbar=True, title="IG")
                
                # #show many images 
                # im_sample=1
                # Utility.show_many_images((masked_attribution[im_sample,:,:,:]),36,False)
                
                # for i in range(5):
                #     plt.imshow ((masked_attribution[im_sample,i,:,:]))
                #     plt.colorbar()
                #     plt.show()

                list_batch_relevance_score.append (batch_relevance_score)
                print("Processed Class: "+str(selected_class)+ " Batch:",i)

            
            relevance_score=np.vstack(list_batch_relevance_score)

            avg_relevance_score=np.average(relevance_score,axis=0)
            
            # save IG score    
            np.save('IG/vgg/IG_'+str(layer_names[idx])+'_class_0'+str(selected_class)+'.npy',avg_relevance_score)

            # plt.hist(avg_relevance_score, bins=8, histtype='barstacked')
            # plt.title("IG Score distribution for "+str(layer_names[idx]))
            # plt.show()

            
        