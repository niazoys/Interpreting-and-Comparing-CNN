
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
    
    # plt.imshow(mask[1,:,:])
    # plt.show()
    # plt.imshow(np.transpose(x[1,:,:,:].cpu().numpy(),(1,2,0)))
    # plt.show()
    # agg=np.sum(attribution[1,:,:,:],axis=0)

    # plt.imshow(agg,cmap="inferno")
    # plt.show()
    att=Utility.resize_IG_batch(attribution_,[227,227])
    mat=np.load('top_bottom_index/resnet18/bottom_class_0105.npy',allow_pickle=True)
    mat=mat[2][0]
    t_att=att[1,mat,:,:]
    t_att=np.sum(t_att,axis=0)

    # th=np.quantile(t_att,0.99999)
    # t_att=t_att>th
    
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=   (8, 8))
    axs[0, 0].set_title('Attribution mask')
    axs[0, 0].imshow(att[1,5,:,:], cmap=plt.cm.inferno)
    axs[0, 0].axis('off')
    axs[0, 1].set_title('Overlay IG on Input image ')
    
    #t_att=t_att-t_att.min()/t_att.max()
    axs[0, 1].imshow(t_att)
    tmp=np.transpose(x[1,:,:,:].cpu().numpy(),(1,2,0))
    #tmp=Utility.normalize_image(tmp)
    axs[0, 1].imshow(tmp, alpha=0.4)
    axs[0, 1].axis('off')
    plt.tight_layout()
    plt.show()
    
    # img = Image.fromarray(np.uint8(att[1,1,:,:] * 255) , 'L')
    # img.show()

    # tmp=np.transpose(x[1,:,:,:].cpu().numpy(),(1,2,0))
    # _ = vizu.visualize_image_attr((att[1,1,:,:]),tmp[:,:,1],sign="absolute_value", method="blended_heat_map",use_pyplot=False,
    #                             show_colorbar=False, title="Integrated Gradient Overlayed on Input")
    
    
    return attribution,masked_attribution,relevance_score



if  __name__ == "__main__":
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #device=torch.device("cpu")   
    # Get the data and annotation mask 
    #dataset_path='broden1_227'
    dataset_path='D:\\Net\\NetDissect\\dataset\\broden1_227'
    clLoader=classLoader(dataset_path)
   
    #Get the model
    '''
    Remember to switch  the full(float32) computation mode by setting the 
    2nd argument to False for non Residual networks.(e.g. resnet18 ,VGG) 
    '''
    model = models.resnet18(pretrained=True)
    model.to(device)
    #Get the layers 
    '''
    Remember to change the 2nd argument to False for non Residual networks.(e.g. resnet18 ,VGG) 
    '''
    vis=Hooker(model)
    layer_names=vis.get_saved_layer_names()
    

    class_list =[105]

    for idx in range(3,len(layer_names)):
        layer=vis.conv_layers[layer_names[idx]]
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
                x=x.to(device)

                if idx==0:
                    #Get relevance score 
                    _,_,batch_relevance_score=revelence_score_pipeline(x,mask,model,layer,True)
                else:
                    #Get relevance score 
                    _,_,batch_relevance_score=revelence_score_pipeline(x,mask,model,layer,True)

                list_batch_relevance_score.append (batch_relevance_score)
                print("Processed Class: "+str(selected_class)+ " Batch:",i)

            #stack the samples gathered 
            relevance_score=np.vstack(list_batch_relevance_score)
            #Compute 
            avg_relevance_score=np.average(relevance_score,axis=0)
            
            # save IG score    
            np.save('IG/resnet18/IG_'+str(layer_names[idx])+'_class_0'+str(selected_class)+'.npy',avg_relevance_score)

          

            
        