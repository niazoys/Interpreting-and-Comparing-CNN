
from captum.attr._core.neuron.neuron_integrated_gradients import NeuronIntegratedGradients
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr import visualization as vizu
import matplotlib.pyplot as plt

from visualize_layers import  VisualizeLayers
from  utility import Utility
from probe_model import probe_model
import numpy as np 
import torch
import torchvision.transforms as transforms
import torchvision
from dataloader import classLoader

def revelence_score_pipeline (x,mask,model,layer):
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
    out=model(x)
    max_val, preds = torch.max(out,dim=1)

    #Initiate layer IG object
    layer_ig=LayerIntegratedGradients(model,layer)

    # get the activation for selected layer 
    attribution=layer_ig.attribute(x,target=preds,attribute_to_layer_input=True)
    attribution=attribution.detach().squeeze(0).numpy()
    
    #Resize the attribution 
    attribution=Utility.resize_IG_batch(attribution)
    
    #Apply the mask on the IG map 
    masked_attribution= np.copy(attribution)
    for img in range(attribution.shape[0]):
        for unit in range(attribution.shape[1]):
            masked_attribution[img,unit,:,:]=np.multiply(masked_attribution[img,unit,:,:],mask[img,:,:])

    #calculate the relevance score per unit 
    relevance_score =np.sum(np.sum(abs(masked_attribution),axis=3),axis=2)
    avg_relevance_score=np.average(relevance_score,axis=0)

    return attribution,masked_attribution,relevance_score

if  __name__ == "__main__":
    
    batch_size=10

    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()])

    testset=torchvision.datasets.CIFAR100(root='./dataset',train=False,
                                 download=True,transform=transform)

    testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False) 


    #Get the model
    prober=probe_model()
    model=prober.get_model()
    
    #Get the layers 
    vis=VisualizeLayers(model)
    names=vis.get_saved_layer_names()
    layer=vis.conv_layers[names[10]]

    #Load a single image
    x=Utility.load_single_image('C:\\Users\\Niaz\OneDrive\\StudyMaterials\\UBx\\TRDP2\\ICCNN\\Crack-The-CNN\\cat_01.jpg',load_mask=False)
    original_image=np.transpose(x.cpu().numpy().squeeze(),(1,2,0))
    

    # a demo mask for testing 
    mask =np.zeros((batch_size,113,113))
    mask[:,25:95,25:95]=1
    
    for x,y in testloader:
        #Get relevance score 
        _,masked_attribution,avg_relevance_score=revelence_score_pipeline(x,mask,model,layer)

        #visualize the masked maps 
        # _ = vizu.visualize_image_attr(np.expand_dims(masked_attribution[:,10,:,:],axis=2),sign="absolute_value",
        #                 show_colorbar=True, title="IG")
        
        #show many images 
        im_sample=1
        Utility.show_many_images(masked_attribution[im_sample,:,:,:],36)
        
        plt.hist(avg_relevance_score[im_sample,:], bins=8, histtype='barstacked')
        plt.show()

        
        