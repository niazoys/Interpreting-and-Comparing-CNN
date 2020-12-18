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



    
def model_eval(model,dataloader,label):
    total = 0
    correct = 0

    for x,y in dataloader:

        #transfer the data to GPU 
        x=x.to("cuda")
        #y=y.to("cuda")
        model.to("cuda")
        out = model(x)

        max_val, preds = torch.max(out,dim=1)
        total += x.shape[0]                 
        correct += (preds == label).sum().item()

        accuracy = (100 * correct)/total

    return accuracy


def get_class_dataLoader(path,batch_size):
    #path = '/path/to/imagenet-folder/train'
    transform = transforms.Compose(
        [transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])]
    )
    imagenet_data = torchvision.datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader


def ablate(model,selected_class,Top,percentile):
    vis=VisualizeLayers(model,False)
    layer_names=vis.get_saved_layer_names()
    
    for idx in range(len(layer_names)):
        idx=2
        
        # Get the layers
        layer=vis.conv_layers[layer_names[idx+1]]
        print(layer_names[idx+1])
       
        # Load IG matrix
        mat=np.load("IG/alexnet/IG_"+layer_names[idx+2]+"_class_0"+str(selected_class)+".npy")

        # get the neuron index to be turned off 
        if Top:
            threshold=np.quantile(mat,1-percentile/100)
            print(1-percentile/100)
            itemindex = np.where(mat>threshold)
            print(mat[itemindex])
        if Top==False:
            threshold=np.quantile(mat,percentile/100)
            print(percentile/100)
            itemindex = np.where(mat<threshold)
            print(mat[itemindex])

        # Turn off the neurons 
        for num_unit in itemindex:
            layer.weight.data[num_unit,:,:,:]=0
            layer.bias.data[num_unit]=0

if __name__ == "__main__":
    model = models.alexnet(pretrained=True)
    data_loader=get_class_dataLoader('n03891251/',20)
    model.eval()
    acc_before=model_eval(model,data_loader,703)
    ablate(model,121,False,2)
    model.eval()
    acc_after=model_eval(model,data_loader,703)
    
    print(acc_before,acc_after)
    
    