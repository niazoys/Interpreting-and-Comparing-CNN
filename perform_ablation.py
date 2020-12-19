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

QT_DEBUG_PLUGINS=1 

    
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

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def ablate(model,selected_class,Top,percentile):
    vis=VisualizeLayers(model,False)
    layer_names=vis.get_saved_layer_names()
    
    for idx in range(1,len(layer_names)-1):    
        # Get the layers
        layer=vis.conv_layers[layer_names[idx]]
        print(layer_names[idx])
       
        # Load IG matrix
        mat=np.load("IG/alexnet/IG_"+layer_names[idx+1]+"_class_0"+str(selected_class)+".npy")

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
    percentile_list=[2,5,10]
    
    selected_class=121
    selected_imagenet_class=703
    data_loader=get_class_dataLoader('n03891251/',20)
    model = models.alexnet(pretrained=True)
    model.eval()
    acc_before=[]
    acc_top=[]
    acc_bottom=[]

    """Top percentile Testing Block"""
    for percentile in percentile_list:
        model = models.alexnet(pretrained=True)
        model.eval()
        acc_before.append(model_eval(model,data_loader,selected_imagenet_class))
        ablate(model,selected_class,True,percentile)
        acc_top.append(model_eval(model,data_loader,selected_imagenet_class))
    
    """Bottom percentile Testing Block"""
    for percentile in percentile_list:
        model = models.alexnet(pretrained=True)
        model.eval()
        ablate(model,selected_class,False,percentile)
        acc_bottom.append(model_eval(model,data_loader,selected_imagenet_class))

    

    labels = ['Top-Bottom : 2%', 'Top-Bottom : 5%', 'Top-Bottom : 10%']
   
    x = np.arange(len(labels))  # the label locations
    width = 0.20  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x -0.40, acc_before, width, label='Before Ablation')
    rects1 = ax.bar(x - width/2, acc_top, width, label='Top')
    rects2 = ax.bar(x + width/2, acc_bottom, width, label='Bottom')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by Percentile')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects)
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()

   
    # # open file for writng the 
    # f = open(str(model.__class__.__name__)+"ablation_test.txt", 'w')
    # f.write("|class: "+str(selected_class)+" | Original Accuracy : = "+str(acc_after))
    # f.write("|class: "+str(selected_class)+" | Accuracy (Top : "+str(percentile)+") :  = "+str(acc_after))
    # f.write("\n")


    
    