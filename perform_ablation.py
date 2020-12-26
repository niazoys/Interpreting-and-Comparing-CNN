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
        [transforms.Resize(256),
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


def autolabel(ax,rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def ablate(model,selected_class,Top,percentile):
   
    vis=VisualizeLayers(model)
    layer_names=vis.get_saved_layer_names()
    print(layer_names)
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
            #layer.bias.data[num_unit]=0


   

if __name__ == "__main__":
    percentile_list=[0.5,1,2]
    
    broden_class={'bench': 121, 'sea': 135, 'boat': 123,'refrigerator':191,'lamp':50}
    imagenet_class={'bench': 703, 'sea': 978, 'boat': 554,'refrigerator':760,'lamp':846}


    for broden_name, broden_label in broden_class.items():
        for imgnet_name, imgnet_label in imagenet_class.items():   

            data_loader=get_class_dataLoader(str(imgnet_name)+'/',20)
          
            model = models.alexnet(pretrained=True)
            model.eval()
            acc_before=[]
            acc_top=[]
            acc_bottom=[]

            """Top percentile Testing Block"""
            for percentile in percentile_list:
                model = models.alexnet(pretrained=True)
                model.eval()
                acc_before.append(model_eval(model,data_loader,imgnet_label))
                ablate(model,broden_label,True,percentile)
                acc_top.append(model_eval(model,data_loader,imgnet_label))
            
            """Bottom percentile Testing Block"""
            for percentile in percentile_list:
                model = models.alexnet(pretrained=True)
                model.eval()
                ablate(model,broden_label,False,percentile)
                acc_bottom.append(model_eval(model,data_loader,imgnet_label))

            
            ##
            ## ploting Block 
            ##

            labels = ['Top-Bottom :'+str(percentile_list[0])+'%','Top-Bottom :'+str(percentile_list[1])+'%','Top-Bottom :'+str(percentile_list[2])+'%']
        
            x = np.arange(len(labels))  # the label locations
            width = 0.20  # the width of the bars

            fig, ax = plt.subplots(figsize=(9,9))
            rects = ax.bar(x -0.40, acc_before, width, label='Before Ablation')
            rects1 = ax.bar(x - width/2, acc_top, width, label='Top')
            rects2 = ax.bar(x + width/2, acc_bottom, width, label='Bottom')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Scores')
            ax.set_title('Ablation Performed on: '+str(broden_name)+'_Tested: '+str(imgnet_name)+'.jpg')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            #label the histogram bar
            autolabel(ax,rects)
            autolabel(ax,rects1)
            autolabel(ax,rects2)
            fig.tight_layout()
           
            plt.show() 
            #plt.savefig('output_imgs/vgg/ablation'+str(broden_name)+'_Tested'+str(imgnet_name)+'.jpg')

        # # open file for writng the 
        # f = open(str(model.__class__.__name__)+"ablation_test.txt", 'w')
        # f.write("|class: "+str(selected_class)+" | Original Accuracy : = "+str(acc_after))
        # f.write("|class: "+str(selected_class)+" | Accuracy (Top : "+str(percentile)+") :  = "+str(acc_after))
        # f.write("\n") 


    
    