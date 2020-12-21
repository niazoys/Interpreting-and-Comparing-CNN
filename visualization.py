from probe_model import probe_model
from visualize_layers import VisualizeLayers
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_TopThreeIOU(iou,top):
    ''' Computes top three IOU score per unit
    '''
    TopThreeIOU = np.zeros([top,iou.shape[0]])

    for u in range(iou.shape[0]):
        U_iou = iou[u,:,:]

        for t in range(top):
            idx =unravel_index(U_iou.argmax(), U_iou.shape)
            TopThreeIOU[t,u]=idx[1]
            U_iou[idx]=0

    return TopThreeIOU


def get_summary(iou,top, num_concept_type):
    '''Takes input the iou, concept type and top value
    computes a summary: no. of concepts per concept type
    '''

    #shape of tp_iou = #top x #No. of units
    tp_iou = generate_TopThreeIOU(iou,top)

    c= np.zeros([top,num_concept_type])
    for t in range(top):
        temp = np.array(tp_iou[t,:])
        for i in range (num_concept_type):
            c[t,i] = sum(temp==i)
    return c


def get_concept_summary(top, num_concept_type,Network_name):
    ''' Gives Layer names and a summary of no. of concepts per layer
    output:
    values = top * num_concept  *  no. of layers
    names  = layer names of the networks
    '''

    #create object for model probe
    pm=probe_model(True)
    vis=VisualizeLayers(pm.get_model())
    #get the names of the layers in the network 
    layer_names=vis.get_saved_layer_names()

    values = np.zeros((top,num_concept_type,np.size(layer_names)))

    for l in range(np.size(layer_names)):

        #Load IOU of the current layer
        iou=np.load('IOU/alexnet/iou_'+str(layer_names[l])+'.npy')
        #iou=np.load('IOU/iou_Conv2d_Layer0.npy')  

        value = get_summary(iou,top,num_concept_type)
        values[:,:,l] = value

    return layer_names,values    

if __name__ == "__main__":
    top = 3
    num_concept_type = 4
    # Get Layer names and a summary of no. of concepts per layer

    # values = np.random.randint(10, size=(top,num_concept,5))
    names,values = get_concept_summary(top,num_concept_type,"resnet18")


    for t in range(top):
        plt.figure(figsize=(15, 3))
        plt.suptitle("Top "+str(t+1))

        p1,=plt.plot(names,values[t,0,:],'o--',label="color")
        p2,=plt.plot(names,values[t,1,:],'o--',label="Object")
        p3,=plt.plot(names,values[t,2,:],'o--',label="Part")
        p4,=plt.plot(names,values[t,3,:],'o--',label="Material")

        # Place a legend above this subplot, expanding itself to
        # fully use the given bounding box.
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right',
            ncol=1, mode="expand", borderaxespad=0.)
        plt.show()