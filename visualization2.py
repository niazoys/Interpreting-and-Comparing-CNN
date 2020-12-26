from probe_model import probe_model
from visualize_layers import VisualizeLayers
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def gen_IG_visualization(IG_table,Layer_names,Class_labels):
    
    '''
    It plots the IG score summary 
    IG_table dimension = #class_label x # Layer_names

    '''
    fig, ax = plt.subplots(figsize=(5,8))
    xticklabels=Layer_names
    yticklabels=Class_labels
    ax = sns.heatmap(IG_table, xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    plt.xlabel('Layers')
    plt.ylabel('Class')
    plt.show()

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
    computes a summary: no. of concepts per concept type per layer
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
    pm=probe_model(Network_name)
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

def find_top_unit(iou,c_idx,t_percentage):
    ''' 
    takes input iou of a layer, concept id and the percentage
    value which will determine the number of units with max iou
    score to find out 
    '''

    # num of top units to compute
    num_unit  = np.int(np.round(iou.shape[0]*t_percentage/100))
    unit_list = np.zeros((num_unit)) 
    
    #iou score list of desired concept index
    temp = iou[:,c_idx-1,:]
    temp[np.isnan(temp)] = 0

    for u in range(num_unit):
        idx          = unravel_index(temp.argmax(), temp.shape)
        unit_list[u] = idx[0]
        temp[idx]    = 0    #Setting zero to discard from next iteration
    
    return unit_list

def find_top_unit_IG(IG,t_percentage):
    '''
    Takes input a IG map and a percentage. Based on the percentage, computes the 
    top units. For instance, 64 unit IG with a percentage of 10%, will return a 
    list of location with top 6 unit's index.
    '''
    IG=np.load("E:\TRDP_II\ICNN\IG/alexnet/IG_Conv2d_Layer6_class_0101.npy")
    # num of top units to compute
    num_unit  = np.int(np.round(IG.shape[0]*t_percentage/100))

    unit_list = np.zeros((num_unit)) 

    for u in range(num_unit):
        idx          = unravel_index(IG.argmax(), IG.shape)
        unit_list[u] = idx[0]
        IG[idx]    = 0    #Setting zero to discard from next iteration

    return unit_list

# THIS FUNCTIONS ARE NEEDED TO GENERATE VISUALIZATION FOR IOU

def reshape_mat(mat,nrows):
    n_unit   = mat.shape[0]
    padding  = nrows-n_unit%nrows
    temp_mat = np.concatenate((mat,-1*np.ones(padding)))
    new_mat  = temp_mat.reshape(nrows,-1)
    return new_mat

def vis_iou_score_dist_per_layer(Network_name):
    '''
    generates a heatmap using the top IOU scores per layer in the network
    '''
    #create object for model probe
    pm=probe_model(Network_name)
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
    fig_ratio      = np.ones(3)
    colorbar_ratio = np.array([0.08])
    width_ratios_=np.concatenate((fig_ratio,colorbar_ratio))
    fig, ax= plt.subplots(1,4, 
                gridspec_kw={'width_ratios':width_ratios_})

    for i in range(3):
        g = sns.heatmap(a,cmap="YlGnBu",cbar=False,ax=ax[i])
        g.set_xlabel(str(i))
        g.set_xticks([])
        g.set_yticks([])
        if i==0:
            g.set_ylabel('Units')

    g3 = sns.heatmap(a,cmap="YlGnBu",ax=ax3, cbar_ax=ax[3])
    g3.set_ylabel('')
    g3.set_xlabel('')
    g3.set_yticks([])
    plt.show()

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
    


    a = np.random.randint(1,10,size=(5,8))
    Layer_names = ['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8']
    Class_labels = ['tina','mona','ayes','ms','pudding']
    gen_IG_visualization(a,Layer_names,Class_labels)