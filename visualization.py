import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

class visualize_network():
    def __init__(self,net_name,nrows = 25):
        self.net_name         = net_name
        self.num_concept_type = 6
        self.nrows            = nrows


    def get_layer_list(self):
        '''
        returns the layer list of the network
        '''
        if self.net_name == 'alexnet':
            layer_list = ['Layer0','Layer3','Layer6','Layer8','Layer10']
        elif self.net_name == 'resnet18':
            layer_list = ['Layer0','Layer3','Layer6','Layer8','Layer10']
        else:
            layer_list = ['Layer0','Layer3','Layer6','Layer8','Layer10']
        
        return layer_list, np.size(layer_list)
    

    def vis_iou_score_dist_per_layer(self):
        '''
        generates a heatmap using the top IOU scores per layer in the network
        '''
        #get the names of the layers in the network 
        layer_names, nlayers = self.get_layer_list()

        # Parameters of the figure
        fig_ratio      = np.ones(nlayers)
        colorbar_ratio = np.array([0.08])
        width_ratios_=np.concatenate((fig_ratio,colorbar_ratio))
        fig, ax= plt.subplots(1,nlayers+1, 
                    gridspec_kw={'width_ratios':width_ratios_})

        for l in range(nlayers):

            #Load IOU of the current layer
            iou=np.load('IOU/alexnet/iou_Conv2d_'+str(layer_names[l])+'.npy')

            value = self.get_top_iou_per_unit(iou)
            
            g = sns.heatmap(value,cmap="YlGnBu",cbar=False,ax=ax[l])
            g.set_xlabel(str(l))
            g.set_xticks([])
            g.set_yticks([])
            if l==0:
                g.set_ylabel('Units')

        g_caxis = sns.heatmap(value,cmap="YlGnBu",ax=ax[nlayers-1], cbar_ax=ax[nlayers])
        g_caxis.set_ylabel('')
        g_caxis.set_xlabel('')
        g_caxis.set_yticks([])
        plt.show()


    def get_top_iou_per_unit(self,iou):

        top_iou = iou.max(axis=1).max(axis=1)
        top_iou = self.reshape_mat(top_iou)
        return top_iou

def generate_TopThreeIOU(iou,top):
    ''' Computes top three IOU score per unit
    '''
    iou_summary = {"concept_idx":np.zeros((top,iou.shape[0])),
    "concept_type":np.zeros((top,iou.shape[0]))
                    }
    for u in range(iou.shape[0]):
        U_iou = iou[u,:,:]

        for t in range(top):
            idx =unravel_index(U_iou.argmax(), U_iou.shape)
            iou_summary["concept_idx"][t,u]=idx[0]
            iou_summary["concept_type"][t,u]=idx[1]
            U_iou[idx]=0

    return iou_summary

    def reshape_mat(self,mat):
        n_unit   = mat.shape[0]
        padding  = self.nrows-n_unit%self.nrows
        temp_mat = np.concatenate((mat,-1*np.ones(padding)))
        new_mat  = temp_mat.reshape(self.nrows,-1)
        return new_mat


if __name__ == "__main__":
    a = visualize_network('alexnet')
    a.vis_iou_score_dist_per_layer()
