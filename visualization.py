import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from numpy import unravel_index
import os


class visualize_network():
    def __init__(self,net_name,top = 3,nrows = 32):
        self.net_name         = net_name
        self.dir_path         = os.path.join('IOU',net_name)    # Intersection over union scores directory
        self.ig_path          = os.path.join('IG',net_name)     # Integrated gradient scores directory
        self.num_concept_type = 6                               # [color,object,part,material,scene,texture]
        self.top              = top 
        self.nrows            = nrows                           # In figure, No. rows to plot per column.
                                                                # for 64 neurons, if nrow=32, then new matrix will be 32x2  
   
    def vis_iou_score_dist_per_layer(self):
        '''
        generates a heatmap using the top IOU scores per layer in the network
        '''
        layer_names = os.listdir(self.dir_path)
        nlayers     = np.size(layer_names)

        # # Parameters of the figure
        width_ratios_=self.get_width_ratios()
        fig, ax= plt.subplots(1,nlayers+1, 
                    gridspec_kw={'width_ratios':width_ratios_})

        for l in range(nlayers):

            #Load IOU of the current layer
            iou=np.load(os.path.join(self.dir_path,layer_names[l]))

            # value = self.get_top_iou_per_unit(iou)


            iou_summary = self.generate_TopThreeIOU(iou)
            # value       = np.squeeze(iou_summary["unit_iou_pair"][0,:])
            value       = np.squeeze(iou_summary["concept_type"][0,:])
            value       = self.check_uniqueness(value)
            value       = self.reshape_mat(value)
            bounds      = [0.5,1.5,2.5,3.5,4.5]
            print(np.unique(value))

            if l==nlayers-1:
                # g = sns.heatmap(value,ax=ax[l], cbar_ax=ax[nlayers])
                g = sns.heatmap(value,ax=ax[l], cbar_ax=ax[nlayers], cbar_kws= {"boundaries": bounds,"ticks":[]})
            else:
                g = sns.heatmap(value,cbar=False,ax=ax[l])
            g.set_xlabel('Layer_'+str(l))
            g.set_xticks([])
            g.set_yticks([])
            tl = g.get_xlabel()
            g.set_xlabel(tl, rotation=90)
            if l==0:
                g.set_ylabel('Units')

        fig.suptitle("QI score map of "+str(self.net_name))
        plt.show()
        # plt.savefig('gui_resources/'+self.net_name+'_iou_layerwise_dist/'+self.net_name+'.png')

    def check_uniqueness(self,value):
        '''
        Checks uniqueness of the matrix before plotting to avoid misleading colorbar
        '''
        k = np.unique(value)
        for i in range(1,5):
            q=True
            for j in k:
                if (i==j):
                    q = False
                    break
            if(q):
                idx = np.random.randint(1,len(value),1)
                value[idx]=i
        return value


    def get_top_iou_per_unit(self,iou):
        '''
        returns the top iou score of each unit in a layer
        '''
        iou = np.nan_to_num(iou)
        top_iou = iou.max(axis=1).max(axis=1)
        top_iou = self.reshape_mat(top_iou)
        return top_iou

    def generate_TopThreeIOU(self,iou):
        ''' Computes top three IOU score per unit
        '''
        iou = np.nan_to_num(iou)
        iou_summary = {"concept_idx":np.zeros((self.top,iou.shape[0])),
        "concept_type":np.zeros((self.top,iou.shape[0])),
        "unit_iou_pair":np.zeros((self.top,iou.shape[0]))}
        
        for u in range(iou.shape[0]):
            U_iou = iou[u,:,:]

            for t in range(self.top):
                idx =unravel_index(U_iou.argmax(), U_iou.shape)
                iou_summary["concept_idx"][t,u]   = idx[0]
                iou_summary["concept_type"][t,u]  = idx[1]+1
                iou_summary["unit_iou_pair"][t,u] = U_iou[idx]
                U_iou[idx]=0

        return iou_summary

    def get_width_ratios(self):
        layer_names = os.listdir(self.dir_path)
        nlayers     = np.size(layer_names)

        # Parameters of the figure
        fig_ratio      = np.ones(nlayers)
        colorbar_ratio = np.array([0.08])
        
        for l in range(nlayers):
            #Load IOU of the current layer
            iou       = np.load(os.path.join(self.dir_path,layer_names[l]))
            fig_ratio[l] = iou.shape[0]
        
        fig_ratio    = fig_ratio/max(fig_ratio)
        width_ratios_= np.concatenate((fig_ratio,colorbar_ratio))

        return width_ratios_

    def reshape_mat(self,mat):
        '''
        Reshapes the matrix into a new shape to fit into the plotting window
        for 64 neurons, if nrow=32, then new matrix will be 32x2
        '''
        new_mat  = mat.reshape(self.nrows,-1)
        return new_mat


    def  vis_concept_dist_per_layer(self):
        '''
        plot the concept type summary per layer in the network
        ''' 
        names,values = self.get_concept_summary()

        fig_name = os.path.join("gui_resources",self.net_name+"_iou_dist")
        for t in range(self.top):
            fig_title = "Concept type Distribution: "+self.net_name +" (Top "+str(t+1)+")"
            plt.subplots(figsize=(9,7))


            plt.plot(names,values[t,0,:],'o--',label="color")
            plt.plot(names,values[t,1,:],'o--',label="Object")
            plt.plot(names,values[t,2,:],'o--',label="Part")
            plt.plot(names,values[t,3,:],'o--',label="Material")

            plt.xticks(rotation=45)
            # rotate x-axis labels by 45 degrees
            plt.suptitle(fig_title)
            plt.legend()
            plt.show()
            # plt.savefig(fig_name+"/Top_"+str(t+1)+".png")

    def get_concept_summary(self):
        ''' Gives Layer names and a summary of no. of concepts per layer
        output:
        values = top * num_concept  *  no. of layers
        names  = layer names of the networks
        '''
        layer_names = os.listdir(self.dir_path)
        nlayers     = np.size(layer_names)

        values = np.zeros((self.top,self.num_concept_type,nlayers))
        names  = []

        for l in range(nlayers):

            #Load IOU of the current layer
            iou   = np.load(os.path.join(self.dir_path,layer_names[l]))

            value = self.get_summary(iou)
            values[:,:,l] = value
            names.append('Layer'+str(l))           

        return names,values 

    def get_summary(self,iou):
        '''
        computes a summary: no. of concepts per concept type per layer
        '''
        #shape of tp_iou = #top x #No. of units
        tp_iou = self.generate_TopThreeIOU(iou)
        tp_iou = tp_iou["concept_type"]
        c= np.zeros([self.top,self.num_concept_type])
        for t in range(self.top):
            temp = np.array(tp_iou[t,:])
            for i in range (self.num_concept_type):
                c[t,i] = sum(temp==i)
        return c


    def vis_classwise_IG_dist(self):
        '''
        genarates a classwise IG map per layer
        '''
        class_list  = [121,135,123,191,50,519,203,70,88,105]
        
        # class_list  = [50]
        layer_names = os.listdir(self.dir_path)
        nlayers     = np.size(layer_names)

        # Parameters of the figure
        fig_ratio      = np.ones(nlayers)
        colorbar_ratio = np.array([0.08])
        width_ratios_=np.concatenate((fig_ratio,colorbar_ratio))
        fig, ax= plt.subplots(1,nlayers+1, 
                    gridspec_kw={'width_ratios':width_ratios_})

        #---------------------------------------------------------------------------------------
        values_range = np.zeros((len(class_list),nlayers))
        for l in range(nlayers):
            layer = layer_names[l]
            layer = layer[4:-4]
            for c in range(len(class_list)):
                ig = np.load(self.ig_path+'/IG_'+layer+'_class_0'+str(class_list[c])+'.npy')
                mat = ig
                percentile = 25
                threshold=np.quantile(mat,1-percentile/100)
                itemindex = np.where(mat>threshold)
                values_range[c,l] = sum(mat[itemindex])

        # temp = np.zeros_like(values_range)
        # for i in range(temp.shape[0]):
        #     temp[i,:] = values_range[i,:]/max(values_range[i,:])
        # values_range = temp
        #-------------------------------------------------------------------------------------------------------------------------------
        for l in range(nlayers):
            a=values_range[:,l]
            a = a[:, np.newaxis]

            if l==nlayers-1:
                g = sns.heatmap(a,vmax=values_range.max(),vmin=values_range.min(),annot=True ,ax=ax[l], cbar_ax=ax[l+1])
            else:
                g = sns.heatmap(a,vmax=values_range.max(),vmin=values_range.min(),annot=True,cbar=False,ax=ax[l])
            g.set_xlabel('Layer_'+str(l))
            g.set_xticks([])
            g.set_yticks([])
            tl = g.get_xlabel()
            g.set_xlabel(tl, rotation=90)
            if l==0:
                g.set_ylabel(class_list)
                tl = g.get_ylabel()
                g.set_ylabel(tl, rotation=90)


        fig.suptitle(self.net_name)
        plt.show()        
        return values_range

if __name__ == "__main__":
    # a = visualize_network('alexnet')
    # a = visualize_network('resnet18')
    a = visualize_network('vgg11')

    # a.vis_iou_score_dist_per_layer()
    # a.vis_concept_dist_per_layer()

    print(a.vis_classwise_IG_dist().shape)
