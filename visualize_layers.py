import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

class VisualizeLayers(object):
    '''
    A class to visualize intermediate layer outputs
    '''
    def __init__(self,model):
        self.model=model
        self.interm_output = {}
        self.conv_layers=dict()
        self.generate_layers_info(self.model)
        if not os.path.exists('output_imgs'):
            os.makedirs('output_imgs')
    
    def __get_activation(self,name):
        def hook(model, input, output):
            self.interm_output[name] = output.detach()
        return hook


    def generate_layers_info(self,model):
        '''
        
        '''
        #get the children list of model 
        model_children=list(model.children())

        layer_counter=0
        counter=0

        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                self.conv_layers[model_children[i]._get_name()+"_Layer"+str(layer_counter)]=model_children[i]
                counter=+1      
            elif type(model_children[i]) == nn.Sequential: 
                layer_counter += 1   
                local_counter = 0
                # tmp1=model_children[i]
                # print(tmp1)
                for j in range(len(model_children[i])):
                    # tmp2=model_children[i][j].children()
                    # print(tmp2)
                    for child in model_children[i][j].children():
                        
                        if type(child) == nn.Conv2d:
                            local_counter +=1
                            namepart="Layer"+str(layer_counter)+"_conv2d_"+str(local_counter)
                            self.conv_layers[namepart]=(child)
                            counter=+1
    

    def remove_all_hooks(self):
            self.hook_handle.remove()

    def hook_layers(self,layer_name=None):
        
        # Attach Hook to the conv layers 
        # if layer_name==None: 
        #     for name in self.conv_layers:
        #         hookself.conv_layers[name].register_forward_hook(self.__get_activation(name))
        # else:

        self.hook_handle=self.conv_layers[layer_name].register_forward_hook(self.__get_activation(layer_name))
        self.existing_hook_flag=True          
    
    def get_interm_output_aslist(self):
        '''
        function to get the intermediate layers(layers which were hooked) output
        
        returns:
            self.interim_output: (list)- a dictionary of intermediate layer 
            outputs which were hooked
        '''
        list_=[]
        for name in self.interm_output:
            list_.append(self.interm_output[name])
        
        self.interm_output.clear()
        return list_


    
    def get_interm_output(self):
        '''
        function to get the intermediate layers(layers which were hooked) output
        
        returns:
            self.interim_output: (dict)- a dictionary of intermediate layer 
            outputs which were hooked
        '''
        return self.interm_output
        
    def get_saved_layer_names(self):
        '''
        function to get the intermediate layer names whose output are saved
        Returns:
            #self.interm_output_keys(): (dict_keys)
            self.layer_names: (list) - list of all the saved layer names 
        '''   
        names=[]
        for name in self.conv_layers:
            names.append(name)
        return names
    

    def plot_single_featuremaps(self,featuremaps,unit_num,name,color_map='color',savefig=False,figsize=12):
        """
            shows featuremap for single neuron 
        """
        featuremaps=featuremaps.squeeze()
        title_name ="Activation_Map_for_{}_and_unit_{}".format(name,str(unit_num))
        
        if color_map=='color':
            color_map='viridis'

        plt.imshow(featuremaps[unit_num,:,:],cmap=color_map)
        plt.title(title_name)
        plt.colorbar()
       
        if savefig:
            plt.savefig("output_imgs/"+title_name+".jpg")

        plt.show()

    def plot_featuremaps(self,featuremaps,name='featuremaps',color_map='color',savefig=False,figsize=12):
        '''
        function to plot the feature maps of an intermediate layer
        Args:
           featuremaps: (torch.tensor) - a tensor of shape [1,64,55,55] 
                       representing (Batch_size, num_featuremaps,
                       height of each featuremap,width of each featuremap)
            name: (string) - name of the feature map
            color_map: (string) - 'gray' or 'color' - 'color' is 'viridis' format
            savefig: (Bool) - True or False , whether or not you want to save 
                      the fig
            figsize: (int) - figure size in th form of (figsize,figsize)
        '''
        featuremaps=featuremaps.squeeze()
        num_feat_maps=featuremaps.size(0)
        subplot_num=int(np.ceil(np.sqrt(num_feat_maps)))
   
        plt.figure(figsize=(figsize,figsize))
        for idx,f_map in enumerate(featuremaps):
            plt.subplot(subplot_num,subplot_num, idx + 1)
            if color_map=='color':
                color_map='viridis'
            plt.imshow(f_map,cmap=color_map)

        print ('Number of output maps: {}'.format(num_feat_maps))

        if savefig:
            plt.savefig("output_imgs/{}".format(name) + '.jpg')
        
        plt.show()

#%%        
# if __name__=='__main__':  
    
#     # load the Pytorch model
#     model = models.resnet18(pretrained=True)
#     # create an object of VisualizeLayers and initialize it with the model and 
#     # the layers whose output you want to visualize
    
#     vis = VisualizeLayers(model)
#     # load the input
#     x = torch.randn([1,3,224,224])
#     # pass the input and get the output
#     output = model(x)
#     # get the intermediate layers output which was passed during initialization
#     interm_output = vis.get_interm_output()
    
#     # plot the featuremap of the layer which you want, to see what are the layers
#     # saved simply call vis.get_saved_layer_names
#     name_list=vis.get_saved_layer_names()
    
#     vis.plot_featuremaps(interm_output[name_list[0]],name='noise_inpt_color_fmap-1',color_map='color',savefig=True)
    
    

