import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

class Hooker(object):
    '''
    A class to visualize intermediate layer outputs
    '''
    def __init__(self,model):
        self.model=model
        self.interm_output = {}
        self.conv_layers=dict()
        if model.__class__.__name__=="ResNet":
            self.generate_layers_info(self.model)
        else:
            self.generate_layers_info_non_residualNetwork(self.model)
        if not os.path.exists('output_imgs'):
            os.makedirs('output_imgs')
    
    def __get_activation(self,name):
        def hook(model, input, output):
            self.interm_output[name] = output.detach()
        return hook

    def generate_layers_info_non_residualNetwork(self,model):
    
        '''
        Method iterates over the layer of model and put layer objects  in a list (Works for Non  resudual networks)
        
        Args:
            model - () - pytorch model or layers
        '''
        layer_counter=0
        for name, layer in model.features._modules.items():
            if type(layer) == nn.Conv2d:
                self.conv_layers[layer._get_name()+"_Layer"+str(name)]=layer
                layer_counter=layer_counter+1
    
    def generate_layers_info(self,model):
       
        '''
        Method iterates over the layer of model and put layer objects  in a list (Works for resudual networks)
        
        Args:
            model - () - pytorch model or layers
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
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            local_counter +=1
                            namepart="Layer"+str(layer_counter)+"_conv2d_"+str(local_counter)
                            self.conv_layers[namepart]=(child)
                            counter=+1
    
    def remove_all_hooks(self):
            self.hook_handle.remove()

    def hook_layers(self,layer_name=None):

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
    
