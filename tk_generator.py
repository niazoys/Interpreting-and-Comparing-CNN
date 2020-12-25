from utility import Utility
from compute_iou import compute_iou
from probe_model import probe_model
from dataloader import conceptLoader
import numpy as np 
from visualize_layers import VisualizeLayers
from compute_qd import Compute_qd
import gc
import os

if __name__ == "__main__":

        ###########################################
        #create object for model probe
        pm=probe_model(True)

        # create activation generator & visualizer object
        '''
           Remember to change the 2nd argument to False for non Residual networks.(e.g. ALexnet ,VGG) 
        '''
        vis=VisualizeLayers(pm.get_model())
        
        # falg for hook tracking        
        existing_hook=False
            
        
        # Total data 
        total_data=60000
        #Batch Size
        batch_size=150
        #number of iteratio
        iteration=int((total_data/batch_size))

        # in how many parts we want to do the calculation (Must be even Number)
        part_ln = 16
        
        ###########################################  

        #get the names of the layers in the network 
        layer_names=vis.get_saved_layer_names()
        
        # # Create Tk directory for the network
        # model_name = pm.get_model().__class__.__name__
        # cwd = os.getcwd()  
        # dir = os.path.join(cwd,"Tk",model_name)
        # if not os.path.exists(dir):
        #     os.mkdir(dir)


        for layer in layer_names:
            print(layer)
            tk=[]
            #loop over all the convolutional layers  
            for part in range (1,part_ln+1):            
                print("Processing(Part " +str(part)+"):"+ layer)
                
                #check if there is already any hook attached and remove it 
                if existing_hook:
                    vis.remove_all_hooks()

                #attch hook for different cnn layers
                vis.hook_layers(layer)
                existing_hook=True

                #Generate tk 
                featuremap=pm.probe(iteration=iteration,batch_size=batch_size,vis=vis,layer=layer,part_ln=part_ln,part=part)

                print("Generating Tk")
                #Generate tk and save them
                for unit in range(featuremap.shape[1]):
                    tk.append(np.quantile(featuremap[:,unit,:,:],0.995))
                
                #clear the featuremap from memory 
                del featuremap
                
                #Reset the image loading counter to zero     
                pm.imLoader.data_counter=0
                gc.collect()
            
            # save TK matrix    
            np.save('TK/vgg11/tk_'+str(layer)+'.npy',tk)
                
