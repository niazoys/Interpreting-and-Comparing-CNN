from probe_model import probe_model
from dataloader import conceptLoader
import numpy as np 
from Layer_hooker import Hooker
import gc
import os


class ComputeTK():

    def __init__(self):
        #create object for model probe
        self.feature_generator=probe_model(True)
        # create activation generator & self.hookerualizer object
        self.hooker=Hooker(self.feature_generator.get_model())
      
    def main(self):
        ###########################################
        # falg for hook tracking        
        existing_hook=False
        # Total data 
        total_data=60000
        #Batch Size
        batch_size=50
        #number of iteration
        iteration=int((total_data/batch_size))
        # in how many parts we want to do the calculation (Must be even Number)
        part_ln = 16
        ###########################################  

        #get the names of the layers in the network 
        layer_names=self.hooker.get_saved_layer_names()
        
        for layer in layer_names:
            print(layer)
            tk=[]
            #loop over all the convolutional layers  
            for part in range (1,part_ln+1):            
                print("Processing(Part " +str(part)+"):"+ layer)
                
                #check if there is already any hook attached and remove it 
                if existing_hook:
                    self.hooker.remove_all_hooks()

                #attch hook for different cnn layers
                self.hooker.hook_layers(layer)
                existing_hook=True

                #Generate tk 
                featuremap=self.feature_generator.probe(iteration=iteration,batch_size=batch_size,hooker=self.hooker,layer=layer,part_ln=part_ln,part=part)

                print("Generating Tk")
                #Generate tk and save them
                for unit in range(featuremap.shape[1]):
                    tk.append(np.quantile(featuremap[:,unit,:,:],0.995))
                
                #clear the featuremap from memory 
                del featuremap
                
                #Reset the image loading counter to zero     
                self.feature_generator.imLoader.data_counter=0
                gc.collect()
            
            # save TK matrix    
            np.save('TK/vgg11/tk_'+str(layer)+'.npy',tk)
                



if __name__ == "__main__":
    tk=ComputeTK()
    tk.main()