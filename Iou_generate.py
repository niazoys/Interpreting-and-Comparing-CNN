from utility import Utility
from compute_iou import compute_iou
from probe_model import probe_model
from dataloader import conceptLoader
import numpy as np 
from visualize_layers import VisualizeLayers
import os

if __name__ == "__main__":

       
        #create object for model probe
        pm=probe_model(True)

        # create activation generator & visualizer object
        '''
           Remember to change the 2nd argument to False for non Residual networks.(e.g. ALexnet ,VGG) 
        '''
        vis=VisualizeLayers(pm.get_model())
    
        #################
        # flag for hook tracking        
        existing_hook=False
            
        #dataset_path='D:\\Net\\NetDissect\\dataset\\broden1_227'
        dataset_path='broden1_227'
        
        # Create IOU directory for the network
        # model_name = pm.get_model().__class__.__name__
        # cwd = os.getcwd()  
        # dir = os.path.join(cwd,"IOU",model_name)
        # if not os.path.exists(dir):
        #     os.mkdir(dir)
        
        ###########################################
        # In how many part we want to do the calculation 
        part_ln=4
        # Total data 
        total_data=60000
        #Batch Size
        batch_size=200
        #number of iteratio
        iteration=int((total_data/batch_size))
        #iteration=int((total_data/batch_size)/(part_ln))
        ###########################################  


        #get the names of the layers in the network 
        layer_names=vis.get_saved_layer_names()
        #loop over all the convolutional layers  
        for layer in layer_names:
            print(layer)
            #check if there is already any hook attached and remove it 
            if existing_hook:
                vis.remove_all_hooks()
            
            # Intersection and union matrices intialization flag
            iou_mat_flag=True

            #attch hook for different cnn layers
            vis.hook_layers(layer)
            existing_hook=True
            iou=compute_iou(conceptLoader(dataset_path))
            
            iou_part_list=[]

            for part in range(1,part_ln+1):
                print("Processing(Part " +str(part)+"):"+ layer)
                featuremap=pm.probe(iteration=iteration,batch_size=batch_size,vis=vis,layer=layer,part_ln=part_ln,part=part)
                
                #Load the previously computed IoU 
                tk= np.load('TK/resnet18/tk_'+str(layer)+'.npy')

                #Get the logical map from the featuremap 
                featuremask=Utility.resize_image_bilinear_generate_mask_batch(featuremap,tk=tk)
                
                del featuremap
                
                # import matplotlib.pyplot as plt
                # for i in range(featuremask.shape[1]): 
                #     plt.imshow(featuremask[5,i,:,:])
                #     plt.show()


                #Intialize the Union and Intersection matrices for the first time.
                iou.intialize_matrices(featuremask.shape[1])
                
    
                #Compute Iou 
                iou.do_calculation(featuremask)
                
                del featuremask
                
                iou_part_list.append(iou.get_iou())

                #set the data counter to zero for next iteration
                pm.imLoader.data_counter=0

                

            #get the iou and save it 
            iou_full=np.vstack(iou_part_list)
        
            np.save('IOU/resnet18/iou_'+str(layer)+'.npy',iou_full)

            #Delete the Iou Object  
            del iou


# %%%
# import numpy as np
# from numpy import unravel_index
# mat=np.load('IOU/iou_Conv2d_Layer0.npy')  
# mat=np.array(mat)
# # mat=mat[:,1:,:]


# print(mat.shape) 
# print(mat.argmax())      
# print(unravel_index(mat.argmax(),mat.shape))
# print(mat[0,0,5])

# def get_unique_labels(annotation_map):
#     labels=np.unique(annotation_map)
#     idx = np.where(labels==0)
#     new_labels = np.delete(labels, idx)
#     return new_labels
# arr=[1,2,4,0,5,6,7]
# print(get_unique_labels(arr))


# %%
