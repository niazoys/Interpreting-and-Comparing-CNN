from utility import Utility
from probe_model import probe_model
from dataloader import conceptLoader
import numpy as np 
from Layer_hooker import Hooker
import os

import numpy as np 

class ComputeIoU:

    def __init__(self,conLoader,num_label=1200):
        self.conLoader=conLoader
        self.num_label=num_label

    def IoU(self,activation,concept_map,unit,label,concept_number):
        '''Input params: Activation map of a unit for an image 
                        Concept annotation map
                        for which Unit iou being calculated
                        for which label IoU being computed
        '''
        temp_is=np.sum(np.logical_and(activation,concept_map))
        self.intersection_score[unit,label,concept_number]+=temp_is
        
        temp_us=np.sum(np.logical_or(activation,concept_map))
        self.union_score[unit,label,concept_number]+=temp_us

    def intialize_matrices(self,shape):
        """instantiate the necessary ndarrays"""
        
        self.intersection_score=np.zeros((shape,self.num_label,6),dtype=np.float16)
        self.union_score=np.zeros((shape,self.num_label,6),dtype=np.float16)
    
    def get_unique_labels(self,annotation_map):
        labels=np.unique(annotation_map)
        # idx = np.where(labels==0)
        # new_labels = np.delete(labels, idx)
        return labels

    def do_calculation(self,featuremask):
        '''Goes through the featuremask ,loads the corresponding Annotation mask and Computes intersection and union with the help of IoU() method '''


        #Define the concept list 
        concept_pixel_level_types = ["color","object","part","material","scene","texture"]
        
        #iterate over all the concepts
        for i in range(featuremask.shape[0]):
            #iterate over the Concept types
            concept_num=0
            for concept in concept_pixel_level_types:
                flag,concept_image=self.conLoader.load_concept(i,concept)
                

                #chech if the concept is available
                if flag:
                    #check if the concept is scene or textures
                    if concept == "color" or  concept =="object" or  concept == "material":              
                        labels=self.get_unique_labels(concept_image)
                        for unit in range (featuremask.shape[1]):
                            for label in labels:
                                self.IoU(featuremask[i,unit,:,:],concept_image==label,unit,int(float(label)),concept_num)
                    #check if the concept is a part type 
                    elif concept=="part":
                        for part in range(len(concept_image)):
                            part_labels=self.get_unique_labels(concept_image[part])
                            for unit in range (featuremask.shape[1]):    
                                for label in part_labels:
                                    self.IoU(featuremask[i,unit,:,:],concept_image[part]==label,unit,int(float(label)),concept_num)
                    else:
                        labels=(self.get_unique_labels(concept_image))
                        for unit in range (featuremask.shape[1]):
                            for label in labels:
                                label=int(float(label))
                                one_matrix=np.ones((113,113))
                                self.IoU(featuremask[i,unit,:,:],one_matrix,unit,label,concept_num)    
                concept_num+=1 
                      
    def get_iou(self):
        return self.intersection_score/self.union_score


if __name__ == "__main__":

       
        #create object for model probe
        pm=probe_model(True)

        # create Layer info generator and layer hooker object
        hooker=Hooker(pm.get_model())
    
        #################
        # flag for hook tracking        
        existing_hook=False
            
        dataset_path='D:\\Net\\NetDissect\\dataset\\broden1_227'
        #dataset_path='broden1_227'
        
     
        ###########################################
        # In how many part we want to do the calculation 
        part_ln=16
        # Total data 
        total_data=60000
        #Batch Size
        batch_size=50
        #number of iteratio
        iteration=int((total_data/batch_size))
        ###########################################  


        #get the names of the layers in the network 
        layer_names=hooker.get_saved_layer_names()
        #loop over all the convolutional layers  
        for layer in layer_names:
            print(layer)
            #check if there is already any hook attached and remove it 
            if existing_hook:
                hooker.remove_all_hooks()
            
            # Intersection and union matrices intialization flag
            iou_mat_flag=True

            #attch hook for different cnn layers
            hooker.hook_layers(layer)
            existing_hook=True
            iou=ComputeIoU(conceptLoader(dataset_path))
            
            iou_part_list=[]

            for part in range(1,part_ln+1):
                print("Processing(Part " +str(part)+"):"+ layer)
                featuremap=pm.probe(iteration=iteration,batch_size=batch_size,hooker=hooker,layer=layer,part_ln=part_ln,part=part)
                
                #Load the previously computed IoU 
                tk= np.load('TK/vgg11/tk_'+str(layer)+'.npy')

                #Get the logical map from the featuremap 
                featuremask=Utility.resize_image_bilinear_generate_mask_batch(featuremap,tk=tk)
                
                del featuremap

                #Intialize the Union and Intersection matrices for the first time.
                iou.intialize_matrices(featuremask.shape[1])
                
    
                #Compute Iou 
                iou.do_calculation(featuremask)
                
                del featuremask
                
                iou_part_list.append(iou.get_iou())

                #set the data counter to zero for next iteration
                pm.imLoader.data_counter=0

                

            #stack the appended iou  and save it 
            iou_full=np.vstack(iou_part_list)
        
            np.save('IOU/vgg11/iou_'+str(layer)+'.npy',iou_full)

            #Delete the Iou Object  
            del iou

