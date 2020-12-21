import numpy as np 

class compute_iou:

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

