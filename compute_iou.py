import numpy as np 

class compute_iou:

    def __init__(self,conLoader,num_label=1200):
        self.conLoader=conLoader
        self.num_label=num_label
        self.index_counter=0

    def IoU(self,activation,concept_map,unit,label):
        '''Input params: Activation map of a unit for an image 
                        Concept annotation map
                        for which Unit iou being calculated
                        for which label IoU being computed
        '''
        self.intersection_score[unit,label]+=np.sum(np.logical_and(activation,concept_map))
        self.union_score[unit,label]+=np.sum(np.logical_or(activation,concept_map))

    def intialize_matrices(self,shape):
        """instantiate the necessary ndarrays"""
        
        self.intersection_score=np.zeros((shape,self.num_label),dtype=np.float16)
        self.union_score=np.zeros((shape,self.num_label),dtype=np.float16)


    def do_calculation(self,featuremask):
        '''Goes through the featuremask ,loads the corresponding Annotation mask and Computes intersection and union with the help of IoU() method '''


        #Define the concept list 
        concept_pixel_level_types = ["color","object","part","material","scene","texture"]
        
        #iterate over all the concepts
        for i in range(featuremask.shape[0]):
            #iterate over the Concept types
            for concept in concept_pixel_level_types:
                flag,concept_image=self.conLoader.load_concept(i,concept)
                labels=np.unique(concept_image)

                #chech if the concept is available
                if flag:

                    #check if the concept is scene or textures
                    if concept != "scene" or "texture":
                        for unit in range (featuremask.shape[1]):
                            for label in labels:
                                self.IoU(featuremask[i,unit,:,:],concept_image==label,unit,int(float(label)))
                    #check if the concept is a part type 
                    elif concept== "part":
                        for part in range(len(featuremask)):
                            part_labels=np.unique(featuremask[part])
                            for unit in range (featuremask.shape[1]):    
                                for label in part_labels:
                                    self.IoU(featuremask[i,unit,:,:],concept_image==label,unit,int(float(label)))
                    else:
                        for unit in range (featuremask.shape[1]):
                            for label in labels:
                                one_matrix=np.ones((113,113))
                                self.IoU(featuremask[i,unit,:,:],one_matrix,unit,label)       

    def get_iou(self):
        return self.intersection_score/self.union_score