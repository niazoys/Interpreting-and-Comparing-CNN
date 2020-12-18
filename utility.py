import numpy as np 
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt 

class Utility ():

    @staticmethod
    def resize_image_bilinear_generate_mask(images_batch,tk,img_shape=[113,113]):
        '''
        processes single neuron
        resize the image with bilinear interpolation and generate the logical featuremap by comparing with tk
        input  : float16/32 image batch & TK . Input dimension is a 3D numpy array as following (BatchSize * height * Width)
        output : float 16 image batch  & logical featureMask 
        '''
        resized_image=np.zeros((images_batch.shape[0],img_shape[0],img_shape[1]),dtype=np.float16)
        feature_mask=np.zeros((images_batch.shape[0],img_shape[0],img_shape[1]),dtype=np.bool)
        for num in range (images_batch.shape[0]):
            current_img=cv2.resize((images_batch[num,:,:]).astype(np.float32), dsize=(113, 113), interpolation=cv2.INTER_CUBIC)
            resized_image[num]=current_img
            feature_mask[num]=current_img>=tk[0]
            
        return resized_image,feature_mask
        
    @staticmethod
    def resize_IG_batch(images_batch,img_shape=[113,113]):
        '''
        processes in layers
        resize the Integrated Gradients with bilinear interpolation 
        input  : IG of one entire layer. Input dimension is a 4D numpy array as following (BatchSize * Units * height * Width)
        output : resized IG
        '''
        resized_ig=np.zeros((images_batch.shape[0],images_batch.shape[1],img_shape[0],img_shape[1]))
        for im_num in range (images_batch.shape[0]):
            for unit_num in range(images_batch.shape[1]):
                resized_ig[im_num,unit_num,:,:] = cv2.resize( images_batch[im_num,unit_num,:,:] , dsize=(113, 113), interpolation=cv2.INTER_CUBIC)
        return resized_ig

    @staticmethod
    def resize_image_bilinear_generate_mask_batch(images_batch,tk,img_shape=[113,113]):
        ''' 
        processes in batch
        resize the image with bilinear interpolation and generate the logical featuremap by comparing with tk
        input  : float16 image batch & TK. Input dimension is a 4D numpy array as following (BatchSize * Units * height * Width)
        output : float 16 image batch  & logical featureMask 
        '''
       
        feature_mask=np.zeros((images_batch.shape[0],images_batch.shape[1],img_shape[0],img_shape[1]),dtype=np.bool)
        
        for im_num in range (images_batch.shape[0]):
            for unit_num in range(images_batch.shape[1]):
                current_img=cv2.resize((images_batch[im_num,unit_num,:,:]).astype(np.float32), dsize=(113, 113), interpolation=cv2.INTER_CUBIC)
                feature_mask[im_num,unit_num,:,:]=current_img>=tk[unit_num]
        #print("Feature Mask Generation Done.")
        return feature_mask
    
    @staticmethod
    def load_single_image (path,load_mask):
            '''Load a single image directly feed able to network''' 
            img = Image.open(path)

            if load_mask:            
                transform = transforms.Compose([
                transforms.ToTensor()
                ])
                input = np.array(img)
                input=(input[:,:,0] + (input[:,:,1])*256)
            else:
                transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                ])
                input  = transform(img)
                input = input.unsqueeze(0)

            # transform_normalize = transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
            #input = transform_normalize(input)


            return input
    
    @staticmethod
    def show_many_images(imgs,num,colormap_gray=True):
        """ Too show multiple images in grid fashion  """
        fig=plt.figure(figsize=(15, 15))
        columns = 6
        rows = 6
        for i in range(num):
            img = imgs[i,:,:]
            fig.add_subplot(rows, columns,i+1)
            if colormap_gray:
                plt.imshow(img,cmap='gray')
            else:
                plt.imshow(img)    
        plt.show()

    @staticmethod
    def normalize_image(img):
        '''image normalizer method '''
        img = img - img.min()
        img = img / img.max()
        return img
