import os
import cv2
import numpy as np

#The directory where all intermediate features are stored.
source_dir = '/media/data2/aifi/all_yoga_features/'

#Where the detected edges are stores.
save_dir   = '/data/coco/all_edges/'

# The operation on output of which layer

conv_type  = 'conv1_1'

image_list = os.listdir(source_dir)

for one_path in image_list:

    #Trival operations to find images.
    image_name = one_path
        
    image_path = source_dir +'/'+image_name+'/'+conv_type+'/'
    
    
    images = os.listdir(image_path)

    # The convolution kernel for morphologyEx algorithm
    kernel = np.ones((3,3),np.uint8)
    
    #Store all detected edges after conv_type, final edge will average that.
    all_edged = []
    
    for path in images:
    
        #Get an image afeter conv_type
        a_path = image_path+path

        im = cv2.imread(a_path)

        #Make image gray
        blurred = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        # Core part of this algorithm, learned it from internet. 
        # Didn't know well for the theoretical behind it yet. :)
        auto = cv2.Canny(blurred,40,255)
        
        auto = cv2.morphologyEx(auto, cv2.MORPH_CLOSE, kernel)

        all_edged.append(auto)                

    #Average and save it.
    edges = np.mean(all_edged,axis=0).astype(np.uint8)

    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    
    cv2.imwrite('./all_edges/'+image_name+'_'+conv_type+'.png',edges)
