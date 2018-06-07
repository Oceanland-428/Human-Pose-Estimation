
# coding: utf-8

# In[1]:

import numpy as np
import os
import cv2
def noisy(image,noise_typ):
    '''
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    '''
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


# In[ ]:

def data_augmentation(images,ytrues,times=2):
    '''
    inputs:
    images as a whole, for lsp dataset it is N*227*227*3
    ytrues as a whole, for lsp dataset it is N*2*14
    times: the times that you want to augment. Not include itself.
    
    return new_images: (N*times)*227*227*3
           new_labels: (N*times)*2*14
    '''
    N,_,_,_ = images.shape
    
    new_n = N*(times+1)
    new_images = np.zeros((new_n,227,227,3))
    new_labels = np.zeros((new_n,2,14))
    
    for i in range(N):
        
        temp_image = images[i]  # 227*227*3
        temp_label = ytrues[i] # 2*14
        
        new_images[i*(times+1)] = temp_image
        new_labels[i*(times+1)] = temp_label
        
        for j in range(times):
            
            temp_index = i*(times+1) + j+1
            temp_noise_images = noisy(temp_image,"gauss")
            new_images[temp_index] = temp_noise_images
            new_labels[temp_index] = temp_label
            
    return new_images,new_labels


