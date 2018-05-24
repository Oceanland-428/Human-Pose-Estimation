
# coding: utf-8

# In[2]:

import argparse
import glob
import numpy as np
import re
import matplotlib.image as img
from scipy import misc
import matplotlib.pyplot as plt
import skimage.transform
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from os.path import basename as b
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2 as cv


# In[3]:

def compute_dimy(Y_pred):
    '''
    compute the dim(y) mentioned in the paper
    
    input: Y_pred a 2*14 matrix indicating predicted joints of an image
    
    return: dim(y) of the bounding box
    '''
    left_shoulder_index = 9
    right_hip_index = 2
    
    distance = np.sqrt(np.sum((Y_pred[:,right_hip_index]-Y_pred[:,left_shoulder_index])**2))
    return distance


# In[4]:

# # usage & test correctness
# dimY = compute_dimy(test_label)
# print(dimY)


# In[5]:

def get_reshape_size(Y_pred,dimY,sigma=2):
    '''
    compute the left width, left hight, right weight, right hight of the bounding box at each y (2*1) in Y (2*14) 
    
    input: Y is a 2*14 matrix(vector) indicating the (x,y) corrdinates of an predicted joint
           dimY is the size of the bounding box, see equation 7 in the deeppose paper
           sigma is the coefficient in front of dimY
           
    return reshape_size which is the left width, left hight, right weight, right hight of the bounding box at each y. 
    So reshape_size has shape 4*14 for lsp dataset
    '''
    # get the shape of Y, for lsp dataset it is 2*14
    yshape0,yshape1 = Y_pred.shape
    # set up the shape of return variable (reshape_size)
    reshape_size = np.zeros((4,yshape1))
    
    dimY = sigma*dimY
    # loop through every 2*1 element in Y
    for i in range(yshape1):
        # get x and y coordinates of a ith joint
        x = Y_pred[0,i]
        y = Y_pred[1,i]
        
        # get the size after reshape
        x_reshape_left = x-dimY/2
        x_reshape_right = x+dimY/2
        y_reshape_left = y-dimY/2
        y_reshape_right = y+dimY/2
        
        # adjust the size, cannot be out of bound
        if x_reshape_left<0:
            x_reshape_left = 0
        if x_reshape_right>227:
            x_reshape_right = 227
        if y_reshape_left<0:
            y_reshape_left = 0
        if y_reshape_right>227:
            y_reshape_right = 227
        
        temp_size_i = np.array([x_reshape_left,x_reshape_right,y_reshape_left,y_reshape_right]).T
        reshape_size[:,i] = temp_size_i
    return reshape_size


# In[6]:

# # usage & test correctness
# reshape_size = get_reshape_size(test_label,dimY)
# print(reshape_size.shape)


# In[7]:

def image_crop(image,Y_pred,reshape_size):
    '''
    Input: image is a single image with shape 227*227*3
           Y_pred is of shape 2*14 for lsp dataset
           reshape_size is of shape 4*14 for lsp dataset
           
    Return: crop_image_list is a list of images that being croped. len(crop_image_list) is 14 for lsp dataset
            actual_reshape_size is a matrix of shape 2*14 for lsp dataset indicating the actual size of croped images
    '''
    # a list of crop image.First the size of each image depends on reshape_size, then each image will be rescaled to 227*227*3
    crop_image_list = []
    
    # get the shape of Y, for lsp dataset it is 2*14
    yshape0,yshape1 = Y_pred.shape
    
    # a matrix of actual reshape weight and height. It is of shape 2*14 for lsp dataset
    actual_reshape_size = np.zeros((2,yshape1))
    
    
    for i in range(yshape1):
        # crop image. Notice each image has to receive integer value. So cache the actual crop image size before rescale
        temp_image = image[int(reshape_size[0,i]):int(reshape_size[1,i]),int(reshape_size[2,i]):int(reshape_size[3,i])]
        # get the actual image size
        temp_image_weight, temp_image_height,_ = temp_image.shape
        actual_reshape_size[0,i] = temp_image_weight
        actual_reshape_size[1,i] = temp_image_height
        
        # append to list
        crop_image_list.append(temp_image)
    
    return crop_image_list,actual_reshape_size


# In[8]:

# # usage & test case
# crop_image_list,actual_reshape_size = image_crop(test_image,test_label,reshape_size)


# In[9]:

def scale_label(label, original_height, original_width):
    label[0, :] *= (227 / float(original_width))
    label[1,:] *= (227 / float(original_height))
    return label


# In[10]:

def scale_image(image):
    # scale the image to 227*227
    scaled_image = misc.imresize(image, (227, 227), interp='bicubic')
    return scaled_image


# In[11]:

def images_crop(images,Y_preds,Y_trues):
    '''
    Inputs: images is a list of length N and each element has shape 227*227*3
            Y_preds is a matrix of shape(N,2,14) for lsp dataset
            Y_trues is a list of length N and each element has shape 2*14 for lsp dataset
            
    Return: crop_image_matrix is a matrix of shape(N,14,227,227,3) in lsp dataset
            actual_reshape_size_matrix is a matrix of shape(N,14,2,14) in lsp dataset. It indidcates the original shape of the bounding box
            scaled_preds_label is a matrix of shape(N,14,2,14) in lsp dataset indicating the predicted labels after resclae of the bounding box
            scaled_trues_label is a matrix of shape(N,14,2,14) in lsp dataset indicating the true labels after resclae of the bounding box
    '''
    N,C,length = Y_preds.shape
    
    crop_image_matrix = np.zeros((N,length,227,227,3))
    actual_reshape_size_matrix = np.zeros((N,2,length))
    scaled_preds_label = np.zeros((N,length,2,length))
    scaled_trues_label = np.zeros((N,length,2,length))
    
    # loop through every image and do image crop
    for i in range(N):
        
        pred_label = Y_preds[i]
        true_label = Y_trues[i]
        
        dimY = compute_dimy(pred_label)
        reshape_size = get_reshape_size(pred_label,dimY)
        
        crop_image_list,actual_reshape_size = image_crop(images[i],pred_label,reshape_size)
        
        # store in the actual_reshape_size_matrix
        actual_reshape_size_matrix[i] = actual_reshape_size
        
        # loop through every croped image and rescale to 227*227*3 and also rescale the true and predicted label accordingly
        for j in range(len(crop_image_list)):
            
            single_image = crop_image_list[j]
            
            original_width = actual_reshape_size[0,j]
            original_height = actual_reshape_size[1,j]
            
            scaled_image = scale_image(single_image)
            scaled_label_pred = scale_label(pred_label,original_width,original_height)
            scaled_true_pred = scale_label(true_label,original_width,original_height)
            
            # store everything
            
            crop_image_matrix[i,j] = scaled_image
            scaled_preds_label[i,j] = scaled_label_pred
            scaled_trues_label[i,j] = scaled_true_pred
            
    return crop_image_matrix,actual_reshape_size_matrix,scaled_preds_label,scaled_trues_label


# In[ ]:

# usage:
# crop_image_matrix,actual_reshape_size_matrix,scaled_preds_label,scaled_trues_label = images_crop(test_list,y_preds,test_label)

