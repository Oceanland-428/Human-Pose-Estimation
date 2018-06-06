
# coding: utf-8

# In[152]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import glob
import numpy as np
import re
import matplotlib.image as img
from scipy import misc
import matplotlib.pyplot as plt
import skimage.transform
from os.path import basename as b
from scipy.io import loadmat
import random


# In[153]:

def prepare_image(original_image_path):
    image = misc.imread(original_image_path)
    # scale the image to 227*227
    scaled_image = misc.imresize(image, (227, 227), interp='bicubic')
    return scaled_image, image.shape[0], image.shape[1]


# In[154]:

def scale_label(label, original_height, original_width):
    label[0, :] *= (227 / float(original_width))
    label[1,:] *= (227 / float(original_height))
    return label


# In[157]:

def getFLICData(train_set_ratio=0.8,validation_set_ratio = 0.1):
    # load in examples.mat
    examples = loadmat('examples.mat')
    examples = examples['examples'][0]
    
    # just for reference, not used in the following code
    joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip',
                 'lkne', 'lank', 'rhip', 'rkne', 'rank', 'leye', 'reye',
                 'lear', 'rear', 'nose', 'msho', 'mhip', 'mear', 'mtorso',
                 'mluarm', 'mruarm', 'mllarm', 'mrlarm', 'mluleg', 'mruleg',
                 'mllleg', 'mrlleg']
    available_index = [0,1,2,3,4,5,6,9,12,13,16]
    
    image_list = []
    label_list = []
    
    for i, example in enumerate(examples):
        joint_matrix = example[2]
        joint_matrix = joint_matrix[:,available_index]
        # get the label which is the joint_matrix
        label = joint_matrix

        # extract the image information
        image_name = str(example[3][0])
        image_path = './images/'+image_name
        image, or_height, or_width = prepare_image(image_path)

        label = scale_label(label,or_height,or_width)

        image_list.append(image)
        label_list.append(label)
    
    # shuffle the data
    c = list(zip(image_list, label_list))
    random.shuffle(c)
    image_list, label_list = zip(*c)
    image_list = list(image_list)
    label_list = list(label_list)
    
    train_set_ratio=0.8
    validation_set_ratio = 0.1
    # get the training, val and test set indexes
    train_validation_split = int(len(indexs)*train_set_ratio)
    validation_test_split = int(len(indexs)*(train_set_ratio+validation_set_ratio))
    train_list = image_list[:train_validation_split]
    train_label = label_list[:train_validation_split]
    val_list = image_list[train_validation_split:validation_test_split]
    val_label = label_list[train_validation_split:validation_test_split]
    test_list = image_list[validation_test_split:]
    test_label = label_list[validation_test_split:]
    
    return np.array(train_list),np.array(train_label),np.array(val_list),np.array(val_label),np.array(test_list),np.array(test_label)


# In[158]:

#train_list,train_label,val_list,val_label,test_list,test_label = getFLICData()


# In[ ]:



