
# coding: utf-8

# In[95]:

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


# In[96]:

def prepare_image(original_image_path):
    image = misc.imread(original_image_path)
    # scale the image to 227*227
    scaled_image = misc.imresize(image, (227, 227), interp='bicubic')
    return scaled_image, image.shape[0], image.shape[1]


# In[97]:

def scale_label(label, original_height, original_width):
    label[:, :] *= (227 / float(original_width))
    return label


# In[98]:

def generate_dataset(image_paths,labels,dataset):
    num_examples = image_paths.shape[0]
    image_list = []
    label_list = []
    print('Start to process '+ dataset + ' dataset')
    for index in range(num_examples):
        image, or_height, or_width = prepare_image(image_paths[index])
        image_list.append(image)
        label = scale_label(labels[index], or_height, or_width)
        label_list.append(label)
    print('Done processing the ' + dataset + ' dataset')
    return image_list,label_list


# In[99]:

def getLSPDataset(train_set_ratio=0.8,validation_set_ratio = 0.1):
    print('Resizing and packing images and labels to lists.\n')
    np.random.seed(1701)  # to fix test set
    # load the dataset. Make sure you put the joints.mat in the same folder as this .ipynb or .py program
    # otherwise you can change the path here
    joints = loadmat('joints.mat')
    # transpose the shape to N*C*number of features, in this case it is 2000*3*14
    joints = joints['joints'].transpose(2, 0, 1)
    
    # I saw some code such as this one: https://github.com/samitok/deeppose/blob/master/Codes/Original/GetLSPData.py only extracts two joints
    # which is Right ankle and Right knee
    # this one as well: https://github.com/mitmul/deeppose/blob/master/datasets/lsp_dataset.py
    # invisible_joints = joints[:, :, 2] < 0.5
    # joints[invisible_joints] = 0
    # joints = joints[..., :2]
    
    # get the list of images names. Make sure you put the images directory in the same folder as this .ipynb or .py program
    # otherwise you can change the path here
    image_list = np.asarray(sorted(glob.glob('./images/*.jpg')))
    
    # get image indexes
    image_indexes = list(range(0, len(image_list)))
    
    # random shuffle the data
    # shuffle the index and use the indexes to select images. So it is equivalent to shuffle images
    np.random.shuffle(image_indexes)
   
    # get the training, val and test set indexes
    train_validation_split = int(len(image_list)*train_set_ratio)
    validation_test_split = int(len(image_list)*(train_set_ratio+validation_set_ratio))
    train_indexes = np.asarray(image_indexes[:train_validation_split])
    validation_indexes = np.asarray(image_indexes[train_validation_split:validation_test_split])
    test_indexes = np.asarray(image_indexes[validation_test_split:])

    # generate label
    train_list,train_label = generate_dataset(image_list[train_indexes],joints[train_indexes],'training')
    val_list,val_label = generate_dataset(image_list[validation_indexes],joints[validation_indexes],'validation')
    test_list,test_label = generate_dataset(image_list[test_indexes],joints[test_indexes],'test')
    
    return train_list,train_label,val_list,val_label,test_list,test_label


# In[110]:

#train_list,train_label,val_list,val_label,test_list,test_label = getLSPDataset()