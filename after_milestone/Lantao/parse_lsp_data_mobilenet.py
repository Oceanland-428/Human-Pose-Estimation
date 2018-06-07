
# coding: utf-8

# In[10]:


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


# In[11]:


# reshape the image to 227*227*3 since this is the input size for alexnet
def prepare_image(original_image_path):
    image = misc.imread(original_image_path)
    # scale the image to 227*227
    scaled_image = misc.imresize(image, (224, 224), interp='bicubic')
    return scaled_image, image.shape[0], image.shape[1]


# In[12]:


# reshape the label accordingly. Notice label[0] should be rescaled by width
# label[1] should be rescaled by height. I guess this is what we should do (might reverse width and height? if the output is not good)
# refer to: https://github.com/samitok/deeppose/blob/master/Codes/Original/GetLSPData.py scale_label function
def scale_label(label, original_height, original_width):
    label[0, :] *= (224 / float(original_width))
    label[1,:] *= (224 / float(original_height))
    return label


# In[13]:


# each image x has shape 227*227*3, each label y has shape 3*14 
# each image_list contains num_examples images. Each label_list contains num_examples labels
# image_list[i] has label label_list[i]
def generate_dataset(image_paths,labels,dataset):
    num_examples = image_paths.shape[0]
    image_list = []
    label_list = []
    print('Start to process '+ dataset + ' dataset')
    for index in range(num_examples):
        image, or_height, or_width = prepare_image(image_paths[index])
        image_list.append(image)
        label = scale_label(labels[index], or_height, or_width)
        # only extract x and y coordinates since z is 0 for all data
        label_xy = label[0:2, :]
        label_list.append(label_xy)
    print('Done processing the ' + dataset + ' dataset')
    return np.array(image_list), np.array(label_list)


# In[14]:


# get the train,val,test dataset
def getLSPDataset(train_set_ratio=0.8,validation_set_ratio = 0.1):
    print('Resizing and packing images and labels to lists.\n')
    np.random.seed(1701)  # to fix test set
    # load the dataset. Make sure you put the joints.mat in the same folder as this .ipynb or .py program
    # otherwise you can change the path here
    joints = loadmat('/home/oceanland/lsp_dataset/joints.mat')
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
    image_list = np.asarray(sorted(glob.glob('/home/oceanland/lsp_dataset/images/*.jpg')))
    
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



