
# coding: utf-8

# In[1]:


import numpy as np
from parse_lsp_data import *
import tensorflow as tf


# In[2]:


from keras.preprocessing import image
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout,Reshape
from keras.optimizers import SGD,Adam
import keras.backend as K


# In[3]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[4]:


# get the data set
train_list, train_label, val_list, val_label, test_list, test_label = getLSPDataset(0.8, 0.1)


# In[5]:


# print some information
train_X = train_list	# (N, 227, 227, 3)
train_Y = train_label	# (N, 2, 14)
print("train X: ", train_list.shape, "val X: ", val_list.shape, "test X: ", test_list.shape)
print("train Y: ", train_label.shape, "val Y: ", val_label.shape, "test Y: ", test_label.shape)


# In[6]:


# build keras model
def build_model():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(227,227, 3))
    last = base_model.output
    x = Flatten()(last)
    x = Dense(4096, activation='elu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(4096, activation='elu')(x)
    x = Dense(28, activation='elu')(x)
    preds = Reshape((2,14))(x)
    model = Model(base_model.input, preds)
    return model


# In[7]:


model = build_model()


# In[8]:


#model.summary()


# In[9]:


# custom loss function (L2 distance)
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# In[10]:


def loss_accuracy(y_true, y_pred):
    diff = tf.square(y_true - y_pred)
    loss = tf.reduce_sum(diff)
    dist = tf.sqrt(tf.reduce_sum(diff, axis = 1))	# (N, 2, 14) -> (N, 14)
    accuracy = tf.reduce_sum(tf.to_int32(tf.greater(1.0, dist))) / 128/14
    return accuracy


# In[11]:


# compile the model
model.compile(optimizer=Adam(lr=0.001), loss=euclidean_distance_loss, metrics=[loss_accuracy])


# In[13]:


csv_logger = CSVLogger('logvgg_v2.csv', append=True, separator=';')


# In[14]:


with tf.device('/device:GPU:0'):
    # fit the model
    history = model.fit(train_X, train_Y, epochs=1000,batch_size=128, validation_data = (val_list,val_label),callbacks=[csv_logger])

model.save('vgg_v2.h5')


