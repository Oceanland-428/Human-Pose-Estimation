from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import os
from tempfile import TemporaryFile
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#from stage_1_model_v1 import *
import stage_1_util_v1
from parse_flic_data import *
from image_noise_generation import *
from crop_image import *
from drawLines_v2 import drawLines
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import stage_1_modelClass_v1 as s1
import stage_s_modelClass_v1 as ss
tfe.enable_eager_execution()

train_list, train_label, val_list, val_label, test_list, test_label = getFLICData()
train_X = train_list	# (N, 227, 227, 3)
train_Y = train_label	# (N, 2, 14)
print("train X: ", train_X.shape, "val X: ", val_list.shape, "test X: ", test_list.shape)
print("train Y: ", train_Y.shape, "val Y: ", val_label.shape, "test Y: ", test_label.shape)

train_X = train_list
train_Y = train_label

n_samples = train_Y.shape[0] # N
val_n = val_label.shape[0]
n_joints = train_Y.shape[2]
n_stage = 2

# Parameters
learning_rate = 0.00005
batch_size = 128
display_step = 1
thresh = 0.5

if n_samples % batch_size == 0:
    n_batch = int(n_samples / batch_size)
else:
    n_batch = int(n_samples / batch_size) + 1

if val_n % batch_size == 0:
    val_n_batch = int(val_n / batch_size)
else:
    val_n_batch = int(val_n / batch_size) + 1

    
def loss(model, inputs, targets, stage, is_training = True, b_array = None, prev_predictions = None):
    #print(inputs)
    if stage == 1:
        predictions = model.predict(inputs, is_training) # (N, 2, 14) in original scale
    else:
        predictions = model.predict(inputs, is_training, b_array, prev_predictions)
        
    loss = tf.reduce_sum(tf.square(predictions - targets))
    return loss

def get_loss_acc(targets, predictions, torsor_frac):
    joint_diff = tf.square(targets - predictions)
    joint_dist = tf.sqrt(tf.reduce_sum(joint_diff, axis = 1))   # (N, 2, 14) -> (N, 14)
    loss = tf.reduce_sum(joint_diff)
    valid_mask = tf.to_int32(tf.greater(torsor_frac, joint_dist)) * 1.0
    accuracy = tf.reduce_sum(valid_mask) / (int(joint_dist.shape[0]) * int(joint_dist.shape[1]))
    wrist_acc = tf.reduce_sum(valid_mask[:, 2] + valid_mask[:, 5]) / (int(joint_dist.shape[0]) * 2)
    elbow_acc = tf.reduce_sum(valid_mask[:, 1] + valid_mask[:, 4]) / (int(joint_dist.shape[0]) * 2)
    return loss.numpy(), accuracy.numpy(), wrist_acc.numpy(), elbow_acc.numpy()

def loss_accuracy_1(model, inputs, targets, thresh, is_training):
    torsor_xy = targets[:, :, 0] - targets[:, :, 7] #distance left_shoulder <-> right_hip in xy, result (N, 2)
    torsor_dist = tf.sqrt(tf.reduce_sum(tf.square(torsor_xy), axis = 1, keep_dims = True)) # distance scaler, (N, 2) -> (N)
    torsor_frac = torsor_dist * thresh # max error distance

    predictions = model.predict(inputs, is_training) # (N, 2, 14) in original scale
    loss, accuracy, wrist_acc, elbow_acc = get_loss_acc(targets, predictions, torsor_frac)
    
    return loss, accuracy, wrist_acc, elbow_acc

def loss_accuracy_s(model, inputs, targets, thresh, prev_pred, b_array, is_training):
    torsor_xy = targets[:, :, 0] - targets[:, :, 7] #distance left_shoulder <-> right_hip in xy, result (N, 2)
    torsor_dist = tf.sqrt(tf.reduce_sum(tf.square(torsor_xy), axis = 1, keep_dims = True)) # distance scaler, (N, 2) -> (N)
    torsor_frac = torsor_dist * thresh # max error distance

    inputs = tf.convert_to_tensor(inputs, np.float32)
    predictions = model.predict(inputs, is_training, b_array, prev_pred)
    loss, accuracy, wrist_acc, elbow_acc = get_loss_acc(targets, predictions, torsor_frac)
    
    return loss, accuracy, wrist_acc, elbow_acc

def next_batch_XY(batch, batch_size, X, Y):
    if batch < n_batch - 1:
        batch_X = X[batch * batch_size: (batch + 1) * batch_size, :, :, :]
        batch_Y = Y[batch * batch_size: (batch + 1) * batch_size, :, :]
    else:
        batch_X = X[batch * batch_size:, :, :, :]
        batch_Y = Y[batch * batch_size:, :, :]
    batch_X = tf.convert_to_tensor(batch_X, np.float32)
    batch_Y = tf.convert_to_tensor(batch_Y, np.float32)
    #print(batch_X.shape, batch_Y.shape)
    return batch_X, batch_Y

def next_batch_bp(batch, batch_size, X, Y):
    if batch < n_batch - 1:
        batch_X = X[batch * batch_size: (batch + 1) * batch_size, :, :]
        batch_Y = Y[batch * batch_size: (batch + 1) * batch_size, :, :]
    else:
        batch_X = X[batch * batch_size:, :, :]
        batch_Y = Y[batch * batch_size:, :, :]
    batch_X = tf.convert_to_tensor(batch_X, np.float32)
    batch_Y = tf.convert_to_tensor(batch_Y, np.float32)
    #print(batch_X.shape, batch_Y.shape)
    return batch_X, batch_Y

def shuffle(X, Y):
    image_indexes = list(range(Y.shape[0]))
    np.random.shuffle(image_indexes)
    train_X_new = X[np.asarray(image_indexes)]
    train_Y_new = Y[np.asarray(image_indexes)]
    #print(image_indexes)
    #print(train_X_new[:, 0, :, :])
    #print(train_Y_new[:, 0, :])
    return train_X_new, train_Y_new

model1 = s1.stage_1_model(n_joints)
model2 = ss.stage_s_model(n_joints)

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

# Compute gradients
grad = tfe.implicit_gradients(loss)

# Training
# is_training = False for no dropout
tr_costs_1 = []
tr_costs_2 = []
val_cost_1 = []
val_cost_2 = []
train_acc_list_1 = []
train_acc_list_2 = []
val_acc_list_1 = []
val_acc_list_2 = []
wrist_acc_list_1 = []
wrist_acc_list_2 = []
elbow_acc_list_1 = []
elbow_acc_list_2 = []
loss_incr_max = 0.0
params = stage_1_util_v1.params # all stages use the same params
train_X_s, train_Y_s = train_X, train_Y

tr_X = train_list.astype(np.float32)
tr_Y = train_label.astype(np.float32)
val_X = val_list.astype(np.float32)
val_Y = val_label.astype(np.float32)
for epoch in range(100):
    with tf.device('/device:GPU:0'):
        batch_cost_1 = 0.0

        train_X_s, train_Y_s = shuffle(train_X_s, train_Y_s)

        for batch in range(n_batch):
            batch_X, batch_Y = next_batch_XY(batch, batch_size, train_X_s, train_Y_s)
            optimizer.apply_gradients(grad(model1, batch_X, batch_Y, 1))
            batch_cost_1 += loss(model1, batch_X, batch_Y, 1)

        batch_cost_1 = batch_cost_1 / n_samples
        tr_costs_1.append(batch_cost_1)
        if epoch % display_step == 0:
            print("Stage: 1, Epoch:", epoch, "cost", batch_cost_1.numpy())
            
        tr_acc_1 = 0.0
        for batch in range(n_batch):
            batch_X, batch_Y = next_batch_XY(batch, batch_size, tr_X, tr_Y)
            _, cur_tr_acc_1, _, _ = loss_accuracy_1(model1, batch_X, batch_Y, thresh, False)
            tr_acc_1 += cur_tr_acc_1 * int(batch_X.shape[0])
        tr_acc_1 /= int(tr_X.shape[0])
        print("train accuracy 1: ", tr_acc_1)
        train_acc_list_1.append(tr_acc_1)
        val_loss_1, val_acc_1, wrist_acc_1, elbow_acc_1 = loss_accuracy_1(model1, val_X, val_Y, thresh, False)
        val_acc_list_1.append(val_acc_1)
        wrist_acc_list_1.append(wrist_acc_1)
        elbow_acc_list_1.append(elbow_acc_1)
        val_loss_1 /= int(val_Y.shape[0])
        val_cost_1.append(val_loss_1)
        print("val loss 1: ", val_loss_1, "val accuracy 1: ", val_acc_1)
        print("wrist acc 1: ", wrist_acc_1, "elbow acc 1: ", elbow_acc_1)
        
with tf.device('/device:GPU:0'):    
    val_predictions_1 = model1.predict(val_X, is_training = False)
    val_inputs_2, val_b_array_1 = images_crop(val_X, val_predictions_1, val_Y)

for epoch in range(30):
    with tf.device('/device:GPU:0'):
        batch_cost_2 = 0.0

        for batch in range(n_batch):
            print("epoch: ", epoch, "batch: ", batch)
            batch_X_1, batch_Y_1 = next_batch_XY(batch, batch_size, tr_X, tr_Y)
            predictions_1 = model1.predict(batch_X_1, is_training = False)
            inputs_2, b_array_1 = images_crop(batch_X_1, predictions_1, batch_Y_1)
            inputs_2 = tf.convert_to_tensor(inputs_2, np.float32)

            optimizer.apply_gradients(grad(model2, inputs_2, batch_Y_1, 2, True, b_array_1, predictions_1))
            batch_cost_2 += loss(model2, inputs_2, batch_Y_1, 2, True, b_array_1, predictions_1)
        batch_cost_2 = batch_cost_2 / n_samples
        tr_costs_2.append(batch_cost_2)
        if epoch % display_step == 0:
            print("Stage: 2, Epoch:", epoch, "cost", batch_cost_2.numpy()) 
        tr_acc_2 = 0.0
        for batch in range(n_batch):
            print("train, ", batch)
            batch_X_1, batch_Y_1 = next_batch_XY(batch, batch_size, tr_X, tr_Y)
            predictions_1 = model1.predict(batch_X_1, is_training = False)
            inputs_2, b_array_1 = images_crop(batch_X_1, predictions_1, batch_Y_1)
            _, cur_tr_acc_2, _, _ = loss_accuracy_s(model2, inputs_2, batch_Y_1, thresh, predictions_1, b_array_1, False)
            tr_acc_2 += cur_tr_acc_2 * int(batch_X_1.shape[0])
        tr_acc_2 /= int(tr_X.shape[0])
        print("train accuracy 2: ", tr_acc_2)
        train_acc_list_2.append(tr_acc_2)
        
        val_acc_2 = 0.0
        val_loss_2 = 0.0
        wrist_acc_2 = 0.0
        elbow_acc_2 = 0.0
        for batch in range(val_n_batch):
            print("val, ", batch)
            val_batch_X, val_batch_Y = next_batch_XY(batch, batch_size, val_inputs_2, val_Y)
            val_batch_b_array_1, val_batch_predictions_1 = next_batch_bp(batch, batch_size, val_b_array_1, val_predictions_1)
            print(val_batch_X.shape)
            cur_val_loss_2, cur_val_acc_2, cur_wrist_acc_2, cur_elbow_acc_2 = loss_accuracy_s(model2, val_batch_X, val_batch_Y, thresh, val_batch_predictions_1, val_batch_b_array_1, False)
            val_acc_2 += cur_val_acc_2 * int(val_batch_X.shape[0])
            wrist_acc_2 += cur_wrist_acc_2 * int(val_batch_X.shape[0])
            elbow_acc_2 += cur_elbow_acc_2 * int(val_batch_X.shape[0])
            val_loss_2 += cur_val_loss_2
        val_acc_2 /= int(val_Y.shape[0])
        wrist_acc_2 /= int(val_Y.shape[0])
        elbow_acc_2 /= int(val_Y.shape[0])
        val_acc_list_2.append(val_acc_2)
        wrist_acc_list_2.append(wrist_acc_2)
        elbow_acc_list_2.append(elbow_acc_2)
        val_loss_2 /= int(val_Y.shape[0])
        val_cost_2.append(val_loss_2)
        print("val loss 2: ", val_loss_2, "val accuracy 2: ", val_acc_2)
        print("wrist acc 2: ", wrist_acc_2, "elbow acc 2: ", elbow_acc_2)
        
        
with open('tr_costs_1.txt', 'w') as filehandle:  
    for listitem in np.array(tr_costs_1):
        filehandle.write('%s\n' % listitem)
        
with open('tr_costs_2.txt', 'w') as filehandle:  
    for listitem in np.array(tr_costs_2):
        filehandle.write('%s\n' % listitem)
        
with open('val_cost_1.txt', 'w') as filehandle:  
    for listitem in np.array(val_cost_1):
        filehandle.write('%s\n' % listitem)
        
with open('val_cost_2.txt', 'w') as filehandle:  
    for listitem in np.array(val_cost_2):
        filehandle.write('%s\n' % listitem)
        
with open('train_acc_list_1.txt', 'w') as filehandle:  
    for listitem in np.array(train_acc_list_1):
        filehandle.write('%s\n' % listitem)
        
with open('train_acc_list_2.txt', 'w') as filehandle:  
    for listitem in np.array(train_acc_list_2):
        filehandle.write('%s\n' % listitem)
        
with open('val_acc_list_1.txt', 'w') as filehandle:  
    for listitem in np.array(val_acc_list_1):
        filehandle.write('%s\n' % listitem)
        
with open('val_acc_list_2.txt', 'w') as filehandle:  
    for listitem in np.array(val_acc_list_2):
        filehandle.write('%s\n' % listitem)
        
with open('wrist_acc_list_1.txt', 'w') as filehandle:  
    for listitem in np.array(wrist_acc_list_1):
        filehandle.write('%s\n' % listitem)
    
with open('elbow_acc_list_1.txt', 'w') as filehandle:  
    for listitem in np.array(elbow_acc_list_1):
        filehandle.write('%s\n' % listitem)
        
with open('wrist_acc_list_2.txt', 'w') as filehandle:  
    for listitem in np.array(wrist_acc_list_2):
        filehandle.write('%s\n' % listitem)
    
with open('elbow_acc_list_2.txt', 'w') as filehandle:  
    for listitem in np.array(elbow_acc_list_2):
        filehandle.write('%s\n' % listitem)
    

with tf.device('/device:GPU:0'):
    for image in range(int(train_list.shape[0])):
        one = train_list[image, :, :, :]
        one_1 = tf.convert_to_tensor(one, np.float32)
        one_1 = tf.reshape(one_1, [1, one_1.shape[0], one_1.shape[1], one_1.shape[2]])
        predictions_1 = model1.predict(one_1, False) # (N, 2, 14) in original scale
        inputs_2, b_array_1 = images_crop(one_1, predictions_1, train_label[0, :, :])    # inputs_2 (N, 227, 227, 3*14), b_array_1 (N, 14)
        inputs_2 = tf.convert_to_tensor(inputs_2, np.float32)
        predictions_2 = model2.predict(inputs_2, False, b_array_1, predictions_1)
        data = np.array(predictions_2)
        with open('train_pred_vF_1.txt', 'a') as outfile:
            outfile.write('# Array shape: {0}\n'.format(data.shape))
            for data_slice in data:
                np.savetxt(outfile, data_slice)
                outfile.write('# New slice\n')

with tf.device('/device:GPU:0'):
    for image in range(int(val_list.shape[0])):
        one = val_list[image, :, :, :]
        one_1 = tf.convert_to_tensor(one, np.float32)
        one_1 = tf.reshape(one_1, [1, one_1.shape[0], one_1.shape[1], one_1.shape[2]])
        predictions_1 = model1.predict(one_1, False) # (N, 2, 14) in original scale
        inputs_2, b_array_1 = images_crop(one_1, predictions_1, val_label[0, :, :])    # inputs_2 (N, 227, 227, 3*14), b_array_1 (N, 14)
        inputs_2 = tf.convert_to_tensor(inputs_2, np.float32)
        predictions_2 = model2.predict(inputs_2, False, b_array_1, predictions_1)
        data = np.array(predictions_2)
        with open('val_pred_vF_1.txt', 'a') as outfile:
            outfile.write('# Array shape: {0}\n'.format(data.shape))
            for data_slice in data:
                np.savetxt(outfile, data_slice)
                outfile.write('# New slice\n')