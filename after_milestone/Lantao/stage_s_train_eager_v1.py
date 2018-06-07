from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#from stage_1_model_v1 import *
import stage_1_util_v1
from parse_lsp_extend_data import *
from crop_image import *
from drawLines_v2 import drawLines
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import stage_1_modelClass_v1 as s1
import stage_s_modelClass_v1 as ss
tfe.enable_eager_execution()

train_list, train_label, val_list, val_label, test_list, test_label = getLSPExtendDataset(0.9, 0.05)
train_X = train_list	# (N, 227, 227, 3)
train_Y = train_label	# (N, 2, 14)
print("train X: ", train_list.shape, "val X: ", val_list.shape, "test X: ", test_list.shape)
print("train Y: ", train_label.shape, "val Y: ", val_label.shape, "test Y: ", test_label.shape)

train_X = train_list
train_Y = train_label

n_samples = train_Y.shape[0] # N
val_n = val_label.shape[0]
n_joints = train_Y.shape[2]
n_stage = 3

# Parameters
learning_rate = 0.00005
batch_size = 128
display_step = 1
epochs = 250
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
    #print(predictions_1)
    #inputs_2, b_array_1 = images_crop(inputs, predictions_1, targets)    # inputs_2 (N, 227, 227, 3*14), b_array_1 (N, 14)
    #predictions_2 = model2.predict(inputs_2, is_training, b_array_1, predictions_1)
    #inputs_3, b_array_2, _, _ = images_crop(inputs, predictions_2, targets)
    #predictions_3 = model3.predict(inputs_3, is_training, b_array_2, predictions_2)
    loss = tf.reduce_sum(tf.square(predictions - targets))
    return loss

def get_loss_acc(targets, predictions, torsor_frac):
    joint_diff = tf.square(targets - predictions)
    joint_dist = tf.sqrt(tf.reduce_sum(joint_diff, axis = 1))   # (N, 2, 14) -> (N, 14)
    loss = tf.reduce_sum(joint_diff)
    accuracy = tf.reduce_sum(tf.to_int32(tf.greater(torsor_frac, joint_dist))) * 1.0 / (int(joint_dist.shape[0]) * int(joint_dist.shape[1]))
    return loss.numpy(), accuracy.numpy()

def loss_accuracy_1(model, inputs, targets, thresh, is_training):
    torsor_xy = targets[:, :, 9] - targets[:, :, 2] #distance left_shoulder <-> right_hip in xy, result (N, 2)
    torsor_dist = tf.sqrt(tf.reduce_sum(tf.square(torsor_xy), axis = 1, keep_dims = True)) # distance scaler, (N, 2) -> (N)
    torsor_frac = torsor_dist * thresh # max error distance

    predictions = model.predict(inputs, is_training) # (N, 2, 14) in original scale
    loss, accuracy = get_loss_acc(targets, predictions, torsor_frac)
    
    return loss, accuracy

def loss_accuracy_s(model, inputs, targets, thresh, prev_pred, b_array, is_training):
    torsor_xy = targets[:, :, 9] - targets[:, :, 2] #distance left_shoulder <-> right_hip in xy, result (N, 2)
    torsor_dist = tf.sqrt(tf.reduce_sum(tf.square(torsor_xy), axis = 1, keep_dims = True)) # distance scaler, (N, 2) -> (N)
    torsor_frac = torsor_dist * thresh # max error distance

    inputs = tf.convert_to_tensor(inputs, np.float32)
    predictions = model.predict(inputs, is_training, b_array, prev_pred)
    loss, accuracy = get_loss_acc(targets, predictions, torsor_frac)

    #inputs_3, b_array_2, _, _ = images_crop(inputs, predictions_2, targets)
    #predictions_3 = model3.predict(inputs_3, is_training, b_array_2, predictions_2)
    #loss_3, accuracy_3 = get_loss_acc(targets, predictions_3, torsor_frac)
    #return loss_1, loss_2, loss_3, accuracy_1, accuracy_2, accuracy_3
    return loss, accuracy

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
model3 = ss.stage_s_model(n_joints)

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
loss_incr_max = 0.0
params = stage_1_util_v1.params
train_X_s, train_Y_s = train_X, train_Y

tr_X = train_list.astype(np.float32)
tr_Y = train_label.astype(np.float32)
val_X = val_list.astype(np.float32)
val_Y = val_label.astype(np.float32)
for epoch in range(5):
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
            _, cur_tr_acc_1 = loss_accuracy_1(model1, batch_X, batch_Y, thresh, False)
            tr_acc_1 += cur_tr_acc_1 * int(batch_X.shape[0])
        tr_acc_1 /= int(tr_X.shape[0])
        print("train accuracy 1: ", tr_acc_1)
        train_acc_list_1.append(tr_acc_1)
        val_loss_1, val_acc_1 = loss_accuracy_1(model1, val_X, val_Y, thresh, False)
        val_acc_list_1.append(val_acc_1)
        val_loss_1 /= int(val_Y.shape[0])
        val_cost_1.append(val_loss_1)
        print("val loss 1: ", val_loss_1, "val accuracy 1: ", val_acc_1)
with tf.device('/device:GPU:0'):    
    val_predictions_1 = model1.predict(val_X, is_training = False)
    val_inputs_2, val_b_array_1 = images_crop(val_X, val_predictions_1, val_Y)

for epoch in range(3):
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
            _, cur_tr_acc_2 = loss_accuracy_s(model2, inputs_2, batch_Y_1, thresh, predictions_1, b_array_1, False)
            tr_acc_2 += cur_tr_acc_2 * int(batch_X_1.shape[0])
        tr_acc_2 /= int(tr_X.shape[0])
        print("train accuracy 2: ", tr_acc_2)
        train_acc_list_2.append(tr_acc_2)
        
        val_acc_2 = 0.0
        val_loss_2 = 0.0
        for batch in range(val_n_batch):
            print("val, ", batch)
            val_batch_X, val_batch_Y = next_batch_XY(batch, batch_size, val_inputs_2, val_Y)
            val_batch_b_array_1, val_batch_predictions_1 = next_batch_bp(batch, batch_size, val_b_array_1, val_predictions_1)
            print(val_batch_X.shape)
            cur_val_loss_2, cur_val_acc_2 = loss_accuracy_s(model2, val_batch_X, val_batch_Y, thresh, val_batch_predictions_1, val_batch_b_array_1, False)
            val_acc_2 += cur_val_acc_2 * int(val_batch_X.shape[0])
            val_loss_2 += cur_val_loss_2
        val_acc_2 /= int(val_Y.shape[0])
        val_acc_list_2.append(val_acc_2)
        val_loss_2 /= int(val_Y.shape[0])
        val_cost_2.append(val_loss_2)
        print("val loss 2: ", val_loss_2, "val accuracy 2: ", val_acc_2)

        
fig = plt.figure()
plt.plot(range(len(tr_costs_1)), np.squeeze(tr_costs_1), label = "train")
plt.ylabel('cost')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("stage 1, Learning rate: " + str(learning_rate))
fig.savefig('AlexNet_cost_1.png')
plt.show()

fig = plt.figure()
plt.plot(range(len(tr_costs_2)), np.squeeze(tr_costs_2), label = "train")
plt.ylabel('cost')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("stage 2, Learning rate: " + str(learning_rate))
fig.savefig('AlexNet_cost_2.png')
plt.show()

tr_costs_whole = tr_costs_1 + tr_costs_2
fig = plt.figure()
plt.plot(range(len(tr_costs_whole)), np.squeeze(tr_costs_whole), label = "train")
plt.ylabel('cost')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("whole, Learning rate: " + str(learning_rate))
fig.savefig('AlexNet_cost_12.png')
plt.show()


fig = plt.figure()
plt.plot(range(len(val_acc_list_1)), np.squeeze(train_acc_list_1), label = "train")
plt.plot(range(len(val_acc_list_1)), np.squeeze(val_acc_list_1), label = "val")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("stage 1, Learning rate: " + str(learning_rate) + ", thresh: " + str(thresh))
fig.savefig('AlexNet_accuracy_1.png')
plt.show()

fig = plt.figure()
plt.plot(range(len(val_acc_list_2)), np.squeeze(train_acc_list_2), label = "train")
plt.plot(range(len(val_acc_list_2)), np.squeeze(val_acc_list_2), label = "val")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("stage 2, Learning rate: " + str(learning_rate) + ", thresh: " + str(thresh))
fig.savefig('AlexNet_accuracy_2.png')
plt.show()

tr_acc_whole = train_acc_list_1 + train_acc_list_2
val_acc_whole = val_acc_list_1 + val_acc_list_2
fig = plt.figure()
plt.plot(range(len(tr_acc_whole)), np.squeeze(tr_acc_whole), label = "train")
plt.plot(range(len(tr_acc_whole)), np.squeeze(val_acc_whole), label = "val")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("whole, Learning rate: " + str(learning_rate) + ", thresh: " + str(thresh))
fig.savefig('AlexNet_accuracy_12.png')
plt.show()


fig = plt.figure()
plt.plot(range(len(val_cost_1)), np.squeeze(val_cost_1), label = "train")
plt.ylabel('cost')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("stage 1, Learning rate: " + str(learning_rate))
fig.savefig('AlexNet_cost_1_val.png')
plt.show()

fig = plt.figure()
plt.plot(range(len(val_cost_2)), np.squeeze(val_cost_2), label = "train")
plt.ylabel('cost')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("stage 2, Learning rate: " + str(learning_rate))
fig.savefig('AlexNet_cost_2_val.png')
plt.show()

val_costs_whole = val_cost_1 + val_cost_2
fig = plt.figure()
plt.plot(range(len(val_costs_whole)), np.squeeze(val_costs_whole), label = "train")
plt.ylabel('cost')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("whole, Learning rate: " + str(learning_rate))
fig.savefig('AlexNet_cost_12_val.png')
plt.show()


with tf.device('/device:GPU:0'):
    one = train_list[0, :, :, :]
    one_1 = tf.convert_to_tensor(one, np.float32)
    one_1 = tf.reshape(one_1, [1, one_1.shape[0], one_1.shape[1], one_1.shape[2]])
    predictions_1 = model1.predict(one_1, False) # (N, 2, 14) in original scale
    inputs_2, b_array_1 = images_crop(one_1, predictions_1, train_label[0, :, :])    # inputs_2 (N, 227, 227, 3*14), b_array_1 (N, 14)
    inputs_2 = tf.convert_to_tensor(inputs_2, np.float32)
    predictions_2 = model2.predict(inputs_2, False, b_array_1, predictions_1)
    print(predictions_2)
    print(train_label[0, :, :])
    
drawLines(train_list[0].copy(),train_label[0].copy(),predictions_2[0])

with tf.device('/device:GPU:0'):
    one = val_list[0, :, :, :]
    one_1 = tf.convert_to_tensor(one, np.float32)
    one_1 = tf.reshape(one_1, [1, one_1.shape[0], one_1.shape[1], one_1.shape[2]])
    predictions_1 = model1.predict(one_1, False) # (N, 2, 14) in original scale
    inputs_2, b_array_1 = images_crop(one_1, predictions_1, val_label[0, :, :])    # inputs_2 (N, 227, 227, 3*14), b_array_1 (N, 14)
    inputs_2 = tf.convert_to_tensor(inputs_2, np.float32)
    predictions_2 = model2.predict(inputs_2, False, b_array_1, predictions_1)
    print(predictions_2)
    print(val_label[0, :, :])
    
drawLines(val_list[0].copy(),val_label[0].copy(),predictions_2[0])