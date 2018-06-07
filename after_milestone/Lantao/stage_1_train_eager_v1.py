from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#from stage_1_model_v1 import *
import stage_1_util_v1
from parse_lsp_data import *
from drawLines_v2 import drawLines
from stage_1_modelClass_v1 import *
tfe.enable_eager_execution()

train_list, train_label, val_list, val_label, test_list, test_label = getLSPDataset(0.8, 0.1)
train_X = train_list	# (N, 227, 227, 3)
train_Y = train_label	# (N, 2, 14)
print("train X: ", train_list.shape, "val X: ", val_list.shape, "test X: ", test_list.shape)
print("train Y: ", train_label.shape, "val Y: ", val_label.shape, "test Y: ", test_label.shape)

train_X = train_list
train_Y = train_label

n_samples = train_Y.shape[0] # N
n_joints = train_Y.shape[2]

# Parameters
learning_rate = 0.00005
batch_size = 128
display_step = 1
epochs = 250
if n_samples % batch_size == 0:
    n_batch = int(n_samples / batch_size)
else:
    n_batch = int(n_samples / batch_size) + 1
thresh = 0.5

'''
# don't use this', hinge
def loss(model, inputs, targets, is_training = True):
    predictions = model.predict(inputs, is_training)
    diff = tf.square(targets - predictions)
    #loss = tf.reduce_sum(diff)
    dist = tf.sqrt(tf.reduce_sum(diff, axis = 1))	# (N, 2, 14) -> (N, 14)
    loss = tf.reduce_sum(tf.maximum(dist - thresh, 0))
    #accuracy = tf.reduce_sum(tf.to_int32(tf.greater(thresh, dist))) * 1.0 / (int(dist.shape[0]) * int(dist.shape[1]))
    #loss = -(accuracy * accuracy)
    return loss
'''

# use this, L2
def loss(model, inputs, targets, is_training = True):
    predictions = model.predict(inputs, is_training)
    loss = tf.reduce_sum(tf.square(predictions - targets))
    return loss

'''
# don't use this. This is pixel accuracy
def loss_accuracy(model, inputs, targets, thresh, is_training):
    predictions = model.predict(inputs, is_training)
    diff = tf.square(targets - predictions)
    loss = tf.reduce_sum(diff)
    dist = tf.sqrt(tf.reduce_sum(diff, axis = 1))	# (N, 2, 14) -> (N, 14)

    accuracy = tf.reduce_sum(tf.to_int32(tf.greater(thresh, dist))) * 1.0 / (int(dist.shape[0]) * int(dist.shape[1]))
    return loss.numpy(), accuracy.numpy()
'''

# use this. This is PDJ accuracy
def loss_accuracy(model, inputs, targets, thresh, is_training):
    predictions = model.predict(inputs, is_training)
    joint_diff = tf.square(targets - predictions)
    joint_dist = tf.sqrt(tf.reduce_sum(joint_diff, axis = 1))	# (N, 2, 14) -> (N, 14)
    loss = tf.reduce_sum(joint_diff)
    torsor_xy = targets[:, :, 9] - targets[:, :, 2] #distance left_shoulder <-> right_hip in xy, result (N, 2)
    torsor_dist = tf.sqrt(tf.reduce_sum(tf.square(torsor_xy), axis = 1, keep_dims = True)) # distance scaler, (N, 2) -> (N)
    torsor_frac = torsor_dist * thresh # max error distance

    accuracy = tf.reduce_sum(tf.to_int32(tf.greater(torsor_frac, joint_dist))) * 1.0 / (int(joint_dist.shape[0]) * int(joint_dist.shape[1]))
    return loss.numpy(), accuracy.numpy()

def next_batch(batch, batch_size, X, Y):
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

def shuffle(X, Y):
    image_indexes = list(range(Y.shape[0]))
    np.random.shuffle(image_indexes)
    train_X_new = X[np.asarray(image_indexes)]
    train_Y_new = Y[np.asarray(image_indexes)]
    #print(image_indexes)
    #print(train_X_new[:, 0, :, :])
    #print(train_Y_new[:, 0, :])
    return train_X_new, train_Y_new

model = stage_1_model(n_joints)

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

# Compute gradients
grad = tfe.implicit_gradients(loss)

# Training
# is_training = False for no dropout
costs = []
train_acc_list = []
val_acc_list = []
loss_incr_max = 0.0
params = stage_1_util_v1.params
train_X_s, train_Y_s = train_X, train_Y

for epoch in range(epochs):
    with tf.device('/device:GPU:0'):
        batch_cost = 0.0
        
        train_X_s, train_Y_s = shuffle(train_X_s, train_Y_s)
        
        for batch in range(n_batch):
            #print("epoch: ", epoch, "batch: ", batch)
            batch_X, batch_Y = next_batch(batch, batch_size, train_X_s, train_Y_s)
            if batch == 0 and epoch == 0:
                ini_loss = loss(model, batch_X, batch_Y, is_training = True).numpy() / int(batch_Y.shape[0])
                print('initial loss: ', ini_loss)
            #pred = stage_1_model(batch_X, n_joints, is_training, params) # (N, 2, # joints)
            optimizer.apply_gradients(grad(model, batch_X, batch_Y))
            batch_cost += loss(model, batch_X, batch_Y, is_training = True)

        batch_cost = batch_cost / n_samples
        costs.append(batch_cost)
        if epoch > 2:
            incre = ((costs[-1] - costs[-2]) / costs[-2]).numpy()
            print('loss increase percent: ', incre)
            if incre > loss_incr_max:
                loss_incr_max = incre

        if epoch % display_step == 0:
            print("Epoch:", epoch, "cost", batch_cost.numpy())
            
        tr_X = tf.convert_to_tensor(train_list, np.float32)
        tr_Y = tf.convert_to_tensor(train_label, np.float32)
        _, train_acc = loss_accuracy(model, tr_X, tr_Y, thresh, is_training = False)
        print("train accuracy: ", train_acc)
        train_acc_list.append(train_acc)
        val_X = tf.convert_to_tensor(val_list, np.float32)
        val_Y = tf.convert_to_tensor(val_label, np.float32)
        val_loss, val_acc = loss_accuracy(model, val_X, val_Y, thresh, is_training = False)
        val_acc_list.append(val_acc)
        val_loss /= int(val_Y.shape[0])
        print("val loss: ", val_loss, "val accuracy: ", val_acc)

fig = plt.figure()
plt.plot(np.squeeze(np.sqrt(costs)))
plt.ylabel('cost')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("Learning rate: " + str(learning_rate))
fig.savefig('AlexNet_cost.png')
plt.show()

fig = plt.figure()
plt.plot(np.squeeze(train_acc_list))
plt.ylabel('train accuracy')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("Learning rate: " + str(learning_rate) + ", thresh: " + str(thresh))
fig.savefig('AlexNet_train_accuracy.png')
plt.show()

fig = plt.figure()
plt.plot(np.squeeze(val_acc_list))
plt.ylabel('val accuracy')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("Learning rate: " + str(learning_rate) + ", thresh: " + str(thresh))
fig.savefig('AlexNet_val_accuracy.png')
plt.show()

fig = plt.figure()
plt.plot(range(len(val_acc_list)), np.squeeze(train_acc_list), label = "train")
plt.plot(range(len(val_acc_list)), np.squeeze(val_acc_list), label = "val")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('iterations (per %d)' %display_step)
plt.title("Learning rate: " + str(learning_rate) + ", thresh: " + str(thresh))
fig.savefig('AlexNet_accuracy.png')
plt.show()

with tf.device('/device:GPU:0'):
    one = train_list[0, :, :, :]
    one_1 = tf.convert_to_tensor(one, np.float32)
    one_1 = tf.reshape(one_1, [1, one_1.shape[0], one_1.shape[1], one_1.shape[2]])
    predictions_one = model.predict(one_1, is_training = False)
    print(predictions_one)
    print(train_label[0, :, :])
drawLines(train_list[0].copy(),train_label[0].copy(),predictions_one[0])

with tf.device('/device:GPU:0'):
    one = val_list[4, :, :, :]
    one_1 = tf.convert_to_tensor(one, np.float32)
    one_1 = tf.reshape(one_1, [1, one_1.shape[0], one_1.shape[1], one_1.shape[2]])
    predictions_one = model.predict(one_1, is_training = False)
    print(predictions_one)
    print(val_label[0, :, :])
drawLines(val_list[5].copy(),val_label[5].copy(),predictions_one[0])