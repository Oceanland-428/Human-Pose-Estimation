from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from stage_1_model_v1 import *
import stage_1_util_v1
import sys
sys.path.insert(0, './lsp_dataset')
import parse_lsp_data

# Set Eager API
tfe.enable_eager_execution()

# Training Data (FLIC)
# If load from Matlab, each image has labels (2, 11)
# May have to write Matlab into csv file and then load

# Training Data (LSP)
# If load from Matlab, each image has labels (2, 14)
# the 3 is x, y, z. z always = 0
# May have to write Matlab into csv file and then load

# This code assume LSP Y (N, 3, 14) easy to change to other dataset
train_list, train_label, val_list, val_label, test_list, test_label = getLSPDataset(0.8, 0.1)
train_X = train_list	# (N, 227, 227, 3)
train_Y = train_label	# (N, 2, 14)


n_samples = train_Y.shape[0] # N
n_joints = train_Y.shape[2]

# Parameters
learning_rate = 0.0005
batch_size = 128
display_step = 100
epochs = 1000
n_batch = n_samples / batch_size + 1


is_training = True

# Mean square error (L2 loss)
def mean_square_fn(labels, predictions):
	return tf.losses.mean_squared_error(labels, predictions)

def next_batch(batch, batch_size):
	if batch < n_batch - 1:
		batch_X = train_list[batch * batch_size: (batch + 1) * batch_size, :, :, :]
		batch_Y = train_label[batch * batch_size: (batch + 1) * batch_size, :, :]
	else:
		batch_X = train_list[batch * batch_size:, :, :, :]
		batch_Y = train_label[batch * batch_size:, :, :]
	return batch_X, batch_Y


# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

# Compute gradients
grad = tfe.implicit_gradients(mean_square_fn)

# Initial cost, before optimizing
print("Initial cost = {:.9f}".format(mean_square_fn(pred, batch_Y)))

# Training
costs = []
params = stage_1_util_v1.params

for epoch in range(epochs):

	batch_cost = 0.0

	for batch in range(n_batch):
		batch_X, batch_Y = next_batch(batch, batch_size)
		pred = stage_1_model(batch_X, n_joints, is_training, params) # (N, 2, # joints)
		batch_cost += mean_square_fn(pred, batch_Y)
		optimizer.apply_gradients(grad(train_Y, pred))

	batch_cost /= n_samples
	costs.append(batch_cost)

	if epoch % display_step == 0:
		print("Epoch:", '%04d' % epoch, "cost=",  "{:.9f}".format(batch_cost))

# visualize costs
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()









