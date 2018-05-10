from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from stage_1_model_v1 import *
import stage_1_util_v1

# Set Eager API
tfe.enable_eager_execution()

# Training Data (FLIC)
# If load from Matlab, each image has labels (2, 11)
# May have to write Matlab into csv file and then load

# Training Data (LSP)
# If load from Matlab, each image has labels (3, 14)
# the 3 is x, y, z. z always = 0
# May have to write Matlab into csv file and then load

# This code assume FLIC Y (N, 2, 11) easy to change to other dataset
train_X, train_Y = load_data()

n_samples = train_Y.shape[0] # N
n_joints = train_Y.shape[2]

# Parameters
learning_rate = 0.0005
batch_size = 128 # TODO: implement mini-batch
display_step = 100
epochs = 1000
n_batch = n_samples / batch_size + 1


is_training = True

# Mean square error (L2 loss)
def mean_square_fn(labels, predictions):
	return tf.losses.mean_squared_error(labels, predictions)

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

	# TODO: shuffle data

	batch_cost = 0.0

	for batch in range(n_batch):
		batch_X, batch_Y = next_batch() # TODO: implement this, or use tf API
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









