import numpy as numpy
import tensorflow as tf
import stage_1_util_v1

params = stage_1_util_v1.params

def stage_1_model(X, num_joints, is_training, params):
	# X in shape of (N, H, W, C) # need to change BatchNorm if this data format(NHWC) is changed
	# H = W = 220
	N = X.shape[0]
	convDict = params['conv']
	poolDict = params['pool']
	denseDict = params['dense']

	# NOTE:
	# According to the paper, they used AlexNet...
	# However, the AlexNet used 227 as input size whereras the paper used 220 input size,
	# I don't know how thay got the same layer dimensions as the AlexNet...
	# I implement the AlexNet here, but we may need to up-sample the input from 220 to 227
	# if we used the raw image directly from the dataset, we need to change the image size to 227

	# In AlexNet: conv -> relu -> pool -> norm, norm is not common now
	# In paper: conv -> relu -> LRN -> pool, don't know how to implement LRN...
	# Here: conv -> batchnorm -> lrelu -> pool, which is consistent with what we learnt

	# for conv layers, three numbers are: # filters, kernel size, stride
	# 96, 11, 4
	# input: 227, output: 55
	conv1 = tf.layers.conv2d(
		inputs = X,
		filters = convDict['filter1'],
		kernel_size = convDict['kernel1'],
		strides = convDict['stride1'],
		padding = 'valid',
		activation = None)

	bn1 = tf.contrib.layers.batch_norm(
		inputs = conv1,
		center = True,
		scale = True,
		is_training = is_training)

	lrelu1 = tf.nn.leaky_relu(bn1)

	# for pool layers, two numbers are: pooling mask size, stride
	# 3, 2
	# input : 55, output: 27
	pool1 = tf.layers.max_pooling2d(
		inputs = lrelu1,
		pool_size = poolDict['pSize1'],
		strides = poolDict['pStride1'])

	# 256, 5, 1
	# input: 27, output: 27
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = convDict['filter2'],
		kernel_size = convDict['kernel2'],
		strides = convDict['stride2'],
		padding = 'same',
		activation = None)

	bn2 = tf.contrib.layers.batch_norm(
		inputs = conv2,
		center = True,
		scale = True,
		is_training = is_training)

	lrelu2 = tf.nn.leaky_relu(bn2)

	# 3, 2
	# input : 27, output: 13
	pool2 = tf.layers.max_pooling2d(
		inputs = lrelu2,
		pool_size = poolDict['pSize2'],
		strides = poolDict['pStride2'])

	# 384, 3, 1
	# input: 13, output: 13
	conv3 = tf.layers.conv2d(
		inputs = pool2,
		filters = convDict['filter3'],
		kernel_size = convDict['kernel3'],
		strides = convDict['stride3'],
		padding = 'same',
		activation = tf.nn.leaky_relu)

	# 384, 3, 1
	# input: 13, output: 13
	conv4 = tf.layers.conv2d(
		inputs = conv3,
		filters = convDict['filter4'],
		kernel_size = convDict['kernel4'],
		strides = convDict['stride4'],
		padding = 'same',
		activation = tf.nn.leaky_relu)

	# 256, 3, 1
	# input: 13, output: 13
	conv5 = tf.layers.conv2d(
		inputs = conv4,
		filters = convDict['filter5'],
		kernel_size = convDict['kernel5'],
		strides = convDict['stride5'],
		padding = 'same',
		activation = tf.nn.leaky_relu)

	# 3, 2
	# input: 13, output: 6
	pool3 = tf.layers.max_pooling2d(
		inputs = conv5,
		pool_size = poolDict['pSize3'],
		strides = poolDict['pStride3'])

	flat = tf.reshape(pool3, [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3]])

	# for dense layers, the number is: # hidden neurons
	# 4096
	# input: 4096, output: 4096
	dense1 = tf.layers.dense(
		inputs = flat,
		units = denseDict['unit1'],
		activation = tf.nn.leaky_relu)

	dropout1 = tf.layers.dropout(
		inputs = dense1,
		rate = 0.6,
		training = is_training)

	# 4096
	# input: 4096, output: 4096
	dense2 = tf.layers.dense(
		inputs = dropout1,
		units = denseDict['unit2'],
		activation = tf.nn.leaky_relu)

	# units used # of joints * 2, x and y coord for each joint
	# first half of units are x coods, second half are y coods
	# eg. tensor[x1, x2, x3, y1, y2, y3]
	# input: 4096, output: 2 * num_joints
	logits = tf.layers.dense(
		inputs = dense2,
		units = 2 * num_joints)

	# reshape to (N, 2, # of joints)
	# eg. tensor[x1, x2, x3, y1, y2, y3] -> tensor[[x1, y1], [x2, y2], [x3, y3]]
	pred = tf.reshape(logits, [N, 2, -1])

	return pred




