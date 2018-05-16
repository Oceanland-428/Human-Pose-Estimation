import numpy as numpy
import tensorflow as tf
import stage_1_util_v1

params = stage_1_util_v1.params
convDict = params['conv']
poolDict = params['pool']
denseDict = params['dense']

# The data format for the layers is channel_last

class stage_1_model(object):
	def __init__(self, num_joints):
		self.conv1 = tf.layers.Conv2D(
			filters = convDict['filter1'],
			kernel_size = convDict['kernel1'],
			strides = convDict['stride1'],
			padding = 'valid',
			activation = None)

		self.bn1 = tf.layers.BatchNormalization(
			center = True,
			scale = True)

		#self.lrelu1 = tf.nn.leaky_relu(bn1)

		self.pool1 = tf.layers.MaxPooling2D(
			pool_size = poolDict['pSize1'],
			strides = poolDict['pStride1'])

		self.conv2 = tf.layers.Conv2D(
			filters = convDict['filter2'],
			kernel_size = convDict['kernel2'],
			strides = convDict['stride2'],
			padding = 'same',
			activation = None)

		self.bn2 = tf.layers.BatchNormalization(
			center = True,
			scale = True)

		#self.lrelu2 = tf.nn.leaky_relu(bn2)

		self.pool2 = tf.layers.MaxPooling2D(
			pool_size = poolDict['pSize2'],
			strides = poolDict['pStride2'])

		self.conv3 = tf.layers.Conv2D(
			filters = convDict['filter3'],
			kernel_size = convDict['kernel3'],
			strides = convDict['stride3'],
			padding = 'same',
			activation = tf.nn.relu)

		self.conv4 = tf.layers.Conv2D(
			filters = convDict['filter4'],
			kernel_size = convDict['kernel4'],
			strides = convDict['stride4'],
			padding = 'same',
			activation = tf.nn.relu)

		self.conv5 = tf.layers.Conv2D(
			filters = convDict['filter5'],
			kernel_size = convDict['kernel5'],
			strides = convDict['stride5'],
			padding = 'same',
			activation = tf.nn.relu)

		self.pool3 = tf.layers.MaxPooling2D(
			pool_size = poolDict['pSize3'],
			strides = poolDict['pStride3'])

		self.dense1 = tf.layers.Dense(
			units = denseDict['unit1'],
			activation = tf.nn.relu)

		self.dropout1 = tf.layers.Dropout(
			rate = 0.6)

		self.dense2 = tf.layers.Dense(
			units = denseDict['unit2'],
			activation = tf.nn.relu)
		self.logits = tf.layers.Dense(
			units = 2 * num_joints)

	def predict(self, X, is_training):
		x = self.conv1(X)
		x = self.bn1(x)
		x = tf.nn.relu(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = tf.nn.relu(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.pool3(x)
		x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
		x = self.dense1(x)
		#print(is_training)
		x = self.dropout1(x, training = is_training)
		x = self.dense2(x)
		logits = self.logits(x)
		#print(logits.shape)
		pred = tf.reshape(logits, [X.shape[0], 2, -1])

		return pred




