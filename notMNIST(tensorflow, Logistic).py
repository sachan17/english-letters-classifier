# -*- coding: utf-8 -*-
# @Author: Prateek Sachan
# @Date:   2017-02-19 16:41:16
# @Last Modified by:   Prateek Sachan
# @Last Modified time: 2017-02-23 19:23:09
# link : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb 
# accuracy = 90%

from __future__ import print_function
import numpy as np 
import tensorflow as tf 
import pickle

image_size = 28
classes = 10

def get_data(train, label):
	global data
	d = data[train].reshape((data[train].shape[0], 784)).astype(np.float32)
	l = (np.arange(classes) == data[label][:,None]).astype(np.float32)#for one hot key
	return d, l

with open('notMNIST.pickle', 'rb') as file:
	data = pickle.load(file)

	train_data, train_labels = get_data('train_dataset', 'train_labels')
	valid_data, valid_labels = get_data('valid_dataset', 'valid_labels')
	test_data, test_labels = get_data('test_dataset', 'test_labels')

#-----------Logistic Regression--------
# Trained using tensor flow
# using Stocastic gradient descent

batch_size = 128

graph = tf.Graph()
with graph.as_default():

	tf_train_data = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(None, classes))
	tf_valid_data = tf.constant(valid_data)
	tf_test_data = tf.constant(test_data)

	# Variables.
	weights = tf.Variable(tf.truncated_normal([image_size * image_size, classes], stddev= np.sqrt(2/(image_size ** 2))))
	biases = tf.Variable(tf.zeros([classes]))

	# Training computation.
	logits = tf.matmul(tf.nn.dropout(x = tf_train_data, keep_prob = 0.75), weights) + biases

	#loss
	beta = 5e-4
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
	regularization = beta * tf.nn.l2_loss(weights)
	loss = tf.reduce_mean(cross_entropy + regularization)

	# Optimizer.
	global_step = tf.Variable(0, trainable = False)
	initial_learning_rate = 0.1
	learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps = 1000, decay_rate = 0.8)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_data, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_data, weights) + biases)

def accuracy(prediction, labels):
	return (100 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0])

num_steps = 10001

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Initialized")
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_data[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]

		feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
		_, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if step % 500 == 0:
			print('----Result at step {}----'.format(step))
			print('Loss : {}'.format(l))
			print('Training Accuracy : {}'.format(accuracy(prediction, batch_labels)))
			print('Valid Accuracy : {}'.format(accuracy(valid_prediction.eval(), valid_labels)))
			print()

	print('Final Test Accuracy : {}'.format(accuracy(test_prediction.eval(), test_labels)))
	final_weights, final_biases = session.run([weights, biases])
	data = {
			'weights' : final_weights,
			'biases' : final_biases
	}
	file = open('logistic data.pickle', 'wb')
	pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
	file.close()
	print('Model parameters saved in logistic data.pickle file.')
	
	