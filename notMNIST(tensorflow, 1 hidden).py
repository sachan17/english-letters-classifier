# -*- coding: utf-8 -*-
# @Author: Prateek Sachan
# @Date:   2017-02-21 18:31:02
# @Last Modified by:   Prateek Sachan
# @Last Modified time: 2017-02-22 15:20:01


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


#-----------1 Hidden layer---------
# using relu for activation
# sgd implementation for fast result

hidden_layer_size = 1024

graph = tf.Graph()
with graph.as_default():
	# placeholders
	tf_train_data = tf.placeholder(tf.float32, shape = (None, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape = (None, classes))
	tf_valid_data = tf.constant(valid_data)
	tf_test_data = tf.constant(test_data)

	#variables
	weights = [
				tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size])),
				tf.Variable(tf.truncated_normal([hidden_layer_size, classes]))
			]
	biases = [
				tf.Variable(tf.zeros([hidden_layer_size])),
				tf.Variable(tf.zeros([classes]))
			]  

	#training computations
	hidden_layer = tf.matmul(tf_train_data, weights[0]) + biases[0]
	hidden_layer = tf.nn.relu(hidden_layer)

	output = tf.matmul(hidden_layer, weights[1]) + biases[1]

	#loss
	beta = 5e-4
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = output)
	regularization = beta * (tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(weights[1])) 
	loss = tf.reduce_mean(cross_entropy + regularization)

	# optimizer
	learning_rate = 0.1
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	train_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_train_data, weights[0]) + biases[0]), weights[1]) + biases[1])
	valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_data, weights[0]) + biases[0]), weights[1]) + biases[1])
	test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_data, weights[0]) + biases[0]), weights[1]) + biases[1])

def accuracy(prediction, labels):
	return (100 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0])

num_steps = 3001
batch_size = 128

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_data[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size)]

		feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
		_, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

		if step % 500 == 0:
			print('----Result at Mini step {}----'.format(step))
			print('Loss : {}'.format(l))
			print('Training Accuracy : {}'.format(accuracy(prediction, batch_labels)))
			print('Valid Accuracy : {}'.format(accuracy(valid_prediction.eval(), valid_labels)))
			print()

	print('Final Test Accuracy : {}'.format(accuracy(test_prediction.eval(), test_labels)))