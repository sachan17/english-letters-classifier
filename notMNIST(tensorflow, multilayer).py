# -*- coding: utf-8 -*-
# @Author: Prateek Sachan
# @Date:   2017-02-21 18:31:02
# @Last Modified by:   Prateek Sachan
# @Last Modified time: 2017-02-23 20:39:50
# accuracy : 94.5%

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
input_layer_size = image_size * image_size
hidden_layer_size = [1024, 350, 75]

graph = tf.Graph()
with graph.as_default():
	# placeholders
	tf_train_data = tf.placeholder(tf.float32, shape = (None, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape = (None, classes))
	tf_valid_data = tf.constant(valid_data)
	tf_test_data = tf.constant(test_data)

	#variables
	weights = [
				tf.Variable(tf.truncated_normal([input_layer_size, hidden_layer_size[0]], stddev = np.sqrt(2.0 / input_layer_size))),
				tf.Variable(tf.truncated_normal([hidden_layer_size[0], hidden_layer_size[1]], stddev = np.sqrt(2.0 / hidden_layer_size[0]))),
				tf.Variable(tf.truncated_normal([hidden_layer_size[1], hidden_layer_size[2]], stddev = np.sqrt(2.0 / hidden_layer_size[1]))),				
				tf.Variable(tf.truncated_normal([hidden_layer_size[2], classes], stddev = np.sqrt(2.0 / hidden_layer_size[2])))
			]
	biases = [
				tf.Variable(tf.zeros([hidden_layer_size[0]])),
				tf.Variable(tf.zeros([hidden_layer_size[1]])),
				tf.Variable(tf.zeros([hidden_layer_size[2]])),
				tf.Variable(tf.zeros([classes]))
			]  

	#training computations
	# 1st hidden layer

	def layer_computation(input_layer, dropout = False):
		if dropout:
			input_layer = tf.nn.dropout(x = input_layer, keep_prob = 0.75)
		hidden_layer_1 = tf.nn.relu(tf.matmul(input_layer, weights[0]) + biases[0])

		if dropout:
			hidden_layer_1 = tf.nn.dropout(x = hidden_layer_1, keep_prob = 0.75)
		hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights[1]) + biases[1])

		if dropout:
			hidden_layer_2 = tf.nn.dropout(x = hidden_layer_2, keep_prob = 0.75)
		hidden_layer_3 = tf.nn.relu(tf.matmul(hidden_layer_2, weights[2]) + biases[2])

		if dropout:
			hidden_layer_3 = tf.nn.dropout(x = hidden_layer_3, keep_prob = 0.75)
		output_layer = tf.nn.relu(tf.matmul(hidden_layer_3, weights[3]) + biases[3])

		return output_layer


	output = layer_computation(tf_train_data, True)
	#loss
	beta = 0.05
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = output)
	regularization = beta * (tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(weights[1]) + tf.nn.l2_loss(weights[2]) + tf.nn.l2_loss(weights[3])) 
	loss = tf.reduce_mean(cross_entropy + regularization)

	# optimizer
	global_step = tf.Variable(0, trainable = False)
	initial_learning_rate = 0.1
	learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 5000, 0.9)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	train_prediction = tf.nn.softmax(layer_computation(tf_train_data, False))
	valid_prediction = tf.nn.softmax(layer_computation(tf_valid_data, False))
	test_prediction = tf.nn.softmax(layer_computation(tf_test_data, False))

def accuracy(prediction, labels):
	return (100 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0])

num_steps = 50001
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

		if step % 1000 == 0:
			print('----Result at Mini step {}----'.format(step))
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
	file = open('multilayer_data.pickle', 'wb')
	pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
	file.close()
