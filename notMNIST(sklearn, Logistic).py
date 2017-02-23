# -*- coding: utf-8 -*-
# @Author: Prateek Sachan
# @Date:   2017-02-18 16:08:41
# @Last Modified by:   Prateek Sachan
# @Last Modified time: 2017-02-20 23:41:48
# link : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_fullyconnected.ipynb

import os
import matplotlib.pyplot as plt
import numpy as np 
import pickle
from sklearn.linear_model import LogisticRegression

def get_data(key):
	global data
	d = data[key]
	if len(d.shape) > 1:
		return d.reshape((data[key].shape[0], 784))
	return d

with open('notMNIST.pickle', 'rb') as file:
	data = pickle.load(file)

	train_data = get_data('train_dataset')
	train_labels = get_data('train_labels')

	valid_data = get_data('valid_dataset')
	valid_labels = get_data('valid_labels')

	test_data = get_data('test_dataset')
	test_labels = get_data('test_labels')


#----------------Logistic Model--------------

model = LogisticRegression(random_state = 1)

print('--------------Model Info-------------')
print('Model : Logistic Regression')
print('Train dataset size :', train_data.shape[0])
print('Validation dataset size :', valid_data.shape[0])

print('-----Training-----')
start = 0
chunk = train_data.shape[0] // 100
end = chunk
for i in range(100):
	train_data_chunk = train_data[start:end, :]
	train_labels_chunk = train_labels[start:end] 
	model.fit(train_data_chunk, train_labels_chunk)
	print('Model Trained : {}%'.format(i + 1), end = '\r')
	start = end
	end += chunk
print('Training Done')

print('Logistic Regresstion Score : {}'.format(model.score(valid_data, valid_labels)))
input()