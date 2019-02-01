import numpy as np
from numpy.random import randint

import random

import os 
from random import sample

import tensorflow as tf

from PIL import Image, ImageOps


x = tf.placeholder(tf.float32, shape=[None, 32, 32,1], name='input')
y = tf.placeholder(tf.float32, shape=[None, 16, 1], name='target')

init = tf.global_variables_initializer()

#y = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#Model
l1 = tf.layers.conv2d(x, 32, 1, activation=tf.nn.relu)
l2 = tf.layers.max_pooling2d(l1, 2,1)
l3 = tf.layers.conv2d(l2, 64, 1, activation=tf.nn.relu)
l4 = tf.layers.max_pooling2d(l3, 1,1)
l5 = tf.layers.conv2d(l4, 64, 1, activation=tf.nn.relu)
l6 = tf.layers.max_pooling2d(l5, 2,1)
l7 = tf.layers.dense(l6, 512, activation=tf.nn.relu)
l8 = tf.layers.dropout(l7)
l9 = tf.reshape(l8, [-1, 16])
l10 = tf.layers.dense(l9, 16, activation=tf.nn.relu)
output = tf.identity(l10, name="output")

loss = tf.reduce_mean(tf.square(output - y), name='loss')
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss, name='train')


def load_images(corpus):
	with open(corpus) as inFile:
		list = []
		images = []
		for line in inFile:
			parts = line.split(' ')
			images.append(np.array(Image.open(parts[0])).reshape(-1, 32, 32, 1).astype("float"))
			labels = []
			for i in parts[1:]:
				labels.append(float(i))
			list.append(labels)
		list = np.array(list)
		images = np.array(list)
		return {
			"data": images,
			"labels": list
		}
		

training_data = load_images('corpus.txt')

print("Training data")

print(len(training_data["data"]))


#model.fit({'input': X}, {'targets': Y}, shuffle=True, batch_size=96, n_epoch=50, validation_set=0.2, show_metric=True, run_id='da-simulated')

#model.save('da.model')

quit()

print("Testing")

folder_names = ["7","8","9","10"]
direction_names = ["Up", "Down", "Left", "Right"]

images = []

for dir in folder_names:
	for subdir in os.listdir(dir):
		images.append(dir + "/" + subdir + "/" + sample(os.listdir(dir + "/" + subdir),1)[0])

predictions = []

for img in images:
	image_vector = np.array(Image.open(img)).reshape(-1, 32, 32, 1).astype("float")
	prediction = model.predict(image_vector)
	#print(img)
	#print(np.round(prediction, 2))

	possibilities = (-prediction[0]).argsort()[:3]

	output = img + "\t\t"
	for pos in possibilities:
		output = output + " " + folder_names[int(pos/len(folder_names))] + "-" + direction_names[pos%len(direction_names)] + "(" + str(round(prediction[0][pos]*100)) + "%) "
		
	print(output)

