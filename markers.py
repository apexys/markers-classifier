import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


# Build the preloader array, resize images to 128x128
from tflearn.data_utils import image_preloader
from tflearn.metrics import R2

#%%
import numpy as np
from numpy.random import randint

import random

import os 
from random import sample

import tensorflow as tf

from PIL import Image, ImageOps

#%%

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)


def defineArchitecture():

	# Input is a 32x32 image with 1 color channels (grayscale)
	network = convnet = input_data(shape=[None, 32, 32, 1], data_preprocessing=img_prep,
                     data_augmentation=img_aug, name='input')
	
	network = conv_2d(network, 32, 1, activation='relu', regularizer='L2')
	
	network = max_pool_2d(network, 2)
	
	network = conv_2d(network, 64, 1, activation='relu', regularizer='L2')
	
	network = max_pool_2d(network, 1)

	network = conv_2d(network, 64, 1, activation='relu', regularizer='L2')
	
	network = max_pool_2d(network, 2)
	
	# Step 6: Fully-connected 512 node neural network
	network = fully_connected(network, 512, activation='relu')
	
	network = dropout(network, 0.5)

	network = fully_connected(network, 16, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

	return network
#%%


model = tflearn.DNN(defineArchitecture(), tensorboard_verbose=0, tensorboard_dir="/output")

train_images = []
train_labels = []

test_images = []
test_labels = []

train_dataset_file = 'files.txt'

feature_set = 'corpus.txt'
with open(feature_set) as inFile:
	list = []
	for line in inFile:
		parts = line.split(' ')
		labels = []
		for i in parts[1:]:
			labels.append(float(i))
		list.append(labels)
	Y = np.array(list)

X, _ = image_preloader(train_dataset_file, image_shape=(32, 32),   mode='file', categorical_labels=False, normalize=False, grayscale=True)
X = np.reshape(X, (-1, 32, 32, 1))

model.fit({'input': X}, {'targets': Y}, shuffle=True, batch_size=96, n_epoch=300, validation_set=0.2, show_metric=True, run_id='da-simulated')

del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

model.save('da.model')



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

