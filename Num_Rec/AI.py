#import libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
import numpy 
import pandas as pd

import matplotlib.pyplot as plt

#load dataset directly from keras library 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#plot first six smaples of MNIST training dataset as gray scale image

for i in range(6):
    plt.subplot(int('23' + str(i+1)))
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    #plt.waitforbuttonpress()

#reshape format [smaples][width][height][channels]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Converts a class vector(integers) to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#normalize inputs
x_train = x_train/255
x_test = x_test/255
