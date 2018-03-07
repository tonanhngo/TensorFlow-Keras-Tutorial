'''Keras model for MNIST
Based on examples from: 
https://github.com/keras-team/keras/tree/master/examples
'''

from __future__ import print_function

import keras
from keras.datasets import mnist

# The MNIST data is from the numpy archive file mnist.npz 
# The data is shuffled and split between a train and test set,
# consisting of 60,000 training entries and 10,000 test entries
# Data shape (60000, 28, 28), (10000, 28, 28), type uint8
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The label is a number 0-9, convert to a one-hot vector of size 10
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
