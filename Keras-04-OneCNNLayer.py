'''Keras model for MNIST
Based on examples from: 
https://github.com/keras-team/keras/tree/master/examples
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop

# The MNIST data is from the numpy archive file mnist.npz 
# The data is shuffled and split between a train and test set,
# consisting of 60,000 training entries and 10,000 test entries
# Data shape (60000, 28, 28), (10000, 28, 28), type uint8
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The label is a number 0-9, convert to a one-hot vector of size 10
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# For convolution, reshape the image data
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Normalize the gray scale value to float value 0.-1.
# First convert the data to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# then divide by the max value of uint8:  255
x_train /= 255
x_test /= 255

# Build the model:
# CNN + maxpool + flatten + dense 
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

batch_size = 128
epochs = 12
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
