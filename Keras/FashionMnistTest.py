# fashion mnist db

from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K 

import numpy as np

# suppresses level 1 and 0 warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# number of class 
# 0	T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot, 
num_classes = 10

# sizes of batch and # of epochs of data
batch_size = 128            # number of samples per gradient update.
epochs = 1                  # number of iterations to train the data

# input image dimensions
img_rows, img_cols = 28, 28     # image is 28 x 28

# the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

def printTensorAttribs(x):
    result = "Dimensions: " + str(x.ndim) + " Shape: " + str(x.shape) + " DataType: " + str(x.dtype)
    print(result)

def prettyPrint(s):
    print('\n ------ ' + s + ' ------\n')

# # print 10 of the test data
# printTensorAttribs(x_train)
# print(x_train[0])
# print("\n")
# printTensorAttribs(y_train)
# print(y_train[0])
# print("\n")

# print(K.image_data_format())

# printTensorAttribs(x_train)
# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
# print(x_train[0])
# printTensorAttribs(x_train)

# printTensorAttribs(x_test)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
# printTensorAttribs(x_test)

x = np.array([[0,100,200,255],[5,6,7,8],[9,10,11,12]])
x = x.astype('float32')
x /= 255
print(x)
printTensorAttribs(x)

print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train[0])
