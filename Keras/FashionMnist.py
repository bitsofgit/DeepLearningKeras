# fashion mnist db

from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K 

# suppresses level 1 and 0 warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# number of class 
# 0	T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot, 
num_classes = 10

# sizes of batch and # of epochs of data
batch_size = 128            # number of samples per gradient update.
epochs = 24                 # number of iterations to train the data

# input image dimensions
img_rows, img_cols = 28, 28     # image is 28 x 28

# the data
# x_train has 60K images of 28 x 28. Each cell containing 0-255 greyscale number. 8 bit greyscale can have 0-255.
# y_train has 60K labels. Ex 9 which means Ankle boot
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Deal with format issues with different backends (Tensorflow, Theano, CNTK etc)
# channels for images are generally either 3 for RGB and 1 for gray scale
# below number 1 denotes that its gray scale
if K.image_data_format() == 'channels_first': 
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else: # Tensor flow uses 'channels_last' so will fall in this else block
    # converts shape from (60000, 28, 28) 3D to (60000, 28, 28, 1) 4D meaning every single value goes in an array of its own
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
    # converts shape from (10000, 28, 28) to (10000, 28, 28, 1)  
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)  
    #input_shape becomes (28,28,1)
    input_shape = (img_rows, img_cols, 1)       

# Type convert and scale the test and training data
# Every value is converted to float and then divided by 255. Earlier the values were between 0 and 255 so after division
# everything becomes between 0 and 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices. One-hot encoding
# so label 3 will become => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model
# 2D because image is 2D
model = Sequential()
# First layer is Conv2D layer
# 32 is number of filters. Each filter is a 3x3 matrix denoted by kernel_size
# input_shape has to be told because this is the first layer
# if activation is provided, it is applied in the end
# so after this layer is done, we will have a 26X26 matrix for the image
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))

# Commenting out this pool layer because some studies have suggested that early pooling helps in 
# increasing accuracy
# model.add(MaxPooling2D(pool_size=(2,2)))

# Second Conv2D layer with 64 filters and each filter of 3x3 matrix and relu activation
# output shape will now be 24x24
model.add(Conv2D(64, (3,3), activation='relu'))

# Pooling layer
# Does MaxPool with pool matrix of 2x2. 
# Strides are 2 so every alternate 2x2 matrix is selected and max value is picked
# output matrix will be 12x12
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Flatten Layer
# flattens the whole input
model.add(Flatten())

# Dense layer for classification
# Dense layer performs the operation output = activation(dot(input, kernel) + bias)
# activation is relu
# kernel is a weight matrix created by the layer
# bias is a bias vector created by the layer if use_bias = true
# 128 is the dimensionality of the output shape
model.add(Dense(128, activation='relu'))

# Dropout layer
# 0.5 is the rate that means that 0.5 of the input units will be dropped
# rate is between 0 and 1
# this is to avoid overfitting of data
# overfitting means a model that learns the training data too well
model.add(Dropout(0.5))

# Another Dense layer whose output shape dimension is 10
# softmax is a math function that is generally used in the final classification layer
# it basically finds the max value 
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# compile
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# train the data
# batch_size = number of samples per gradient update. Default is 32.
# verbose 0 silent, 1 progress bar, 2 one line per epoch
# validation_data is used for evaluation of loss at the end of each epoch. This data is not used for training.
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate
score = model.evaluate(x_test, y_test, verbose = 2)
print('Test loss: ', score[0])
print('Test accuracy:', score[1])

# Plot data 
import numpy as np
import matplotlib.pyplot as plt
epoch_list = list(range(1, len(hist.history['acc'])+1)) #values for x axis
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend('Training Accuracy', 'Validation Accuracy')
plt.show()


