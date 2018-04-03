# Defines a network that can find separate circles of data

# Imports
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt

# suppresses level 1 and 0 warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Helper functions
# plot data on a figure
def plot_data(pl, X, y):
    # plot class where y == 0
    pl.plot(X[y==0, 0], X[y==0,1],'ob', alpha=0.5)
    # plot class where y == 1
    pl.plot(X[y==1,0], X[y==1,1],'xr', alpha=0.5)
    pl.legend(['0','1'])
    return pl


# function that draws decision boundaries
def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make prediction with the model and reshape the output so contourf can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12,8))
    # plot the contour
    plt.contour(aa, bb, Z, cmap='bwr', alpha=0.2)
    # plot the moons of data
    plot_data(plt, X, y)

    return plt

# Generate some data blobs. Data will be either 0 or 1 when 2 is number of centers
# X is a [num of samples, 2] sized array
# ex. X[1] = [1.342, -2.3], X[2] = [-4.342, 2.12]
# y is a [num of samples] sized array. 
# ex. y[1]=0, y[1]=1
# centers 2 tells the function to return 2 clusters - one for class 0 and one for class 1
#X, y = make_blobs(n_samples=1000, centers=2, random_state=42) 

X,y = make_circles(n_samples=1000, factor=0.6, noise=0.1, random_state=42)

pl = plot_data(plt, X, y)
pl.show()

# Split the data into Training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=42)

# general Keras programming model
# Create model, add layers, compile model, train model (via fit), evaluate performance

# create the keras model
from keras.models import Sequential
from keras.layers import Dense # In Dense layer, every neuron is connected to every other neuron in the next layer
from keras.optimizers import Adam

# Simple sequential model
model = Sequential()
# add a dense fully connected layer with 1 neuron
# sigmoid activation function is used to return 0 or 1
model.add(Dense(4,input_shape=(2,),activation="tanh", name="Hidden-1"))
model.add(Dense(4,activation="tanh", name="Hidden-2"))
model.add(Dense(1,activation="sigmoid", name="Output"))
model.summary()

# compile model
model.compile(Adam(lr=0.05),'binary_crossentropy', metrics=['accuracy'])

# from keras.utils import plot_model
# plot_model(model, to_file="deep_circles_model.png", show_layer_names=True, show_shapes=True)

# Define early stopping callback if the accuracy is not improving
from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor='val_acc', patience=5, mode=max)]

# Fit the model. Make 100 cycles through the data. Verbose = 0 means suppress progress messages
# in each cycle the optimizer will try to improve the accuracy
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))
# Get loss and accuracy on test data
eval_result = model.evaluate(X_test, y_test)
# Print test accuracy
print("\n\nTest loss : ", eval_result[0], "Test accuracy:", eval_result[1])
# Plot the decision boundary
plot_decision_boundary(model, X, y).show()





    