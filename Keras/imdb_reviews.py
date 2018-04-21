# sentiment analysis for imdb reviews

import numpy as np 
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras import models
from keras import layers


# load imdb review data
# num_words 10K means keep the top 10K most frequently occurring words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# decode back to english
def DecodeBackToEnglish(id):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
    # index is offset by 3 because 0,1 and 2 are reserved indices for padding, start and unknown respectively
    decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[id]])
    return decoded_review

#print(DecodeBackToEnglish(10))

# to convert list of integers to tensors
# 2 approaches - one is to pad the other is to do one-hot encode in which we create a 
# 10K dimension vector where let's say your data is [3, 5], then after one-hot encoding it will be a 
# 10K dimension vector with indices 3 and 5 having value 1 and rest having 0
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))     # create an all zero matrix 
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1                        # set specific indices to 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# set aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20,batch_size=512, validation_data=(x_val,y_val))

model.save('imdb_reviews.model.h5')

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')          # bo is for blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')     # b is for solid blue line
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()       # clears the current figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()