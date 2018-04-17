# classify imdb reviews based on sentiment - positive or negative
# actual words are encoded so second most popular word is replaced by 2, third most popular by 3 etc

from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb

# suppresses level 1 and 0 warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

NUM_WORDS = 6000        # top most n frequent words to consider
SKIP_TOP = 0            # skip the top most words that are likely like the, and, a etc
MAX_REVIEW_LEN = 400    # Max number of words from review

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS, skip_top=SKIP_TOP)

# Print a sample
# print("encoded word sequence: ", x_train[3])

x_train = sequence.pad_sequences(x_train, maxlen=MAX_REVIEW_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_REVIEW_LEN)

print('x_train.shape:', x_train.shape, 'x_test.shape: ', x_test.shape)

model = Sequential()
model.add(Embedding(NUM_WORDS, 64))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
BATCH_SIZE = 24
EPOCHS = 5
cbk_early_stopping = EarlyStopping(monitor='val_acc', mode='max')

# train
#model.fit(x_train, y_train, BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[cbk_early_stopping])

# save model
#model.save('sentiment_lstm.model')




