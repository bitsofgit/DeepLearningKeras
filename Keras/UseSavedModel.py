# use saved model.py
import keras
import numpy as np

from keras.models import load_model
from keras.preprocessing import sequence
from keras.datasets import imdb

# suppresses level 1 and 0 warning messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# load the saved model
model = load_model('sentiment_lstm.model')

# Compile
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# get data to predict
x_data = [[2, 2, 2, 2, 33, 2804, 2, 2040, 432, 111, 153, 103, 2, 1494, 13, 70, 131, 67, 11, 61, 2, 744, 35, 3715, 761, 61, 
5766, 452, 2, 2, 985, 2, 2, 59, 166, 2, 105, 216, 1239, 41, 1797, 2, 15, 2, 35, 744, 2413, 31, 2, 2, 687, 23, 2, 2, 2, 2, 
3693, 42, 38, 39, 121, 59, 456, 10, 10, 2, 265, 12, 575, 111, 153, 159, 59, 16, 1447, 21, 25, 586, 482, 39, 2, 96, 59, 716, 
12, 2, 172, 65, 2, 579, 11, 2, 2, 1615, 2, 2, 2, 5168, 17, 13, 2, 12, 19, 2, 464, 31, 314, 11, 2, 2, 719, 605, 11, 2, 202, 
27, 310, 2, 3772, 3501, 2, 2722, 58, 10, 10, 537, 2116, 180, 40, 14, 413, 173, 2, 263, 112, 37, 152, 377, 2, 537, 263, 846, 
579, 178, 54, 75, 71, 476, 36, 413, 263, 2504, 182, 2, 17, 75, 2306, 922, 36, 279, 131, 2895, 17, 2867, 42, 17, 35, 921, 2, 
192, 2, 1219, 3890, 19, 2, 217, 4122, 1710, 537, 2, 1236, 2, 736, 10, 10, 61, 403, 2, 2, 40, 61, 4494, 2, 27, 4494, 159, 90, 
263, 2311, 4319, 309, 2, 178, 2, 82, 4319, 2, 65, 15, 2, 145, 143, 5122, 12, 2, 537, 746, 537, 537, 15, 2, 2, 2, 594, 2, 
5168, 94, 2, 3987, 2, 11, 2, 2, 538, 2, 1795, 246, 2, 2, 2, 11, 635, 14, 2, 51, 408, 12, 94, 318, 1382, 12, 47, 2, 2683, 936, 
2, 2, 2, 19, 49, 2, 2, 1885, 2, 1118, 25, 80, 126, 842, 10, 10, 2, 2, 4726, 27, 4494, 11, 1550, 3633, 159, 27, 341, 29, 2733, 
19, 4185, 173, 2, 90, 2, 2, 30, 11, 2, 1784, 86, 1117, 2, 3261, 46, 11, 2, 21, 29, 2, 2841, 23, 2, 1010, 2, 793, 2, 2, 1386, 
1830, 10, 10, 246, 50, 2, 2, 2750, 1944, 746, 90, 29, 2, 2, 124, 2, 882, 2, 882, 496, 27, 2, 2213, 537, 121, 127, 1219, 130, 
2, 29, 494, 2, 124, 2, 882, 496, 2, 341, 2, 27, 846, 10, 10, 29, 2, 1906, 2, 97, 2, 236, 2, 1311, 2, 2, 2, 2, 31, 2, 2, 91, 2, 
3987, 70, 2, 882, 30, 579, 42, 2, 12, 32, 11, 537, 10, 10, 11, 14, 65, 44, 537, 75, 2, 1775, 3353, 2, 1846, 2, 2, 2, 154, 2, 2, 
518, 53, 2, 2, 2, 3211, 882, 11, 399, 38, 75, 257, 3807, 19, 2, 17, 29, 456, 2, 65, 2, 27, 205, 113, 10, 10, 2, 2, 2, 2, 2, 242, 
2, 91, 1202, 2, 2, 2070, 307, 22, 2, 5168, 126, 93, 40, 2, 13, 188, 1076, 3222, 19, 2, 2, 2, 2348, 537, 23, 53, 537, 21, 82, 40, 
2, 13, 2, 14, 280, 13, 219, 2, 2, 431, 758, 859, 2, 953, 1052, 2, 2, 5991, 2, 94, 40, 25, 238, 60, 2, 2, 2, 804, 2, 2, 2, 2, 132, 
2, 67, 2, 22, 15, 2, 283, 2, 5168, 14, 31, 2, 242, 955, 48, 25, 279, 2, 23, 12, 1685, 195, 25, 238, 60, 796, 2, 2, 671, 2, 2804, 
2, 2, 559, 154, 888, 2, 726, 50, 26, 49, 2, 15, 566, 30, 579, 21, 64, 2574]]




x_data = sequence.pad_sequences(x_data, maxlen=400)

#print('x_data.shape:', x_data.shape)

y_data = model.predict(x_data)

#print(y_data.shape)
#print(y_data)

NUM_WORDS = 6000        # top most n frequent words to consider
SKIP_TOP = 0            # skip the top most words that are likely like the, and, a etc
MAX_REVIEW_LEN = 400    # Max number of words from review

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS, skip_top=SKIP_TOP)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_REVIEW_LEN)

testitem = 7
#print(x_test[testitem])
#print("Original Estimate: ", y_test[testitem])

#print(x_test.shape)
#print(x_test[testitem].shape)
#y = x_test[testitem].reshape((1,MAX_REVIEW_LEN))
#print(y.shape)

#newy = model.predict(y, verbose=0)
#print("New Prediction: ", newy)

REVIEW_NUM = 10

INDEX_FROM = 3
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

#input_text = ' '.join(id_to_word[id] for id in x_test[REVIEW_NUM] )
input_text = "The movie was just amazing. guaranteed to please. amazing acting and story. brilliant. It was very realistic and you feel for andy. Morgan Freeman's acting was truly great."
word_index = imdb.get_word_index()
#input_text = "this film was just brilliant casting story  direction really the part they played and you could just imagine being there robert is an amazing actor"
print("Input text: ", input_text)
#print("Original Prediction: ", y_test[REVIEW_NUM])
words = input_text.split()
inputarray = np.array([word_index[word] if word in word_index else 0 for word in words])
#print(inputarray)
preprocessed_input = []
for i in inputarray:
    preprocessed_input.append(i)
#print(preprocessed_input)
preprocessed_input = np.asarray(preprocessed_input)
preprocessed_input = sequence.pad_sequences(preprocessed_input.reshape((1,len(preprocessed_input))), maxlen=MAX_REVIEW_LEN)
#print(preprocessed_input.shape)
prediction = model.predict(preprocessed_input, verbose=1)
print("Prediction: ", prediction)
#prediction = model.predict(x_test[REVIEW_NUM].reshape((1,MAX_REVIEW_LEN)),verbose=1)
#print("Prediction: ", prediction)

