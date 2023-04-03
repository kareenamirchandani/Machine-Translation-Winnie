import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tabulate import tabulate

vocab_size = 90000  # tokenizer will keep the top 50000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 105025

fields = ['sentence', 'English', 'Luganda', 'Runyankole', 'Ateso', 'Lugbara', 'Acholi']
with open("sunbird_split_new.json", 'r') as f:
    data = [json.loads(line) for line in f]

sentences = []
labels = []
sentences_and_labels = [[] for x in range(len(data[0]))]

counter = 0
for item in data[0]:
    sentences.append(item['sentence'])
    sentences_and_labels[counter].append(item['sentence'])
    # labels.append(item['language'])
    labels.append([item['English'], item['Luganda'], item['Runyankole'],
                   item['Ateso'], item['Lugbara'], item['Acholi']])
    sentences_and_labels[counter].append(item['English'])
    sentences_and_labels[counter].append(item['Luganda'])
    sentences_and_labels[counter].append(item['Runyankole'])
    sentences_and_labels[counter].append(item['Ateso'])
    sentences_and_labels[counter].append(item['Lugbara'])
    sentences_and_labels[counter].append(item['Acholi'])
    counter += 1

with open("sunbirdData.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(sentences_and_labels)

df = pd.read_csv("sunbirdData.csv", encoding='ISO-8859-1')
df = df.replace(np.nan, '', regex=True)
# print(tabulate(df, headers='keys'))
print(df.shape)
print(df["sentence"][168])

# print("English:" + str(df["English"][168]))
# print("Luganda:" + str(df["Luganda"][168]))
# print("Runyankole:" + str(df["Runyankole"][168]))
# print("Ateso:" + str(df["Ateso"][168]))
# print("Lugbara:" + str(df["Lugbara"][168]))
# print("Acholi:" + str(df["Acholi"][168]))

df_labels = df[["English", "Luganda", "Runyankole", "Ateso", "Lugbara", "Acholi"]]

X = list(df["sentence"])
y = df_labels.values
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

# vocab_size = len(tokenizer.word_index) + 1

print('Length of X_train:', len(X_train))
print('Length of X_test:', len(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(len(X_train), len(X_test), len(y_train), len(y_test))
word_index = tokenizer.word_index

X_train = pad_sequences(X_train, padding=padding_type, maxlen=max_length, truncating=trunc_type)
X_test = pad_sequences(X_test, padding=padding_type, maxlen=max_length, truncating=trunc_type)

with open('tokenizer_softmax.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/
# This article uses Glove embeddings as a next step

'''
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))

for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

'''
# Create the model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

'''
deep_inputs = Input(shape=(max_length,))
embedding_layer = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 10
history = model.fit(X_train, y_train, epochs=num_epochs,
                    validation_data=(X_test, y_test), verbose=2)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("accuracy")

model.save("trained_model_softmax2.h5")

'''
# Luganda validation sentence
# Translates to 'Could I use your phone?'

validation_sentence = 'Nsabaku simu yo'
val_sequences = tokenizer.texts_to_sequences(validation_sentence)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# print(val_padded)

proba = model.predict(val_padded)
# print(proba)
proba = np.asarray(proba, dtype=float)

prediction = np.sum(proba, 0)

print('Luganda prediction:', prediction)

# validation_sentence = 'Arai ilosi ijo, abuni aupar ke ijo.' Ateso
# Translates to 'If you are going, I shall accompany you'

# English validation sentence

validation_sentence = 'The sun is shining, it is a beautiful day.'
val_sequences = tokenizer.texts_to_sequences(validation_sentence)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

proba = model.predict(val_padded)
proba = np.asarray(proba, dtype=float)

prediction = np.sum(proba, 0)

print('English prediction:', prediction)

# The predictions aren't really making sense
# Look for more multilabel text examples'''
