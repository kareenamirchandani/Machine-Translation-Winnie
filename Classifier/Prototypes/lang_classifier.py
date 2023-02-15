import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
import pickle
import csv

vocab_size = 90000  # tokenizer will keep the top 50000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 105025

with open("../sunbird_split_new.json", 'r') as f:
    data = [json.loads(line) for line in f]

sentences = []
labels = []
sentences_and_labels = [[] for x in range(len(data[0]))]

# Each item in labels list is a list with 0's and 1's
# e.g. if the language is English, the list will be [1 0 0 0 0 0]
# if the language is Lugbara, the list will be [0 0 0 0 1 0]
# I have made the labels in this way to treat this is a multilabel classification task
# This enables detection of code mixing

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

with open("../sunbirdData.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(sentences_and_labels)

# print(sentences[0:10])
# print(labels[0:10])
print(len(sentences))

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

with open('../tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index
#print(len(word_index))
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3)

X_train, X_test, y_train, y_test = training_padded, testing_padded, training_labels, testing_labels

# Create the model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 10
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)


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

model.save("trained_model.h5")

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
