import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from datasets import load_from_disk

MT560 = load_from_disk('C:/Users/emike/MT560')
# SALT_SPLIT contains English, src and src_lang

# Lug, lgg, ach, teo, nyn, swa

vocab_size = 100000  # tokenizer will keep the top 100000 words
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

training_dictionary_list = []
training_sentences = []
training_labels = []

testing_dictionary_list = []
testing_sentences = []
testing_labels = []
fields = ['sentence', 'English', 'Luganda', 'Runyankole', 'Ateso', 'Lugbara', 'Acholi','Swahili']


Data = load_from_disk("SALT_SPLIT")
MT560 = load_from_disk('C:/Users/emike/MT560')

lgg = Data.filter(lambda ex: ex["src_lang"] == "lgg")
lug = Data.filter(lambda ex: ex["src_lang"] == "lug")
ach = Data.filter(lambda ex: ex["src_lang"] == "ach")
teo = Data.filter(lambda ex: ex["src_lang"] == "teo")
swa = Data.filter(lambda ex: ex["src_lang"] == "swa")
nyn = Data.filter(lambda ex: ex["src_lang"] == "nyn")

lugMT560 = MT560.filter(lambda ex: ex["src_lang"] == "lug")
swaMT560 = MT560.filter(lambda ex: ex["src_lang"] == "swa")

lug_length = len(lug["train"])
nyn_length = len(nyn["train"])
teo_length = len(teo["train"])
lgg_length = len(lgg["train"])
ach_length = len(ach["train"])

lug_lengthMT560 = len(lugMT560["train"])
swa_lengthMT560 = len(swaMT560["train"])

#print(lug_length,nyn_length,teo_length,lgg_length,ach_length)

# Add SALT training examples to dictionary list

for i in range(lug_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lug["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 1, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lug["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

for i in range(nyn_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': nyn["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 1, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': nyn["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

for i in range(teo_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': teo["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 1, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': teo["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

for i in range(lgg_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lgg["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 1, 'Acholi':0, 'Swahili':0,'sentence': lgg["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

for i in range(ach_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': ach["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':1, 'Swahili':0,'sentence': ach["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

# Add the MT560 data to training_dictionary_list

for i in range(lug_lengthMT560):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lugMT560["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 1, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lugMT560["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

for i in range(swa_lengthMT560):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': swaMT560["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':1,'sentence': swaMT560["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

print(len(training_dictionary_list))

training_sentences_and_labels = [[] for x in range(len(training_dictionary_list))]

counter = 0
for item in training_dictionary_list:
    training_sentences.append(item['sentence'])
    training_sentences_and_labels[counter].append(item['sentence'])
    training_labels.append([item['English'], item['Luganda'], item['Runyankole'],
                   item['Ateso'], item['Lugbara'], item['Acholi'], item['Swahili']])
    training_sentences_and_labels[counter].append(item['English'])
    training_sentences_and_labels[counter].append(item['Luganda'])
    training_sentences_and_labels[counter].append(item['Runyankole'])
    training_sentences_and_labels[counter].append(item['Ateso'])
    training_sentences_and_labels[counter].append(item['Lugbara'])
    training_sentences_and_labels[counter].append(item['Acholi'])
    training_sentences_and_labels[counter].append(item['Swahili'])
    counter += 1

print('sentences and labels length',len(training_sentences_and_labels))
print('sentences length',len(training_sentences))
print('labels length',len(training_labels))

# Create csv file to store training set
with open("SALT_and_MT560_train.csv", "w", encoding="utf-8",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(training_sentences_and_labels)

print(training_sentences_and_labels[10])
print(training_sentences[10])
print(training_labels[10])


# Carry out same process for test data

lug_length = len(lug["test"])
nyn_length = len(nyn["test"])
teo_length = len(teo["test"])
lgg_length = len(lgg["test"])
ach_length = len(ach["test"])

lug_lengthMT560 = len(lugMT560["test"])
swa_lengthMT560 = len(swaMT560["test"])

#print(lug_length,nyn_length,teo_length,lgg_length,ach_length)

# Add SALT training examples to dictionary list

for i in range(lug_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lug["test"][i]['English']}
    testing_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 1, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lug["test"][i]['src']}
    testing_dictionary_list.append(new_dict2)

for i in range(nyn_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': nyn["test"][i]['English']}
    testing_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 1, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': nyn["test"][i]['src']}
    testing_dictionary_list.append(new_dict2)

for i in range(teo_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': teo["test"][i]['English']}
    testing_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 1, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': teo["test"][i]['src']}
    testing_dictionary_list.append(new_dict2)

for i in range(lgg_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lgg["test"][i]['English']}
    testing_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 1, 'Acholi':0, 'Swahili':0,'sentence': lgg["test"][i]['src']}
    testing_dictionary_list.append(new_dict2)

for i in range(ach_length):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': ach["test"][i]['English']}
    testing_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':1, 'Swahili':0,'sentence': ach["test"][i]['src']}
    testing_dictionary_list.append(new_dict2)

# Add the MT560 data to training_dictionary_list

for i in range(lug_lengthMT560):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lugMT560["test"][i]['English']}
    testing_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 1, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lugMT560["test"][i]['src']}
    testing_dictionary_list.append(new_dict2)

for i in range(swa_lengthMT560):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': swaMT560["test"][i]['English']}
    testing_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':1,'sentence': swaMT560["test"][i]['src']}
    testing_dictionary_list.append(new_dict2)

print(len(testing_dictionary_list))

testing_sentences_and_labels = [[] for x in range(len(testing_dictionary_list))]

counter = 0
for item in testing_dictionary_list:
    testing_sentences.append(item['sentence'])
    testing_sentences_and_labels[counter].append(item['sentence'])
    testing_labels.append([item['English'], item['Luganda'], item['Runyankole'],
                   item['Ateso'], item['Lugbara'], item['Acholi'], item['Swahili']])
    testing_sentences_and_labels[counter].append(item['English'])
    testing_sentences_and_labels[counter].append(item['Luganda'])
    testing_sentences_and_labels[counter].append(item['Runyankole'])
    testing_sentences_and_labels[counter].append(item['Ateso'])
    testing_sentences_and_labels[counter].append(item['Lugbara'])
    testing_sentences_and_labels[counter].append(item['Acholi'])
    testing_sentences_and_labels[counter].append(item['Swahili'])
    counter += 1

print('sentences and labels length',len(testing_sentences_and_labels))
print('sentences length',len(testing_sentences))
print('labels length',len(testing_labels))

# Create csv file to store testing set
'''
with open("SALT_and_MT560_test.csv", "w", encoding="utf-8",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(testing_sentences_and_labels)'''

print(testing_sentences_and_labels[10])
print(testing_sentences[10])
print(testing_labels[10])

# Length of MT560 sentences is 1920556
# Length of SALT examples is 200050

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

with open('tokenizer.pickle', 'wb') as handle:
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
    tf.keras.layers.Dense(7, activation='softmax')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 20
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

model.save("lang_classifier_softmax2.h5")